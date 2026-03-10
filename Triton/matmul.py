import triton
import triton.language as tl
import torch

# Tuning constants
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
GROUP_M = 8

@triton.jit
def matmul_kernel_complex(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Advanced tiled matrix multiplication with:
    - 2D block tiling (M x N)
    - K-dimension reduction loop
    - Automatic memory coalescing
    - Grouped thread blocks for better occupancy
    """
    
    # Program ID mapping with grouping strategy
    pid = tl.program_id(0)
    pid_m = pid // GROUP_M
    pid_n = pid % GROUP_M
    
    # Compute block offsets
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    k_offset = tl.arange(0, BLOCK_K)
    
    # Pointers to first element of A and B blocks
    a_ptrs = a_ptr + (m_offset * stride_am + k_offset[None, :] * stride_ak)
    b_ptrs = b_ptr + (k_offset[:, None] * stride_bk + n_offset * stride_bn)
    
    # Accumulator for matrix multiplication
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main computation loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles from global memory into registers
        # Triton auto-optimizes coalescing and caching
        a_tile = tl.load(a_ptrs, mask=(m_offset < M)[:, None] & (k + k_offset[None, :] < K))
        b_tile = tl.load(b_ptrs, mask=(k + k_offset[:, None] < K) & (n_offset < N)[None, :])
        
        # Perform block matrix multiplication
        # a_tile: (BLOCK_M, BLOCK_K), b_tile: (BLOCK_K, BLOCK_N)
        accumulator += tl.dot(a_tile, b_tile)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Convert accumulator to output dtype (optional quantization)
    c_tile = accumulator.to(tl.float32)
    
    # Compute output pointers
    c_ptrs = c_ptr + (m_offset * stride_cm + n_offset * stride_cn)
    
    # Store result back to global memory with boundary checks
    tl.store(c_ptrs, c_tile, mask=(m_offset < M)[:, None] & (n_offset < N)[None, :])


@triton.jit
def matmul_kernel_fused(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel with:
    - Matrix multiplication
    - Bias addition
    - Scaling (alpha * A @ B + beta * bias)
    - ReLU activation
    """
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    
    a_ptrs = a_ptr + (m_offset * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + n_offset * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K-reduction loop with manual tiling
    for k in range(0, K, BLOCK_K):
        # Masked load with conditional mask
        mask_k = (k + tl.arange(0, BLOCK_K)[None, :] < K)
        mask_m = (m_offset < M)[:, None]
        a_tile = tl.load(a_ptrs, mask=mask_m & mask_k, other=0.0)
        
        mask_k = (k + tl.arange(0, BLOCK_K)[:, None] < K)
        mask_n = (n_offset < N)[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_k & mask_n, other=0.0)
        
        accumulator += tl.dot(a_tile, b_tile)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Scale by alpha
    accumulator = accumulator * alpha
    
    # Load bias and add
    bias_ptrs = bias_ptr + n_offset
    bias = tl.load(bias_ptrs, mask=(n_offset < N)[None, :], other=0.0)
    accumulator = accumulator + beta * bias
    
    # Apply ReLU activation
    accumulator = tl.maximum(accumulator, 0.0)
    
    # Store output
    c_ptrs = c_ptr + (m_offset * stride_cm + n_offset * stride_cn)
    tl.store(c_ptrs, accumulator.to(tl.float32), 
             mask=(m_offset < M)[:, None] & (n_offset < N)[None, :])


def matmul(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Wrapper function to launch Triton kernel
    Supports both basic and fused variants
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    
    M, K = a.shape
    K, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    if bias is not None:
        matmul_kernel_fused[grid](
            a, b, c, bias,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            bias.stride(0),
            alpha=1.0, beta=1.0,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    else:
        matmul_kernel_complex[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
        )
    
    return c


# Test
if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((1024, 512), device='cuda', dtype=torch.float32)
    b = torch.randn((512, 1024), device='cuda', dtype=torch.float32)
    bias = torch.randn((1024,), device='cuda', dtype=torch.float32)
    
    c_triton = matmul(a, b, bias)
    c_torch = torch.matmul(a, b) + bias.unsqueeze(0)
    
    print(f"Max error: {(c_triton - c_torch).abs().max().item()}")