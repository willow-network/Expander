// CUDA kernels for BATCHED M31ext3 sumcheck.
//
// Phase 7: pack many independent fold instances into a single GPU
// dispatch. Each fold gets its own slice of the device buffers,
// indexed by `blockIdx.y`. The kernel processes B = batch_size
// folds simultaneously, sharing kernel launch + memory transfer
// overhead across the batch — exactly the cost-amortization that
// the Phase 6 single-fold dispatch could not capture.
//
// Memory layout (AoS, fold-contiguous):
//   d_bk_f:  [batch][2*eval_size][3 u32] = 6 * eval_size * batch u32 total
//   d_bk_hg: same shape
//   d_result_per_fold: [batch][3 M31ext3] = 9 * batch u32 total
//
// Each fold b's data starts at byte offset b * 6 * eval_size * 4.

#include "m31_field_ops.cuh"

#include <stdio.h>

// Batched poly-eval kernel. blockIdx.y = fold index, blockIdx.x =
// chunk-of-pairs within the fold. Each block writes its partial
// [p0, p1, p2] result to d_block_results[fold][block_x][·].
__global__ void poly_eval_kernel_batched(
    const uint32_t* __restrict__ d_bk_f,
    const uint32_t* __restrict__ d_bk_hg,
    uint32_t* __restrict__ d_block_results,
    uint32_t eval_size,
    uint32_t fold_stride_u32  // u32s per fold in the buffer (constant across rounds)
) {
    int fold = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Per-fold pointer offsets — use FIXED stride, not eval_size,
    // because the buffer's per-fold span never shrinks even though
    // the active region within it does.
    const uint32_t* fold_f = d_bk_f + fold * fold_stride_u32;
    const uint32_t* fold_hg = d_bk_hg + fold * fold_stride_u32;
    uint32_t* fold_results = d_block_results + (fold * gridDim.x + blockIdx.x) * 9;

    extern __shared__ uint32_t smem[];
    uint32_t* s_p0 = smem;
    uint32_t* s_p1 = smem + blockDim.x * 3;
    uint32_t* s_p2 = smem + blockDim.x * 6;

    for (int l = 0; l < 3; l++) {
        s_p0[tid * 3 + l] = 0;
        s_p1[tid * 3 + l] = 0;
        s_p2[tid * 3 + l] = 0;
    }

    if (idx < (int)eval_size) {
        M31ext3 f_v_0, f_v_1;
        int base_f = idx * 6;
        f_v_0.v[0] = fold_f[base_f + 0];
        f_v_0.v[1] = fold_f[base_f + 1];
        f_v_0.v[2] = fold_f[base_f + 2];
        f_v_1.v[0] = fold_f[base_f + 3];
        f_v_1.v[1] = fold_f[base_f + 4];
        f_v_1.v[2] = fold_f[base_f + 5];

        M31ext3 hg_v_0, hg_v_1;
        int base_hg = idx * 6;
        hg_v_0.v[0] = fold_hg[base_hg + 0];
        hg_v_0.v[1] = fold_hg[base_hg + 1];
        hg_v_0.v[2] = fold_hg[base_hg + 2];
        hg_v_1.v[0] = fold_hg[base_hg + 3];
        hg_v_1.v[1] = fold_hg[base_hg + 4];
        hg_v_1.v[2] = fold_hg[base_hg + 5];

        M31ext3 lp0 = m31ext3_mul(hg_v_0, f_v_0);
        M31ext3 lp1 = m31ext3_mul(hg_v_1, f_v_1);
        M31ext3 lp2 = m31ext3_mul(
            m31ext3_add(hg_v_0, hg_v_1),
            m31ext3_add(f_v_0, f_v_1)
        );

        for (int l = 0; l < 3; l++) {
            s_p0[tid * 3 + l] = lp0.v[l];
            s_p1[tid * 3 + l] = lp1.v[l];
            s_p2[tid * 3 + l] = lp2.v[l];
        }
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int l = 0; l < 3; l++) {
                s_p0[tid * 3 + l] = m31_add(s_p0[tid * 3 + l], s_p0[(tid + stride) * 3 + l]);
                s_p1[tid * 3 + l] = m31_add(s_p1[tid * 3 + l], s_p1[(tid + stride) * 3 + l]);
                s_p2[tid * 3 + l] = m31_add(s_p2[tid * 3 + l], s_p2[(tid + stride) * 3 + l]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int l = 0; l < 3; l++) {
            fold_results[l]     = s_p0[l];
            fold_results[3 + l] = s_p1[l];
            fold_results[6 + l] = s_p2[l];
        }
    }
}

// Batched second-level reduction. Same structure: blockIdx.y = fold,
// blockIdx.x = chunk-of-blocks within fold's partial results.
__global__ void reduce_blocks_batched(
    const uint32_t* __restrict__ d_src,
    uint32_t* __restrict__ d_dst,
    uint32_t num_blocks_per_fold
) {
    int fold = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    const uint32_t* fold_src = d_src + fold * num_blocks_per_fold * 9;
    uint32_t* fold_dst = d_dst + (fold * gridDim.x + blockIdx.x) * 9;

    extern __shared__ uint32_t smem[];
    uint32_t* s_p0 = smem;
    uint32_t* s_p1 = smem + blockDim.x * 3;
    uint32_t* s_p2 = smem + blockDim.x * 6;

    for (int l = 0; l < 3; l++) {
        s_p0[tid * 3 + l] = 0;
        s_p1[tid * 3 + l] = 0;
        s_p2[tid * 3 + l] = 0;
    }

    if (idx < (int)num_blocks_per_fold) {
        int base = idx * 9;
        for (int l = 0; l < 3; l++) {
            s_p0[tid * 3 + l] = fold_src[base + l];
            s_p1[tid * 3 + l] = fold_src[base + 3 + l];
            s_p2[tid * 3 + l] = fold_src[base + 6 + l];
        }
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int l = 0; l < 3; l++) {
                s_p0[tid * 3 + l] = m31_add(s_p0[tid * 3 + l], s_p0[(tid + stride) * 3 + l]);
                s_p1[tid * 3 + l] = m31_add(s_p1[tid * 3 + l], s_p1[(tid + stride) * 3 + l]);
                s_p2[tid * 3 + l] = m31_add(s_p2[tid * 3 + l], s_p2[(tid + stride) * 3 + l]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int l = 0; l < 3; l++) {
            fold_dst[l]     = s_p0[l];
            fold_dst[3 + l] = s_p1[l];
            fold_dst[6 + l] = s_p2[l];
        }
    }
}

// Batched receive_challenge: each thread updates one (fold, pair)
// tuple. Each fold has its own challenge `r[fold]`.
__global__ void receive_challenge_kernel_batched(
    uint32_t* __restrict__ d_bk_f,
    uint32_t* __restrict__ d_bk_hg,
    const uint32_t* __restrict__ d_r,   // [batch][3 u32]
    uint32_t eval_size,
    uint32_t fold_stride_u32
) {
    int fold = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)eval_size) return;

    M31ext3 r;
    r.v[0] = d_r[fold * 3 + 0];
    r.v[1] = d_r[fold * 3 + 1];
    r.v[2] = d_r[fold * 3 + 2];

    uint32_t* fold_f = d_bk_f + fold * fold_stride_u32;
    uint32_t* fold_hg = d_bk_hg + fold * fold_stride_u32;

    {
        int base = i * 6;
        M31ext3 f0, f1;
        f0.v[0] = fold_f[base + 0]; f0.v[1] = fold_f[base + 1]; f0.v[2] = fold_f[base + 2];
        f1.v[0] = fold_f[base + 3]; f1.v[1] = fold_f[base + 4]; f1.v[2] = fold_f[base + 5];

        M31ext3 diff = m31ext3_sub(f1, f0);
        M31ext3 scaled = m31ext3_mul(diff, r);
        M31ext3 result = m31ext3_add(f0, scaled);

        int out = i * 3;
        fold_f[out + 0] = result.v[0];
        fold_f[out + 1] = result.v[1];
        fold_f[out + 2] = result.v[2];
    }

    {
        int base = i * 6;
        M31ext3 h0, h1;
        h0.v[0] = fold_hg[base + 0]; h0.v[1] = fold_hg[base + 1]; h0.v[2] = fold_hg[base + 2];
        h1.v[0] = fold_hg[base + 3]; h1.v[1] = fold_hg[base + 4]; h1.v[2] = fold_hg[base + 5];

        M31ext3 diff = m31ext3_sub(h1, h0);
        M31ext3 scaled = m31ext3_mul(diff, r);
        M31ext3 result = m31ext3_add(h0, scaled);

        int out = i * 3;
        fold_hg[out + 0] = result.v[0];
        fold_hg[out + 1] = result.v[1];
        fold_hg[out + 2] = result.v[2];
    }
}

// ============================================================================
// Host wrappers
// ============================================================================

#define SUMCHECK_BLOCK_SIZE_BATCHED 256

// Run batched poly_eval. Output is `batch * 9 u32` (3 M31ext3 per fold).
extern "C" int cuda_m31ext3_poly_eval_batched(
    const uint32_t* d_bk_f,
    const uint32_t* d_bk_hg,
    uint32_t* d_result,
    uint32_t eval_size,
    uint32_t batch_size,
    uint32_t fold_stride_u32
) {
    if (eval_size == 0 || batch_size == 0) {
        cudaError_t err = cudaMemset(d_result, 0, batch_size * 9 * sizeof(uint32_t));
        return (err == cudaSuccess) ? 0 : (int)err;
    }

    int block_size = SUMCHECK_BLOCK_SIZE_BATCHED;
    int blocks_per_fold = (eval_size + block_size - 1) / block_size;
    size_t smem_size = 9 * block_size * sizeof(uint32_t);

    uint32_t* d_block_results = nullptr;
    cudaError_t err = cudaMalloc(
        &d_block_results,
        batch_size * blocks_per_fold * 9 * sizeof(uint32_t)
    );
    if (err != cudaSuccess) return (int)err;

    dim3 grid_first(blocks_per_fold, batch_size);
    poly_eval_kernel_batched<<<grid_first, block_size, smem_size>>>(
        d_bk_f, d_bk_hg, d_block_results, eval_size, fold_stride_u32
    );

    // Iteratively reduce blocks-per-fold until 1 block per fold.
    uint32_t* d_reduce_src = d_block_results;
    uint32_t* d_reduce_dst = nullptr;
    int current_blocks = blocks_per_fold;

    while (current_blocks > 1) {
        int next_blocks = (current_blocks + block_size - 1) / block_size;
        err = cudaMalloc(
            &d_reduce_dst,
            batch_size * next_blocks * 9 * sizeof(uint32_t)
        );
        if (err != cudaSuccess) {
            cudaFree(d_block_results);
            return (int)err;
        }

        dim3 grid_reduce(next_blocks, batch_size);
        reduce_blocks_batched<<<grid_reduce, block_size, smem_size>>>(
            d_reduce_src, d_reduce_dst, current_blocks
        );

        if (d_reduce_src != d_block_results) {
            cudaFree(d_reduce_src);
        }
        d_reduce_src = d_reduce_dst;
        d_reduce_dst = nullptr;
        current_blocks = next_blocks;
    }

    err = cudaMemcpy(
        d_result, d_reduce_src,
        batch_size * 9 * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice
    );

    if (d_reduce_src != d_block_results) {
        cudaFree(d_reduce_src);
    }
    cudaFree(d_block_results);
    return (err == cudaSuccess) ? 0 : (int)err;
}

extern "C" int cuda_m31ext3_receive_challenge_batched(
    uint32_t* d_bk_f,
    uint32_t* d_bk_hg,
    const uint32_t* d_challenge_r,  // [batch][3 u32]
    uint32_t eval_size,
    uint32_t batch_size,
    uint32_t fold_stride_u32
) {
    int block_size = SUMCHECK_BLOCK_SIZE_BATCHED;
    int blocks_per_fold = (eval_size + block_size - 1) / block_size;

    dim3 grid(blocks_per_fold, batch_size);
    receive_challenge_kernel_batched<<<grid, block_size>>>(
        d_bk_f, d_bk_hg, d_challenge_r, eval_size, fold_stride_u32
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}
