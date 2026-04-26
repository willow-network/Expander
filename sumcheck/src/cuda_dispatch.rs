//! Safe wrappers around CUDA sumcheck kernels with threshold checks and CPU fallback.
//!
//! The GPU dispatch is transparent to the caller: if CUDA is not available or the
//! input is too small, the functions return `None` and the caller falls through to
//! the existing CPU implementation.

#[cfg(feature = "cuda")]
use crate::cuda_ffi;

/// Minimum eval_size to dispatch to GPU. Below this threshold, CPU is faster
/// due to kernel launch overhead and PCIe transfer costs.
pub const GPU_DISPATCH_THRESHOLD: usize = 1024;

/// Result of a GPU poly_eval_at call.
/// Contains [p0, p1, p2] as raw u32 limbs (3 limbs per M31ext3 element = 9 u32 total).
#[cfg(feature = "cuda")]
pub struct GpuPolyEvalResult {
    pub p0_limbs: [u32; 3],
    pub p1_limbs: [u32; 3],
    pub p2_limbs: [u32; 3],
}

/// Try to run poly_eval_at on GPU. Returns None if CUDA is unavailable,
/// eval_size is below threshold, or an error occurs (in which case the
/// caller should fall back to CPU).
///
/// # Safety
/// All device pointers must be valid CUDA device memory of the correct size.
#[cfg(feature = "cuda")]
pub unsafe fn try_gpu_poly_eval(
    d_bk_f: *const u32,
    d_bk_hg: *const u32,
    eval_size: usize,
) -> Option<GpuPolyEvalResult> {
    if eval_size < GPU_DISPATCH_THRESHOLD {
        return None;
    }

    // Allocate device memory for result (9 u32)
    let mut d_result: *mut u32 = std::ptr::null_mut();
    let alloc_err = cuda_malloc(&mut d_result, 9 * std::mem::size_of::<u32>());
    if alloc_err != 0 || d_result.is_null() {
        return None;
    }

    let err = cuda_ffi::cuda_m31ext3_poly_eval(d_bk_f, d_bk_hg, d_result, eval_size as u32);

    if err != 0 {
        cuda_free(d_result as *mut std::ffi::c_void);
        return None;
    }

    // Copy result back to host
    let mut result = [0u32; 9];
    let copy_err = cuda_memcpy_d2h(
        result.as_mut_ptr() as *mut std::ffi::c_void,
        d_result as *const std::ffi::c_void,
        9 * std::mem::size_of::<u32>(),
    );

    cuda_free(d_result as *mut std::ffi::c_void);

    if copy_err != 0 {
        return None;
    }

    Some(GpuPolyEvalResult {
        p0_limbs: [result[0], result[1], result[2]],
        p1_limbs: [result[3], result[4], result[5]],
        p2_limbs: [result[6], result[7], result[8]],
    })
}

/// Try to run receive_challenge on GPU.
///
/// # Safety
/// All device pointers must be valid CUDA device memory.
#[cfg(feature = "cuda")]
pub unsafe fn try_gpu_receive_challenge(
    d_bk_f: *mut u32,
    d_bk_hg: *mut u32,
    d_challenge_r: *const u32,
    eval_size: usize,
    first_round: bool,
    d_init_v: *const u32,
) -> bool {
    if eval_size < GPU_DISPATCH_THRESHOLD {
        return false;
    }

    let err = cuda_ffi::cuda_m31ext3_receive_challenge(
        d_bk_f,
        d_bk_hg,
        d_challenge_r,
        eval_size as u32,
        if first_round { 1 } else { 0 },
        d_init_v,
    );

    err == 0
}

// Minimal CUDA runtime bindings for memory management
#[cfg(feature = "cuda")]
extern "C" {
    #[link_name = "cudaMalloc"]
    fn cuda_malloc(devptr: *mut *mut u32, size: usize) -> i32;

    #[link_name = "cudaFree"]
    fn cuda_free(devptr: *mut std::ffi::c_void) -> i32;

    #[link_name = "cudaMemcpy"]
    fn cuda_memcpy_raw(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
}

#[cfg(feature = "cuda")]
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

#[cfg(feature = "cuda")]
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

#[cfg(feature = "cuda")]
unsafe fn cuda_memcpy_d2h(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    count: usize,
) -> i32 {
    cuda_memcpy_raw(dst, src, count, CUDA_MEMCPY_DEVICE_TO_HOST)
}

#[cfg(feature = "cuda")]
unsafe fn cuda_memcpy_h2d(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    count: usize,
) -> i32 {
    cuda_memcpy_raw(dst, src, count, CUDA_MEMCPY_HOST_TO_DEVICE)
}

// ---------------------------------------------------------------------------
// RAII wrapper for CUDA device memory
// ---------------------------------------------------------------------------

/// Thin RAII wrapper around a `cudaMalloc`-ed pointer.  Frees on drop, and
/// provides `into_raw()` so ownership can be transferred to a longer-lived
/// struct without triggering the destructor.
#[cfg(feature = "cuda")]
struct DevicePtr(*mut u32);

#[cfg(feature = "cuda")]
impl DevicePtr {
    fn alloc(bytes: usize) -> Option<Self> {
        let mut ptr: *mut u32 = std::ptr::null_mut();
        if unsafe { cuda_malloc(&mut ptr, bytes) } != 0 || ptr.is_null() {
            return None;
        }
        Some(DevicePtr(ptr))
    }

    fn ptr(&self) -> *mut u32 {
        self.0
    }

    /// Transfer ownership of the raw pointer out of this wrapper.
    /// The caller is now responsible for calling `cudaFree`.
    fn into_raw(self) -> *mut u32 {
        let p = self.0;
        std::mem::forget(self);
        p
    }
}

#[cfg(feature = "cuda")]
impl Drop for DevicePtr {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                cuda_free(self.0 as *mut std::ffi::c_void);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SoA  ↔  AoS conversion utilities
// ---------------------------------------------------------------------------
//
// Rust SIMD types use Structure-of-Arrays (SoA) layout:
//   M31Ext3x16 = [M31x16; 3] → [[u32; PACK]; 3]
//   In memory:  [PACK u32 for limb0] [PACK u32 for limb1] [PACK u32 for limb2]
//   Total: 3 * PACK u32 per element  (48 for PACK=16).
//
// CUDA kernels use Array-of-Structures (AoS), one lane at a time:
//   Per lane: contiguous M31Ext3 values, each 3 u32.
//   16 lanes stored back-to-back: [lane0 segment] [lane1 segment] …

/// Convert `num_elems` M31Ext3x16 values (SoA) → per-lane M31Ext3 (AoS).
///
/// `dst` must hold `pack_size * num_elems * 3` u32.
#[cfg(feature = "cuda")]
pub unsafe fn convert_simd_ext3_to_aos(
    src: *const u32,
    dst: *mut u32,
    num_elems: usize,
    pack_size: usize,
) {
    let limbs = 3usize;
    let src_elem_stride = pack_size * limbs; // 48 u32 per M31Ext3x16
    let dst_lane_stride = num_elems * limbs; // u32 per lane segment in output

    for k in 0..num_elems {
        let src_base = k * src_elem_stride;
        for lane in 0..pack_size {
            let dst_base = lane * dst_lane_stride + k * limbs;
            for limb in 0..limbs {
                *dst.add(dst_base + limb) = *src.add(src_base + limb * pack_size + lane);
            }
        }
    }
}

/// Convert per-lane M31Ext3 (AoS) → M31Ext3x16 (SoA).
///
/// `dst` must hold `num_elems` M31Ext3x16 values = `num_elems * pack_size * 3` u32.
#[cfg(feature = "cuda")]
pub unsafe fn convert_aos_to_simd_ext3(
    src: *const u32,
    dst: *mut u32,
    num_elems: usize,
    pack_size: usize,
) {
    let limbs = 3usize;
    let dst_elem_stride = pack_size * limbs;
    let src_lane_stride = num_elems * limbs;

    for k in 0..num_elems {
        let dst_base = k * dst_elem_stride;
        for lane in 0..pack_size {
            let src_base = lane * src_lane_stride + k * limbs;
            for limb in 0..limbs {
                *dst.add(dst_base + limb * pack_size + lane) = *src.add(src_base + limb);
            }
        }
    }
}

/// Convert `num_elems` M31x16 values (SoA, base field) → per-lane scalar M31 (AoS).
///
/// `dst` must hold `pack_size * num_elems` u32.
#[cfg(feature = "cuda")]
pub unsafe fn convert_simd_base_to_aos(
    src: *const u32,
    dst: *mut u32,
    num_elems: usize,
    pack_size: usize,
) {
    let src_elem_stride = pack_size; // 16 u32 per M31x16
    let dst_lane_stride = num_elems;

    for k in 0..num_elems {
        for lane in 0..pack_size {
            *dst.add(lane * dst_lane_stride + k) = *src.add(k * src_elem_stride + lane);
        }
    }
}

/// Convert M31x16 base-field data → promoted M31Ext3 AoS.
/// Each M31 value `v` becomes `[v, 0, 0]` in extension representation.
///
/// `dst` must hold `pack_size * num_elems * 3` u32.
#[cfg(feature = "cuda")]
pub unsafe fn convert_simd_base_to_promoted_ext3_aos(
    src: *const u32,
    dst: *mut u32,
    num_elems: usize,
    pack_size: usize,
) {
    let src_elem_stride = pack_size;
    let dst_lane_stride = num_elems * 3;

    // Zero everything first (extension limbs 1, 2 will stay zero).
    std::ptr::write_bytes(dst, 0, pack_size * dst_lane_stride);

    for k in 0..num_elems {
        for lane in 0..pack_size {
            let val = *src.add(k * src_elem_stride + lane);
            *dst.add(lane * dst_lane_stride + k * 3) = val; // limb 0
        }
    }
}

// ---------------------------------------------------------------------------
// GpuSumcheckContext — persistent device buffers for sumcheck proving
// ---------------------------------------------------------------------------

/// Persistent GPU buffers that live across sumcheck rounds within one phase.
///
/// Data is stored in per-lane AoS format: 16 contiguous segments of scalar
/// M31Ext3 values.  The context is created at the start of each sumcheck
/// phase (after `prepare_x_vals` / `prepare_y_vals`) and dropped or
/// transitioned back to CPU when `eval_size` falls below the dispatch
/// threshold.
#[cfg(feature = "cuda")]
pub struct GpuSumcheckContext {
    /// Bookkeeping table for f polynomial (M31Ext3, AoS, per-lane).
    d_bk_f: *mut u32,
    /// Bookkeeping table for hg polynomial (M31Ext3, AoS, per-lane).
    d_bk_hg: *mut u32,
    /// Ping-pong output buffers for `receive_challenge`. Same size as
    /// `d_bk_f`/`d_bk_hg`. After each non-first-round receive_challenge
    /// kernel call, `swap()` is called so the freshly-written buffers
    /// become the next round's input.
    d_bk_f_alt: *mut u32,
    d_bk_hg_alt: *mut u32,
    /// Initial input values for first-round receive_challenge (M31 base, AoS, per-lane).
    d_init_v: *mut u32,
    /// Small buffer for the challenge scalar (3 u32 = one M31Ext3).
    d_challenge: *mut u32,
    /// Small buffer for one poly_eval reduction result (9 u32).
    d_result: *mut u32,

    pack_size: usize,
    alloc_elems: usize,
    /// u32 stride between consecutive lane segments for M31Ext3 data.
    lane_stride_ext3: usize,
    /// u32 stride between consecutive lane segments for M31 base data.
    lane_stride_base: usize,
}

#[cfg(feature = "cuda")]
impl GpuSumcheckContext {
    /// Allocate device memory and upload converted data.
    ///
    /// # Safety
    ///
    /// `hg_evals_ptr` must point to `input_size` contiguous M31Ext3x16 values
    /// (i.e. `input_size * 48` u32 in SoA layout).
    ///
    /// `init_v_ptr` must point to `input_size` contiguous M31x16 values
    /// (i.e. `input_size * 16` u32 in SoA layout).
    pub unsafe fn new(
        hg_evals_ptr: *const u32,
        init_v_ptr: *const u32,
        pack_size: usize,
        input_size: usize,
    ) -> Option<Self> {
        let lane_stride_ext3 = input_size * 3;
        let lane_stride_base = input_size;

        let ext3_total_u32 = pack_size * lane_stride_ext3;
        let base_total_u32 = pack_size * lane_stride_base;
        let ext3_bytes = ext3_total_u32 * std::mem::size_of::<u32>();
        let base_bytes = base_total_u32 * std::mem::size_of::<u32>();

        // ---- Host-side SoA → AoS conversion ----

        let mut hg_aos = vec![0u32; ext3_total_u32];
        convert_simd_ext3_to_aos(hg_evals_ptr, hg_aos.as_mut_ptr(), input_size, pack_size);

        // For round 0's poly_eval, the kernel reads d_bk_f but the CPU code
        // reads init_v (base field).  We promote init_v to M31Ext3 ([v,0,0])
        // so the kernel can read d_bk_f uniformly for all rounds.
        let mut bk_f_aos = vec![0u32; ext3_total_u32];
        convert_simd_base_to_promoted_ext3_aos(
            init_v_ptr,
            bk_f_aos.as_mut_ptr(),
            input_size,
            pack_size,
        );

        // receive_challenge round 0 needs base-field init_v separately.
        let mut init_v_aos = vec![0u32; base_total_u32];
        convert_simd_base_to_aos(init_v_ptr, init_v_aos.as_mut_ptr(), input_size, pack_size);

        // ---- Allocate device memory (DevicePtr frees on early return) ----

        let d_bk_f = DevicePtr::alloc(ext3_bytes)?;
        let d_bk_hg = DevicePtr::alloc(ext3_bytes)?;
        // Ping-pong output for receive_challenge. Same size as bk_f / bk_hg.
        let d_bk_f_alt = DevicePtr::alloc(ext3_bytes)?;
        let d_bk_hg_alt = DevicePtr::alloc(ext3_bytes)?;
        let d_init_v = DevicePtr::alloc(base_bytes)?;
        // d_challenge holds the (replicated) per-lane challenge for the
        // batched receive_challenge dispatch: pack_size * 3 u32.
        let d_challenge = DevicePtr::alloc(pack_size * 3 * 4)?;
        // d_result holds the batched poly_eval output: pack_size * 9 u32.
        let d_result = DevicePtr::alloc(pack_size * 9 * 4)?;

        // ---- Upload ----

        if cuda_memcpy_h2d(d_bk_f.ptr() as _, bk_f_aos.as_ptr() as _, ext3_bytes) != 0 {
            return None;
        }
        if cuda_memcpy_h2d(d_bk_hg.ptr() as _, hg_aos.as_ptr() as _, ext3_bytes) != 0 {
            return None;
        }
        if cuda_memcpy_h2d(d_init_v.ptr() as _, init_v_aos.as_ptr() as _, base_bytes) != 0 {
            return None;
        }

        Some(GpuSumcheckContext {
            d_bk_f: d_bk_f.into_raw(),
            d_bk_hg: d_bk_hg.into_raw(),
            d_bk_f_alt: d_bk_f_alt.into_raw(),
            d_bk_hg_alt: d_bk_hg_alt.into_raw(),
            d_init_v: d_init_v.into_raw(),
            d_challenge: d_challenge.into_raw(),
            d_result: d_result.into_raw(),
            pack_size,
            alloc_elems: input_size,
            lane_stride_ext3,
            lane_stride_base,
        })
    }

    /// Whether `eval_size` is large enough to benefit from GPU dispatch.
    pub fn should_use_gpu(&self, eval_size: usize) -> bool {
        eval_size >= GPU_DISPATCH_THRESHOLD
    }

    /// Run `poly_eval_at` on GPU for all SIMD lanes.
    ///
    /// Returns `result[lane] = [p0_l0, p0_l1, p0_l2, p1_l0, …, p2_l2]`
    /// (9 u32 per lane: 3 limbs × 3 polynomial components).
    ///
    /// Implementation: a single batched kernel dispatch with the SIMD
    /// pack treated as the batch dimension. Replaces the legacy
    /// lane-sequential loop, which paid the ~10–50 µs CUDA launch
    /// overhead 16× per round and left the GPU at ~1% utilization.
    /// One batched dispatch + one D2H copy of `pack_size * 9 u32`.
    ///
    /// # Safety
    ///
    /// Device pointers must still be valid (context not dropped).
    pub unsafe fn poly_eval_at(&self, eval_size: usize) -> Option<[[u32; 9]; 16]> {
        debug_assert!(self.pack_size <= 16);
        let mut results = [[0u32; 9]; 16];

        let err = cuda_ffi::cuda_m31ext3_poly_eval_batched(
            self.d_bk_f as *const u32,
            self.d_bk_hg as *const u32,
            self.d_result,
            eval_size as u32,
            self.pack_size as u32,
            self.lane_stride_ext3 as u32,
        );
        if err != 0 {
            return None;
        }

        // Single D2H copy of the whole batched result block.
        let mut packed = vec![0u32; self.pack_size * 9];
        if cuda_memcpy_d2h(
            packed.as_mut_ptr() as *mut std::ffi::c_void,
            self.d_result as *const std::ffi::c_void,
            self.pack_size * 9 * 4,
        ) != 0
        {
            return None;
        }
        for lane in 0..self.pack_size {
            results[lane].copy_from_slice(&packed[lane * 9..lane * 9 + 9]);
        }

        Some(results)
    }

    /// Run `receive_challenge` on GPU for all SIMD lanes.
    ///
    /// Non-first rounds use a single batched kernel dispatch with the
    /// challenge `r` replicated across `pack_size` lanes. First round
    /// (`var_idx == 0`) reads from `init_v` (M31 base field) for an
    /// optimization where the first multiplication is base*ext3 instead
    /// of ext3*ext3 — that path needs the per-lane variant and the
    /// batched kernel doesn't support base-field input yet, so first
    /// round falls back to the lane-sequential path. First round
    /// happens once per phase out of log_n rounds, so the bulk of the
    /// dispatches still benefit from batching.
    ///
    /// # Safety
    ///
    /// Device pointers must still be valid.
    pub unsafe fn receive_challenge(
        &mut self,
        eval_size: usize,
        var_idx: usize,
        r_limbs: &[u32; 3],
    ) -> bool {
        if var_idx == 0 {
            // First-round path: lane-sequential with init_v.
            // Upload r once; the kernel reads it as a single 3-u32 value.
            if cuda_memcpy_h2d(
                self.d_challenge as *mut std::ffi::c_void,
                r_limbs.as_ptr() as *const std::ffi::c_void,
                3 * 4,
            ) != 0
            {
                return false;
            }
            for lane in 0..self.pack_size {
                let f_ptr = self.d_bk_f.add(lane * self.lane_stride_ext3);
                let hg_ptr = self.d_bk_hg.add(lane * self.lane_stride_ext3);
                let init_v_ptr = self.d_init_v.add(lane * self.lane_stride_base) as *const u32;

                let err = cuda_ffi::cuda_m31ext3_receive_challenge(
                    f_ptr,
                    hg_ptr,
                    self.d_challenge as *const u32,
                    eval_size as u32,
                    1,
                    init_v_ptr,
                );
                if err != 0 {
                    return false;
                }
            }
            return true;
        }

        // Non-first round: batched dispatch with replicated r.
        let mut replicated = vec![0u32; self.pack_size * 3];
        for lane in 0..self.pack_size {
            replicated[lane * 3..lane * 3 + 3].copy_from_slice(r_limbs);
        }
        if cuda_memcpy_h2d(
            self.d_challenge as *mut std::ffi::c_void,
            replicated.as_ptr() as *const std::ffi::c_void,
            self.pack_size * 3 * 4,
        ) != 0
        {
            return false;
        }

        let err = cuda_ffi::cuda_m31ext3_receive_challenge_batched(
            self.d_bk_f as *const u32,
            self.d_bk_hg as *const u32,
            self.d_bk_f_alt,
            self.d_bk_hg_alt,
            self.d_challenge as *const u32,
            eval_size as u32,
            self.pack_size as u32,
            self.lane_stride_ext3 as u32,
        );
        if err != 0 {
            return false;
        }
        // Ping-pong: the freshly-written buffers become the next round's
        // input. The kernel guarantees no aliasing between input and
        // output, so subsequent poly_eval reads from `d_bk_f` (now the
        // new data) cleanly.
        std::mem::swap(&mut self.d_bk_f, &mut self.d_bk_f_alt);
        std::mem::swap(&mut self.d_bk_hg, &mut self.d_bk_hg_alt);
        true
    }

    /// Download `bk_f` from device and convert AoS → SoA back into host memory.
    ///
    /// `dst` must point to at least `num_elems` M31Ext3x16 values.
    ///
    /// # Safety
    ///
    /// Device pointers must still be valid.  `dst` must be large enough.
    pub unsafe fn download_bk_f(&self, dst: *mut u32, num_elems: usize) -> bool {
        self.download_ext3_buffer(self.d_bk_f as *const u32, dst, num_elems)
    }

    /// Download `bk_hg` from device and convert AoS → SoA back into host memory.
    pub unsafe fn download_bk_hg(&self, dst: *mut u32, num_elems: usize) -> bool {
        self.download_ext3_buffer(self.d_bk_hg as *const u32, dst, num_elems)
    }

    unsafe fn download_ext3_buffer(
        &self,
        d_src: *const u32,
        dst: *mut u32,
        num_elems: usize,
    ) -> bool {
        // We only need to download the *active* portion of each lane.
        // Each lane's active data starts at `lane * lane_stride_ext3` and
        // contains `num_elems * 3` u32.
        let per_lane_u32 = num_elems * 3;
        let total_u32 = self.pack_size * per_lane_u32;
        let mut host_aos = vec![0u32; total_u32];

        for lane in 0..self.pack_size {
            let src_offset = lane * self.lane_stride_ext3;
            let dst_offset = lane * per_lane_u32;
            if cuda_memcpy_d2h(
                host_aos.as_mut_ptr().add(dst_offset) as *mut std::ffi::c_void,
                d_src.add(src_offset) as *const std::ffi::c_void,
                per_lane_u32 * 4,
            ) != 0
            {
                return false;
            }
        }

        convert_aos_to_simd_ext3(host_aos.as_ptr(), dst, num_elems, self.pack_size);
        true
    }
}

#[cfg(feature = "cuda")]
impl Drop for GpuSumcheckContext {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.d_bk_f as *mut std::ffi::c_void);
            cuda_free(self.d_bk_hg as *mut std::ffi::c_void);
            cuda_free(self.d_bk_f_alt as *mut std::ffi::c_void);
            cuda_free(self.d_bk_hg_alt as *mut std::ffi::c_void);
            cuda_free(self.d_init_v as *mut std::ffi::c_void);
            cuda_free(self.d_challenge as *mut std::ffi::c_void);
            cuda_free(self.d_result as *mut std::ffi::c_void);
        }
    }
}

// ---------------------------------------------------------------------------
// BatchedGpuSumcheckContext — N independent circuit proves, batched into one
// kernel dispatch per round. Drop-in extension of GpuSumcheckContext for
// the cross-instance batching architecture (see Phase 7 / completeness scaling
// bench). At each layer's poly_eval / receive_challenge, total batch dim is
// `n_instances * pack_size`, so 256 instances × 16 SIMD lanes = 4096 effective
// folds in one kernel call. Per-fold cost stays roughly constant across
// batch-size at small/medium eval_size — see batched_dispatch_scaling bench.
// ---------------------------------------------------------------------------

/// Cross-instance batched GPU sumcheck context.
///
/// Holds N independent circuit witnesses in one mega-buffer and dispatches
/// every kernel with `batch_size = n_instances * pack_size` and
/// `fold_stride_u32 = lane_stride_ext3` (one "fold" per SIMD lane, all
/// stacked contiguously). Each instance's transcript is independent so
/// each lane group of 16 may receive a different challenge `r` per round;
/// the kernel takes per-fold challenges, so we replicate each instance's
/// `r` 16 times (once per pack lane) before upload.
#[cfg(feature = "cuda")]
pub struct BatchedGpuSumcheckContext {
    /// Stacked bookkeeping for f polynomial across N instances.
    /// Layout: instance_0 [16 lanes × lane_stride_ext3 u32]
    ///       | instance_1 [16 lanes × lane_stride_ext3 u32]
    ///       | …
    d_bk_f: *mut u32,
    /// Same shape for hg.
    d_bk_hg: *mut u32,
    /// Ping-pong output buffers for receive_challenge — same architecture
    /// as GpuSumcheckContext (race-free non-in-place kernel write).
    d_bk_f_alt: *mut u32,
    d_bk_hg_alt: *mut u32,
    /// Same shape but base field (1 u32 per element per lane).
    d_init_v: *mut u32,
    /// Per-lane challenge buffer: `n_instances * pack_size * 3` u32.
    /// Within an instance, r is replicated across all `pack_size` lanes.
    d_challenge: *mut u32,
    /// Per-lane poly_eval result: `n_instances * pack_size * 9` u32.
    d_result: *mut u32,

    n_instances: usize,
    pack_size: usize,
    /// u32 stride per lane segment (same as GpuSumcheckContext).
    lane_stride_ext3: usize,
    lane_stride_base: usize,
}

#[cfg(feature = "cuda")]
impl BatchedGpuSumcheckContext {
    /// Allocate device memory for `n_instances` independent provers and
    /// upload their initial state.
    ///
    /// `hg_evals_ptrs[i]` and `init_v_ptrs[i]` point to instance `i`'s
    /// host data in the same SoA-x16 layout `GpuSumcheckContext::new`
    /// accepts. All instances must share `pack_size` and `input_size`.
    ///
    /// # Safety
    ///
    /// Each `hg_evals_ptrs[i]` must point to `input_size` contiguous
    /// M31Ext3x16 values (`input_size * 48` u32). Each `init_v_ptrs[i]`
    /// must point to `input_size` contiguous M31x16 values
    /// (`input_size * 16` u32).
    pub unsafe fn new(
        hg_evals_ptrs: &[*const u32],
        init_v_ptrs: &[*const u32],
        pack_size: usize,
        input_size: usize,
    ) -> Option<Self> {
        let n_instances = hg_evals_ptrs.len();
        if n_instances == 0 || init_v_ptrs.len() != n_instances {
            return None;
        }

        let lane_stride_ext3 = input_size * 3;
        let lane_stride_base = input_size;

        let per_instance_ext3_u32 = pack_size * lane_stride_ext3;
        let per_instance_base_u32 = pack_size * lane_stride_base;
        let total_ext3_u32 = n_instances * per_instance_ext3_u32;
        let total_base_u32 = n_instances * per_instance_base_u32;

        let ext3_bytes = total_ext3_u32 * std::mem::size_of::<u32>();
        let base_bytes = total_base_u32 * std::mem::size_of::<u32>();

        // Build host-side stacked buffers, one instance after another.
        let mut hg_aos = vec![0u32; total_ext3_u32];
        let mut bk_f_aos = vec![0u32; total_ext3_u32];
        let mut init_v_aos = vec![0u32; total_base_u32];

        for i in 0..n_instances {
            let off_ext3 = i * per_instance_ext3_u32;
            let off_base = i * per_instance_base_u32;
            convert_simd_ext3_to_aos(
                hg_evals_ptrs[i],
                hg_aos.as_mut_ptr().add(off_ext3),
                input_size,
                pack_size,
            );
            convert_simd_base_to_promoted_ext3_aos(
                init_v_ptrs[i],
                bk_f_aos.as_mut_ptr().add(off_ext3),
                input_size,
                pack_size,
            );
            convert_simd_base_to_aos(
                init_v_ptrs[i],
                init_v_aos.as_mut_ptr().add(off_base),
                input_size,
                pack_size,
            );
        }

        // Allocate.
        let d_bk_f = DevicePtr::alloc(ext3_bytes)?;
        let d_bk_hg = DevicePtr::alloc(ext3_bytes)?;
        let d_bk_f_alt = DevicePtr::alloc(ext3_bytes)?;
        let d_bk_hg_alt = DevicePtr::alloc(ext3_bytes)?;
        let d_init_v = DevicePtr::alloc(base_bytes)?;
        let d_challenge = DevicePtr::alloc(n_instances * pack_size * 3 * 4)?;
        let d_result = DevicePtr::alloc(n_instances * pack_size * 9 * 4)?;

        // Upload.
        if cuda_memcpy_h2d(d_bk_f.ptr() as _, bk_f_aos.as_ptr() as _, ext3_bytes) != 0 {
            return None;
        }
        if cuda_memcpy_h2d(d_bk_hg.ptr() as _, hg_aos.as_ptr() as _, ext3_bytes) != 0 {
            return None;
        }
        if cuda_memcpy_h2d(d_init_v.ptr() as _, init_v_aos.as_ptr() as _, base_bytes) != 0 {
            return None;
        }

        Some(BatchedGpuSumcheckContext {
            d_bk_f: d_bk_f.into_raw(),
            d_bk_hg: d_bk_hg.into_raw(),
            d_bk_f_alt: d_bk_f_alt.into_raw(),
            d_bk_hg_alt: d_bk_hg_alt.into_raw(),
            d_init_v: d_init_v.into_raw(),
            d_challenge: d_challenge.into_raw(),
            d_result: d_result.into_raw(),
            n_instances,
            pack_size,
            lane_stride_ext3,
            lane_stride_base,
        })
    }

    /// Total kernel batch dimension: `n_instances * pack_size`.
    pub fn total_batch(&self) -> usize {
        self.n_instances * self.pack_size
    }

    pub fn n_instances(&self) -> usize {
        self.n_instances
    }

    /// Whether `eval_size` is large enough to benefit from GPU dispatch.
    pub fn should_use_gpu(&self, eval_size: usize) -> bool {
        eval_size >= GPU_DISPATCH_THRESHOLD
    }

    /// Run `poly_eval_at` across all instances in one kernel dispatch.
    ///
    /// Returns a flat `Vec<[[u32; 9]; 16]>` of length `n_instances` where
    /// `result[i][lane]` is `[p0_l0, p0_l1, p0_l2, p1_l0, …, p2_l2]` for
    /// instance `i`'s SIMD lane `lane`. Bit-exact with calling
    /// `GpuSumcheckContext::poly_eval_at` once per instance.
    ///
    /// # Safety
    ///
    /// Device pointers must still be valid (context not dropped).
    pub unsafe fn poly_eval_at(&self, eval_size: usize) -> Option<Vec<[[u32; 9]; 16]>> {
        debug_assert!(self.pack_size <= 16);

        let total_batch = self.total_batch();
        let err = cuda_ffi::cuda_m31ext3_poly_eval_batched(
            self.d_bk_f as *const u32,
            self.d_bk_hg as *const u32,
            self.d_result,
            eval_size as u32,
            total_batch as u32,
            self.lane_stride_ext3 as u32,
        );
        if err != 0 {
            return None;
        }

        let mut packed = vec![0u32; total_batch * 9];
        if cuda_memcpy_d2h(
            packed.as_mut_ptr() as *mut std::ffi::c_void,
            self.d_result as *const std::ffi::c_void,
            total_batch * 9 * 4,
        ) != 0
        {
            return None;
        }

        let mut out = vec![[[0u32; 9]; 16]; self.n_instances];
        for i in 0..self.n_instances {
            for lane in 0..self.pack_size {
                let off = (i * self.pack_size + lane) * 9;
                out[i][lane].copy_from_slice(&packed[off..off + 9]);
            }
        }
        Some(out)
    }

    /// Run `receive_challenge` across all instances in one kernel dispatch.
    ///
    /// `r_limbs_per_instance[i]` is instance `i`'s scalar challenge for
    /// this round. Within an instance, r is replicated across all SIMD
    /// lanes (the lanes share a transcript inside one instance).
    ///
    /// First-round (`var_idx == 0`) currently falls back to a per-instance
    /// loop because the batched kernel doesn't yet have base-field
    /// `init_v` mode. That happens once per phase out of `log_n` rounds,
    /// so the bulk of dispatches still benefit from batching.
    ///
    /// # Safety
    ///
    /// Device pointers must still be valid.
    pub unsafe fn receive_challenge(
        &mut self,
        eval_size: usize,
        var_idx: usize,
        r_limbs_per_instance: &[[u32; 3]],
    ) -> bool {
        if r_limbs_per_instance.len() != self.n_instances {
            return false;
        }

        if var_idx == 0 {
            // First-round path: per-instance lane-sequential. The base-
            // field init_v optimization isn't supported in the batched
            // kernel (yet); follow-up work to extend the kernel.
            for i in 0..self.n_instances {
                if cuda_memcpy_h2d(
                    self.d_challenge as *mut std::ffi::c_void,
                    r_limbs_per_instance[i].as_ptr() as *const std::ffi::c_void,
                    3 * 4,
                ) != 0
                {
                    return false;
                }
                let inst_ext3_off = i * self.pack_size * self.lane_stride_ext3;
                let inst_base_off = i * self.pack_size * self.lane_stride_base;
                for lane in 0..self.pack_size {
                    let f_ptr = self
                        .d_bk_f
                        .add(inst_ext3_off + lane * self.lane_stride_ext3);
                    let hg_ptr = self
                        .d_bk_hg
                        .add(inst_ext3_off + lane * self.lane_stride_ext3);
                    let init_v_ptr = self
                        .d_init_v
                        .add(inst_base_off + lane * self.lane_stride_base)
                        as *const u32;

                    let err = cuda_ffi::cuda_m31ext3_receive_challenge(
                        f_ptr,
                        hg_ptr,
                        self.d_challenge as *const u32,
                        eval_size as u32,
                        1,
                        init_v_ptr,
                    );
                    if err != 0 {
                        return false;
                    }
                }
            }
            return true;
        }

        // Non-first round: replicate r across the pack and dispatch one
        // batched kernel call.
        let total_batch = self.total_batch();
        let mut replicated = vec![0u32; total_batch * 3];
        for i in 0..self.n_instances {
            for lane in 0..self.pack_size {
                let off = (i * self.pack_size + lane) * 3;
                replicated[off..off + 3].copy_from_slice(&r_limbs_per_instance[i]);
            }
        }
        if cuda_memcpy_h2d(
            self.d_challenge as *mut std::ffi::c_void,
            replicated.as_ptr() as *const std::ffi::c_void,
            total_batch * 3 * 4,
        ) != 0
        {
            return false;
        }

        let err = cuda_ffi::cuda_m31ext3_receive_challenge_batched(
            self.d_bk_f as *const u32,
            self.d_bk_hg as *const u32,
            self.d_bk_f_alt,
            self.d_bk_hg_alt,
            self.d_challenge as *const u32,
            eval_size as u32,
            total_batch as u32,
            self.lane_stride_ext3 as u32,
        );
        if err != 0 {
            return false;
        }
        std::mem::swap(&mut self.d_bk_f, &mut self.d_bk_f_alt);
        std::mem::swap(&mut self.d_bk_hg, &mut self.d_bk_hg_alt);
        true
    }

    /// Download instance `i`'s `bk_f` from device and convert AoS → SoA
    /// back into the host buffer at `dst`. Used for CPU fallback when
    /// `eval_size` drops below `GPU_DISPATCH_THRESHOLD` mid-prove.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least `num_elems` M31Ext3x16 values for
    /// instance `i`. Instance index must be in `[0, n_instances)`.
    pub unsafe fn download_instance_bk_f(
        &self,
        instance: usize,
        dst: *mut u32,
        num_elems: usize,
    ) -> bool {
        if instance >= self.n_instances {
            return false;
        }
        let inst_off = instance * self.pack_size * self.lane_stride_ext3;
        let src = self.d_bk_f.add(inst_off) as *const u32;

        let total_u32 = self.pack_size * num_elems * 3;
        let mut host_aos = vec![0u32; total_u32];
        if cuda_memcpy_d2h(
            host_aos.as_mut_ptr() as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            total_u32 * 4,
        ) != 0
        {
            return false;
        }

        convert_aos_to_simd_ext3(host_aos.as_ptr(), dst, num_elems, self.pack_size);
        true
    }
}

#[cfg(feature = "cuda")]
impl Drop for BatchedGpuSumcheckContext {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.d_bk_f as *mut std::ffi::c_void);
            cuda_free(self.d_bk_hg as *mut std::ffi::c_void);
            cuda_free(self.d_bk_f_alt as *mut std::ffi::c_void);
            cuda_free(self.d_bk_hg_alt as *mut std::ffi::c_void);
            cuda_free(self.d_init_v as *mut std::ffi::c_void);
            cuda_free(self.d_challenge as *mut std::ffi::c_void);
            cuda_free(self.d_result as *mut std::ffi::c_void);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod batched_correctness_tests {
    use super::*;

    /// Build synthetic SoA-x16 inputs for one instance, deterministic per seed.
    /// Returns (hg_soa, init_v_soa) of shape required by GpuSumcheckContext::new:
    ///   hg_soa: input_size * 48 u32 (M31Ext3x16, SoA)
    ///   init_v_soa: input_size * 16 u32 (M31x16, SoA)
    fn make_instance_inputs(input_size: usize, pack_size: usize, seed: u64) -> (Vec<u32>, Vec<u32>) {
        let mut hg = vec![0u32; input_size * pack_size * 3];
        let mut init_v = vec![0u32; input_size * pack_size];
        // Cheap LCG instead of pulling rand into the dev-deps just for this.
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Top 31 bits, masked to fit M31 prime (2^31 - 1).
            ((state >> 33) as u32) & 0x7FFF_FFFF
        };
        for v in hg.iter_mut() {
            *v = next();
        }
        for v in init_v.iter_mut() {
            *v = next();
        }
        (hg, init_v)
    }

    /// poly_eval_at output from the batched context must be bit-exact with
    /// running each instance through GpuSumcheckContext separately.
    #[test]
    fn batched_poly_eval_matches_per_instance() {
        const N_INSTANCES: usize = 3;
        const PACK_SIZE: usize = 16;
        const INPUT_SIZE: usize = 4096; // ≥ GPU_DISPATCH_THRESHOLD
        const EVAL_SIZE: usize = 2048; // first round halves input_size

        let mut instance_inputs = Vec::with_capacity(N_INSTANCES);
        for i in 0..N_INSTANCES {
            instance_inputs.push(make_instance_inputs(INPUT_SIZE, PACK_SIZE, 0xC0DE + i as u64));
        }

        // Single-instance dispatch: build one GpuSumcheckContext per instance,
        // call poly_eval_at, capture 16-lane result.
        let mut single_results = Vec::with_capacity(N_INSTANCES);
        for (hg, init_v) in &instance_inputs {
            let ctx = unsafe {
                GpuSumcheckContext::new(hg.as_ptr(), init_v.as_ptr(), PACK_SIZE, INPUT_SIZE)
            }
            .expect("GpuSumcheckContext::new");
            let res = unsafe { ctx.poly_eval_at(EVAL_SIZE) }.expect("single poly_eval_at");
            single_results.push(res);
        }

        // Batched dispatch: build one BatchedGpuSumcheckContext for all N,
        // call poly_eval_at once, get Vec of 16-lane results.
        let hg_ptrs: Vec<*const u32> = instance_inputs.iter().map(|(h, _)| h.as_ptr()).collect();
        let init_v_ptrs: Vec<*const u32> =
            instance_inputs.iter().map(|(_, v)| v.as_ptr()).collect();
        let batched = unsafe {
            BatchedGpuSumcheckContext::new(&hg_ptrs, &init_v_ptrs, PACK_SIZE, INPUT_SIZE)
        }
        .expect("BatchedGpuSumcheckContext::new");
        let batched_results =
            unsafe { batched.poly_eval_at(EVAL_SIZE) }.expect("batched poly_eval_at");

        assert_eq!(batched_results.len(), N_INSTANCES);
        for i in 0..N_INSTANCES {
            for lane in 0..PACK_SIZE {
                assert_eq!(
                    batched_results[i][lane], single_results[i][lane],
                    "instance {i} lane {lane} mismatch: \
                     batched={:?} vs single={:?}",
                    batched_results[i][lane], single_results[i][lane]
                );
            }
        }
    }

    /// receive_challenge then poly_eval_at must also match between
    /// single-instance and batched paths. Uses var_idx=1 (non-first
    /// round) so we exercise the batched receive_challenge dispatch.
    #[test]
    fn batched_receive_challenge_matches_per_instance() {
        const N_INSTANCES: usize = 3;
        const PACK_SIZE: usize = 16;
        const INPUT_SIZE: usize = 4096;
        const EVAL_SIZE: usize = 2048;

        let mut instance_inputs = Vec::with_capacity(N_INSTANCES);
        for i in 0..N_INSTANCES {
            instance_inputs.push(make_instance_inputs(INPUT_SIZE, PACK_SIZE, 0xBEEF + i as u64));
        }
        // Different challenge per instance (mirroring per-transcript divergence).
        let r_per_instance: Vec<[u32; 3]> = (0..N_INSTANCES)
            .map(|i| [0x1111_1111 + i as u32, 0x2222_2222 - i as u32, 0x3333_3333])
            .collect();

        // Single path.
        let mut single_post_results = Vec::with_capacity(N_INSTANCES);
        for (i, (hg, init_v)) in instance_inputs.iter().enumerate() {
            let mut ctx = unsafe {
                GpuSumcheckContext::new(hg.as_ptr(), init_v.as_ptr(), PACK_SIZE, INPUT_SIZE)
            }
            .expect("GpuSumcheckContext::new");
            let ok =
                unsafe { ctx.receive_challenge(EVAL_SIZE / 2, 1, &r_per_instance[i]) };
            assert!(ok, "single receive_challenge instance {i}");
            let res =
                unsafe { ctx.poly_eval_at(EVAL_SIZE / 4) }.expect("single poly_eval_at post-rc");
            single_post_results.push(res);
        }

        // Batched path.
        let hg_ptrs: Vec<*const u32> = instance_inputs.iter().map(|(h, _)| h.as_ptr()).collect();
        let init_v_ptrs: Vec<*const u32> =
            instance_inputs.iter().map(|(_, v)| v.as_ptr()).collect();
        let mut batched = unsafe {
            BatchedGpuSumcheckContext::new(&hg_ptrs, &init_v_ptrs, PACK_SIZE, INPUT_SIZE)
        }
        .expect("BatchedGpuSumcheckContext::new");
        let ok = unsafe { batched.receive_challenge(EVAL_SIZE / 2, 1, &r_per_instance) };
        assert!(ok, "batched receive_challenge");
        let batched_post = unsafe { batched.poly_eval_at(EVAL_SIZE / 4) }
            .expect("batched poly_eval_at post-rc");

        assert_eq!(batched_post.len(), N_INSTANCES);
        for i in 0..N_INSTANCES {
            for lane in 0..PACK_SIZE {
                assert_eq!(
                    batched_post[i][lane], single_post_results[i][lane],
                    "post-receive_challenge mismatch at instance {i} lane {lane}"
                );
            }
        }
    }
}
