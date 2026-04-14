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
        let d_init_v = DevicePtr::alloc(base_bytes)?;
        let d_challenge = DevicePtr::alloc(3 * 4)?;
        let d_result = DevicePtr::alloc(9 * 4)?;

        // ---- Upload ----

        if cuda_memcpy_h2d(
            d_bk_f.ptr() as _,
            bk_f_aos.as_ptr() as _,
            ext3_bytes,
        ) != 0
        {
            return None;
        }
        if cuda_memcpy_h2d(
            d_bk_hg.ptr() as _,
            hg_aos.as_ptr() as _,
            ext3_bytes,
        ) != 0
        {
            return None;
        }
        if cuda_memcpy_h2d(
            d_init_v.ptr() as _,
            init_v_aos.as_ptr() as _,
            base_bytes,
        ) != 0
        {
            return None;
        }

        Some(GpuSumcheckContext {
            d_bk_f: d_bk_f.into_raw(),
            d_bk_hg: d_bk_hg.into_raw(),
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
    /// # Safety
    ///
    /// Device pointers must still be valid (context not dropped).
    pub unsafe fn poly_eval_at(&self, eval_size: usize) -> Option<[[u32; 9]; 16]> {
        debug_assert!(self.pack_size <= 16);
        let mut results = [[0u32; 9]; 16];

        for lane in 0..self.pack_size {
            let f_ptr = self.d_bk_f.add(lane * self.lane_stride_ext3) as *const u32;
            let hg_ptr = self.d_bk_hg.add(lane * self.lane_stride_ext3) as *const u32;

            let err =
                cuda_ffi::cuda_m31ext3_poly_eval(f_ptr, hg_ptr, self.d_result, eval_size as u32);
            if err != 0 {
                return None;
            }

            if cuda_memcpy_d2h(
                results[lane].as_mut_ptr() as *mut std::ffi::c_void,
                self.d_result as *const std::ffi::c_void,
                9 * 4,
            ) != 0
            {
                return None;
            }
        }

        Some(results)
    }

    /// Run `receive_challenge` on GPU for all SIMD lanes.
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
        // Upload the scalar challenge (shared across all lanes).
        if cuda_memcpy_h2d(
            self.d_challenge as *mut std::ffi::c_void,
            r_limbs.as_ptr() as *const std::ffi::c_void,
            3 * 4,
        ) != 0
        {
            return false;
        }

        let first_round: i32 = if var_idx == 0 { 1 } else { 0 };

        for lane in 0..self.pack_size {
            let f_ptr = self.d_bk_f.add(lane * self.lane_stride_ext3);
            let hg_ptr = self.d_bk_hg.add(lane * self.lane_stride_ext3);
            let init_v_ptr = self.d_init_v.add(lane * self.lane_stride_base) as *const u32;

            let err = cuda_ffi::cuda_m31ext3_receive_challenge(
                f_ptr,
                hg_ptr,
                self.d_challenge as *const u32,
                eval_size as u32,
                first_round,
                init_v_ptr,
            );
            if err != 0 {
                return false;
            }
        }

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
            cuda_free(self.d_init_v as *mut std::ffi::c_void);
            cuda_free(self.d_challenge as *mut std::ffi::c_void);
            cuda_free(self.d_result as *mut std::ffi::c_void);
        }
    }
}
