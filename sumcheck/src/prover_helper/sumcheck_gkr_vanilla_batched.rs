//! Batched GKR sumcheck helper — drives N independent circuit-prove
//! instances through one layer's sumcheck rounds in lockstep, with the
//! hot GPU dispatches issued once per round across all N instances via
//! `BatchedGpuSumcheckContext`.
//!
//! Architectural pattern:
//! - Per-instance state (challenges, scratch pads, layer references) is
//!   held in a `Vec<InstanceState<'a, F>>` of length `n_instances`.
//! - The batched GPU context holds one mega-buffer with all N instances'
//!   bookkeeping stacked. At each `poly_evals_at_rx` / `receive_rx` call,
//!   one kernel dispatch covers all N instances at once. The diagnostic
//!   bench shows ~14× per-fold throughput at batch=256 vs batch=16 for
//!   completeness-shaped layers.
//! - SIMD/MPI variable rounds and most prep work run per-instance
//!   sequentially because they are CPU-only and cheap relative to the
//!   GPU-bottlenecked main rounds.
//! - Each instance maintains an independent transcript; the outer
//!   driver (`sumcheck_prove_gkr_layer_batched`) absorbs N evals and
//!   squeezes N challenges per round, feeding them into the batched
//!   helper for the next dispatch.
//!
//! Bit-exact correctness is the soundness guarantee: every per-instance
//! computation routes through the same `unpack_and_combine` and
//! `coef_combine_vec` paths the single-instance helper uses. Tests in
//! `cuda_dispatch.rs` verify the kernels themselves; tests below
//! verify the helper composition.

use arith::{Field, SimdField};
use circuit::CircuitLayer;
#[cfg(feature = "cuda")]
use gkr_engine::FieldType;
use gkr_engine::{ExpanderDualVarChallenge, FieldEngine, MPIEngine};
use polynomials::EqPolynomial;

use crate::{unpack_and_combine, ProverScratchPad};

use super::{product_gate::SumcheckProductGateHelper, simd_gate::SumcheckSimdProdGateHelper};

#[cfg(feature = "cuda")]
use crate::cuda_dispatch::{BatchedGpuSumcheckContext, GPU_DISPATCH_THRESHOLD};

/// Per-instance state mirroring `SumcheckGkrVanillaHelper` — one of
/// these per circuit prove the batched helper is driving in parallel.
///
/// `challenge` is owned by clone (rather than borrowed) so the helper
/// doesn't pin a borrow on the caller's challenge slice across the
/// whole layer prove. Cheap — challenges are tens of field elements.
pub(crate) struct InstanceState<'a, F: FieldEngine> {
    pub(crate) rx: Vec<F::ChallengeField>,
    pub(crate) ry: Vec<F::ChallengeField>,
    pub(crate) r_simd_var: Vec<F::ChallengeField>,
    pub(crate) r_mpi_var: Vec<F::ChallengeField>,

    pub(crate) layer: &'a CircuitLayer<F>,
    pub(crate) sp: &'a mut ProverScratchPad<F>,

    pub(crate) challenge: ExpanderDualVarChallenge<F>,
    pub(crate) alpha: Option<F::ChallengeField>,

    pub(crate) input_var_num: usize,
    pub(crate) simd_var_num: usize,

    pub(crate) xy_helper: SumcheckProductGateHelper,
    pub(crate) simd_var_helper: SumcheckSimdProdGateHelper<F>,
    pub(crate) mpi_var_helper: SumcheckSimdProdGateHelper<F>,

    pub(crate) is_output_layer: bool,
}

/// Drives N independent GKR sumcheck proves in lockstep at one layer.
pub(crate) struct BatchedSumcheckGkrVanillaHelper<'a, F: FieldEngine> {
    pub(crate) instances: Vec<InstanceState<'a, F>>,

    /// Single batched GPU context shared across all instances. Allocated
    /// in `prepare_x_vals_batched` once the layer's eval_size is known
    /// to clear `GPU_DISPATCH_THRESHOLD`. None means CPU fallback.
    #[cfg(feature = "cuda")]
    pub(crate) batched_gpu_ctx: Option<BatchedGpuSumcheckContext>,

    pub(crate) n_instances: usize,
}

impl<'a, F: FieldEngine> BatchedSumcheckGkrVanillaHelper<'a, F> {
    /// Build N per-instance helper states. Each instance gets its own
    /// `(layer, challenge, alpha, sp)` triplet — same layer per instance
    /// (all proving the same circuit), but independent transcripts.
    pub(crate) fn new(
        layer_per_instance: Vec<&'a CircuitLayer<F>>,
        challenge_per_instance: Vec<ExpanderDualVarChallenge<F>>,
        alpha_per_instance: Vec<Option<F::ChallengeField>>,
        sp_per_instance: Vec<&'a mut ProverScratchPad<F>>,
        mpi_config: &impl MPIEngine,
        is_output_layer: bool,
    ) -> Self {
        let n = layer_per_instance.len();
        assert_eq!(challenge_per_instance.len(), n);
        assert_eq!(alpha_per_instance.len(), n);
        assert_eq!(sp_per_instance.len(), n);

        let simd_var_num = F::get_field_pack_size().trailing_zeros() as usize;
        let mpi_logsize = mpi_config.world_size().trailing_zeros() as usize;

        let instances: Vec<InstanceState<'a, F>> = layer_per_instance
            .into_iter()
            .zip(challenge_per_instance)
            .zip(alpha_per_instance)
            .zip(sp_per_instance)
            .map(|(((layer, challenge), alpha), sp)| InstanceState {
                rx: vec![],
                ry: vec![],
                r_simd_var: vec![],
                r_mpi_var: vec![],

                layer,
                sp,
                challenge,
                alpha,

                input_var_num: layer.input_var_num,
                simd_var_num,

                xy_helper: SumcheckProductGateHelper::new(layer.input_var_num),
                simd_var_helper: SumcheckSimdProdGateHelper::new(simd_var_num),
                mpi_var_helper: SumcheckSimdProdGateHelper::new(mpi_logsize),
                is_output_layer,
            })
            .collect();

        Self {
            instances,
            #[cfg(feature = "cuda")]
            batched_gpu_ctx: None,
            n_instances: n,
        }
    }

    /// Run `prepare_simd` per instance (CPU-only; cheap).
    pub(crate) fn prepare_simd(&mut self) {
        for inst in &mut self.instances {
            if inst.is_output_layer || inst.alpha.is_none() {
                EqPolynomial::<F::ChallengeField>::eq_eval_at(
                    &inst.challenge.r_simd,
                    &F::ChallengeField::one(),
                    &mut inst.sp.eq_evals_at_r_simd0,
                    &mut inst.sp.eq_evals_first_half,
                    &mut inst.sp.eq_evals_second_half,
                );
            }
        }
    }

    /// Run `prepare_mpi` per instance.
    pub(crate) fn prepare_mpi(&mut self) {
        for inst in &mut self.instances {
            if inst.is_output_layer || inst.alpha.is_none() {
                EqPolynomial::<F::ChallengeField>::eq_eval_at(
                    &inst.challenge.r_mpi,
                    &F::ChallengeField::one(),
                    &mut inst.sp.eq_evals_at_r_mpi0,
                    &mut inst.sp.eq_evals_first_half,
                    &mut inst.sp.eq_evals_second_half,
                );
            }
        }
    }

    /// Run `prepare_x_vals` per instance, then build the shared
    /// batched GPU context if the layer's initial eval_size is large
    /// enough for GPU dispatch.
    pub(crate) fn prepare_x_vals(&mut self) {
        for inst in &mut self.instances {
            prepare_x_vals_one(inst);
        }

        // Build the batched GPU context once for the whole batch. Same
        // dispatch-threshold gate as the single-instance helper.
        #[cfg(feature = "cuda")]
        {
            if F::FIELD_TYPE == FieldType::M31x16 && !self.instances.is_empty() {
                let input_var_num = self.instances[0].input_var_num;
                let initial_eval_size = 1usize << (input_var_num - 1);
                if initial_eval_size >= GPU_DISPATCH_THRESHOLD {
                    let input_size = 1usize << input_var_num;
                    let pack_size = F::get_field_pack_size();
                    let hg_ptrs: Vec<*const u32> = self
                        .instances
                        .iter()
                        .map(|inst| inst.sp.hg_evals.as_ptr() as *const u32)
                        .collect();
                    let init_v_ptrs: Vec<*const u32> = self
                        .instances
                        .iter()
                        .map(|inst| inst.layer.input_vals.as_ptr() as *const u32)
                        .collect();
                    self.batched_gpu_ctx = unsafe {
                        BatchedGpuSumcheckContext::new(
                            &hg_ptrs,
                            &init_v_ptrs,
                            pack_size,
                            input_size,
                        )
                    };
                    if self.batched_gpu_ctx.is_none() {
                        log::warn!(
                            "BatchedGpuSumcheckContext creation failed; falling back to CPU"
                        );
                    }
                }
            }
        }
    }

    /// Run `prepare_simd_var_vals` per instance.
    pub(crate) fn prepare_simd_var_vals(&mut self) {
        for inst in &mut self.instances {
            inst.sp.simd_var_v_evals = inst.sp.v_evals[0].unpack();
            inst.sp.simd_var_hg_evals = inst.sp.hg_evals[0].unpack();
        }
    }

    /// Run `prepare_mpi_var_vals` per instance.
    pub(crate) fn prepare_mpi_var_vals(&mut self, mpi_config: &impl MPIEngine) {
        for inst in &mut self.instances {
            mpi_config.gather_vec(&[inst.sp.simd_var_v_evals[0]], &mut inst.sp.mpi_var_v_evals);
            mpi_config.gather_vec(
                &[inst.sp.simd_var_hg_evals[0] * inst.sp.eq_evals_at_r_simd0[0]],
                &mut inst.sp.mpi_var_hg_evals,
            );
        }
    }

    /// Run `prepare_y_vals` per instance, rebuild the batched GPU
    /// context for phase 2.
    pub(crate) fn prepare_y_vals(&mut self, mpi_config: &impl MPIEngine) {
        for inst in &mut self.instances {
            prepare_y_vals_one(inst, mpi_config);
        }

        #[cfg(feature = "cuda")]
        {
            // Drop any phase-1 context.
            self.batched_gpu_ctx = None;

            if F::FIELD_TYPE == FieldType::M31x16 && !self.instances.is_empty() {
                let input_var_num = self.instances[0].input_var_num;
                let initial_eval_size = 1usize << (input_var_num - 1);
                if initial_eval_size >= GPU_DISPATCH_THRESHOLD {
                    let input_size = 1usize << input_var_num;
                    let pack_size = F::get_field_pack_size();
                    let hg_ptrs: Vec<*const u32> = self
                        .instances
                        .iter()
                        .map(|inst| inst.sp.hg_evals.as_ptr() as *const u32)
                        .collect();
                    let init_v_ptrs: Vec<*const u32> = self
                        .instances
                        .iter()
                        .map(|inst| inst.layer.input_vals.as_ptr() as *const u32)
                        .collect();
                    self.batched_gpu_ctx = unsafe {
                        BatchedGpuSumcheckContext::new(
                            &hg_ptrs,
                            &init_v_ptrs,
                            pack_size,
                            input_size,
                        )
                    };
                    if self.batched_gpu_ctx.is_none() {
                        log::warn!(
                            "BatchedGpuSumcheckContext creation failed; falling back to CPU"
                        );
                    }
                }
            }
        }
    }

    /// Batched poly_evals_at_rx: returns one `[F::ChallengeField; 3]` per
    /// instance. GPU fast path: ONE `cuda_m31ext3_poly_eval_batched`
    /// dispatch covers all `n_instances * pack_size` lanes; per-instance
    /// SIMD/MPI reductions run after the readback. CPU fallback path
    /// runs each instance through its own xy_helper sequentially.
    pub(crate) fn poly_evals_at_rx(
        &mut self,
        var_idx: usize,
        degree: usize,
        mpi_config: &impl MPIEngine,
    ) -> Vec<[F::ChallengeField; 3]> {
        for inst in &self.instances {
            assert!(var_idx < inst.input_var_num);
        }

        #[cfg(feature = "cuda")]
        {
            let input_var_num = self.instances[0].input_var_num;
            let eval_size = 1usize << (input_var_num - var_idx - 1);
            let use_gpu = self
                .batched_gpu_ctx
                .as_ref()
                .map_or(false, |ctx| ctx.should_use_gpu(eval_size));

            if use_gpu {
                let ctx = self.batched_gpu_ctx.as_ref().unwrap();
                if let Some(per_instance_per_lane) = unsafe { ctx.poly_eval_at(eval_size) } {
                    let pack_size = F::get_field_pack_size();
                    let result: Vec<[F::ChallengeField; 3]> = self
                        .instances
                        .iter()
                        .zip(per_instance_per_lane)
                        .map(|(inst, per_lane)| {
                            let [p0, p1, mut p2] =
                                gpu_pack_poly_eval_results::<F>(&per_lane, pack_size);

                            // Same M31 p2 correction as single-instance helper.
                            p2 = p1.mul_by_6() + p0.mul_by_3() - p2.double();

                            let local_vals_simd = [p0, p1, p2];
                            let local_vals = local_vals_simd
                                .iter()
                                .map(|p| unpack_and_combine(p, &inst.sp.eq_evals_at_r_simd0))
                                .collect::<Vec<F::ChallengeField>>();

                            mpi_config
                                .coef_combine_vec(&local_vals, &inst.sp.eq_evals_at_r_mpi0)
                                .try_into()
                                .unwrap()
                        })
                        .collect();
                    return result;
                }
                // GPU kernel failed — fall through to per-instance CPU.
            } else if self.batched_gpu_ctx.is_some() {
                self.transition_gpu_to_cpu(eval_size);
            }
        }

        // CPU path: per-instance sequentially.
        self.instances
            .iter_mut()
            .map(|inst| {
                let local_vals_simd = inst.xy_helper.poly_eval_at::<F>(
                    var_idx,
                    degree,
                    &inst.sp.v_evals,
                    &inst.sp.hg_evals,
                    &inst.layer.input_vals,
                    &inst.sp.gate_exists,
                );
                let local_vals = local_vals_simd
                    .iter()
                    .map(|p| unpack_and_combine(p, &inst.sp.eq_evals_at_r_simd0))
                    .collect::<Vec<F::ChallengeField>>();

                mpi_config
                    .coef_combine_vec(&local_vals, &inst.sp.eq_evals_at_r_mpi0)
                    .try_into()
                    .unwrap()
            })
            .collect()
    }

    /// Batched poly_evals_at_r_simd_var (CPU-only currently — the SIMD
    /// var phase has no GPU kernel in upstream Expander).
    pub(crate) fn poly_evals_at_r_simd_var(
        &mut self,
        var_idx: usize,
        degree: usize,
        mpi_config: &impl MPIEngine,
    ) -> Vec<[F::ChallengeField; 4]> {
        self.instances
            .iter_mut()
            .map(|inst| {
                assert!(var_idx < inst.simd_var_num);
                let local_vals = inst
                    .simd_var_helper
                    .poly_eval_at(
                        var_idx,
                        degree,
                        &mut inst.sp.eq_evals_at_r_simd0,
                        &mut inst.sp.simd_var_v_evals,
                        &mut inst.sp.simd_var_hg_evals,
                    )
                    .to_vec();

                mpi_config
                    .coef_combine_vec(&local_vals, &inst.sp.eq_evals_at_r_mpi0)
                    .try_into()
                    .unwrap()
            })
            .collect()
    }

    /// Batched poly_evals_at_r_mpi_var (CPU-only).
    pub(crate) fn poly_evals_at_r_mpi_var(
        &mut self,
        var_idx: usize,
        degree: usize,
    ) -> Vec<[F::ChallengeField; 4]> {
        self.instances
            .iter_mut()
            .map(|inst| {
                assert!(var_idx < inst.mpi_var_helper.var_num);
                inst.mpi_var_helper.poly_eval_at(
                    var_idx,
                    degree,
                    &mut inst.sp.eq_evals_at_r_mpi0,
                    &mut inst.sp.mpi_var_v_evals,
                    &mut inst.sp.mpi_var_hg_evals,
                )
            })
            .collect()
    }

    /// Batched poly_evals_at_ry: like rx but applies phase2_coef.
    pub(crate) fn poly_evals_at_ry(
        &mut self,
        var_idx: usize,
        degree: usize,
        mpi_config: &impl MPIEngine,
    ) -> Vec<[F::ChallengeField; 3]> {
        let raw = self.poly_evals_at_rx(var_idx, degree, mpi_config);
        raw.into_iter()
            .zip(&self.instances)
            .map(|([p0, p1, p2], inst)| {
                [
                    p0 * inst.sp.phase2_coef,
                    p1 * inst.sp.phase2_coef,
                    p2 * inst.sp.phase2_coef,
                ]
            })
            .collect()
    }

    /// Batched receive_rx: takes one challenge per instance.
    pub(crate) fn receive_rx(&mut self, var_idx: usize, r_per_instance: &[F::ChallengeField]) {
        assert_eq!(r_per_instance.len(), self.n_instances);

        #[cfg(feature = "cuda")]
        {
            let input_var_num = self.instances[0].input_var_num;
            let eval_size = 1usize << (input_var_num - var_idx - 1);
            let use_gpu = self
                .batched_gpu_ctx
                .as_ref()
                .map_or(false, |ctx| ctx.should_use_gpu(eval_size));

            if use_gpu {
                let r_limbs: Vec<[u32; 3]> = r_per_instance
                    .iter()
                    .map(gpu_challenge_to_limbs::<F>)
                    .collect();
                let mut ctx = self.batched_gpu_ctx.take().unwrap();
                let ok = unsafe { ctx.receive_challenge(eval_size, var_idx, &r_limbs) };
                self.batched_gpu_ctx = Some(ctx);
                if ok {
                    // Maintain gate_exists per instance.
                    for inst in &mut self.instances {
                        gpu_update_gate_exists(&mut inst.sp.gate_exists, eval_size);
                    }
                    for (inst, r) in self.instances.iter_mut().zip(r_per_instance) {
                        inst.rx.push(*r);
                    }
                    return;
                }
                // GPU failed — fall through to CPU.
            }
        }

        for (inst, r) in self.instances.iter_mut().zip(r_per_instance) {
            inst.xy_helper.receive_challenge::<F>(
                var_idx,
                *r,
                &mut inst.sp.v_evals,
                &mut inst.sp.hg_evals,
                &inst.layer.input_vals,
                &mut inst.sp.gate_exists,
            );
            inst.rx.push(*r);
        }
    }

    /// Batched receive_r_simd_var (CPU-only).
    pub(crate) fn receive_r_simd_var(
        &mut self,
        var_idx: usize,
        r_per_instance: &[F::ChallengeField],
    ) {
        assert_eq!(r_per_instance.len(), self.n_instances);
        for (inst, r) in self.instances.iter_mut().zip(r_per_instance) {
            inst.simd_var_helper.receive_challenge(
                var_idx,
                *r,
                &mut inst.sp.eq_evals_at_r_simd0,
                &mut inst.sp.simd_var_v_evals,
                &mut inst.sp.simd_var_hg_evals,
            );
            inst.r_simd_var.push(*r);
        }
    }

    /// Batched receive_r_mpi_var (CPU-only).
    pub(crate) fn receive_r_mpi_var(
        &mut self,
        var_idx: usize,
        r_per_instance: &[F::ChallengeField],
    ) {
        assert_eq!(r_per_instance.len(), self.n_instances);
        for (inst, r) in self.instances.iter_mut().zip(r_per_instance) {
            inst.mpi_var_helper.receive_challenge(
                var_idx,
                *r,
                &mut inst.sp.eq_evals_at_r_mpi0,
                &mut inst.sp.mpi_var_v_evals,
                &mut inst.sp.mpi_var_hg_evals,
            );
            inst.r_mpi_var.push(*r);
        }
    }

    /// Batched receive_ry: one challenge per instance, same dispatch
    /// path as receive_rx.
    pub(crate) fn receive_ry(&mut self, var_idx: usize, r_per_instance: &[F::ChallengeField]) {
        assert_eq!(r_per_instance.len(), self.n_instances);

        #[cfg(feature = "cuda")]
        {
            let input_var_num = self.instances[0].input_var_num;
            let eval_size = 1usize << (input_var_num - var_idx - 1);
            let use_gpu = self
                .batched_gpu_ctx
                .as_ref()
                .map_or(false, |ctx| ctx.should_use_gpu(eval_size));

            if use_gpu {
                let r_limbs: Vec<[u32; 3]> = r_per_instance
                    .iter()
                    .map(gpu_challenge_to_limbs::<F>)
                    .collect();
                let mut ctx = self.batched_gpu_ctx.take().unwrap();
                let ok = unsafe { ctx.receive_challenge(eval_size, var_idx, &r_limbs) };
                self.batched_gpu_ctx = Some(ctx);
                if ok {
                    for inst in &mut self.instances {
                        gpu_update_gate_exists(&mut inst.sp.gate_exists, eval_size);
                    }
                    for (inst, r) in self.instances.iter_mut().zip(r_per_instance) {
                        inst.ry.push(*r);
                    }
                    return;
                }
            }
        }

        for (inst, r) in self.instances.iter_mut().zip(r_per_instance) {
            inst.xy_helper.receive_challenge::<F>(
                var_idx,
                *r,
                &mut inst.sp.v_evals,
                &mut inst.sp.hg_evals,
                &inst.layer.input_vals,
                &mut inst.sp.gate_exists,
            );
            inst.ry.push(*r);
        }
    }

    pub(crate) fn vx_claim_per_instance(&self) -> Vec<F::ChallengeField> {
        self.instances
            .iter()
            .map(|inst| inst.sp.mpi_var_v_evals[0])
            .collect()
    }

    pub(crate) fn vy_claim_per_instance(
        &self,
        mpi_config: &impl MPIEngine,
    ) -> Vec<F::ChallengeField> {
        self.instances
            .iter()
            .map(|inst| {
                let vy_local = unpack_and_combine(&inst.sp.v_evals[0], &inst.sp.eq_evals_at_r_simd0);
                mpi_config.coef_combine_vec(&[vy_local], &inst.sp.eq_evals_at_r_mpi0)[0]
            })
            .collect()
    }

    /// Download per-instance state from the batched GPU context back to
    /// each scratch pad and drop the context. Used when eval_size drops
    /// below the GPU dispatch threshold mid-prove.
    ///
    /// Both `bk_f` (v_evals) and `bk_hg` (hg_evals) must come back; the
    /// CPU helper reads both in its `xy_helper.poly_eval_at` round.
    /// Skipping `bk_hg` leaves stale pre-fold values on host — the bug
    /// that caused single-vs-batched prove divergence on layers whose
    /// initial eval_size triggered exactly one GPU round before
    /// dropping below threshold.
    #[cfg(feature = "cuda")]
    fn transition_gpu_to_cpu(&mut self, current_eval_size: usize) {
        if let Some(ctx) = self.batched_gpu_ctx.take() {
            let elems_to_download = current_eval_size * 2;
            for (i, inst) in self.instances.iter_mut().enumerate() {
                unsafe {
                    ctx.download_instance_bk_f(
                        i,
                        inst.sp.v_evals.as_mut_ptr() as *mut u32,
                        elems_to_download,
                    );
                    ctx.download_instance_bk_hg(
                        i,
                        inst.sp.hg_evals.as_mut_ptr() as *mut u32,
                        elems_to_download,
                    );
                }
            }
            // ctx dropped → device memory freed.
        }
    }
}

// -----------------------------------------------------------------
// Per-instance prepare functions — direct ports of the single-instance
// helper's `prepare_x_vals` / `prepare_y_vals`, except the GPU context
// is allocated by the batched helper instead of per-instance.
// -----------------------------------------------------------------

fn prepare_x_vals_one<'a, F: FieldEngine>(inst: &mut InstanceState<'a, F>) {
    let mul = &inst.layer.mul;
    let add = &inst.layer.add;
    let vals = &inst.layer.input_vals;
    let eq_evals_at_rz0 = &mut inst.sp.eq_evals_at_rz0;
    let gate_exists = &mut inst.sp.gate_exists;
    let hg_vals = &mut inst.sp.hg_evals;
    unsafe {
        std::ptr::write_bytes(hg_vals.as_mut_ptr(), 0, vals.len());
    }
    unsafe {
        std::ptr::write_bytes(gate_exists.as_mut_ptr(), 0, vals.len());
    }

    assert_eq!(inst.challenge.rz_1.is_none(), inst.alpha.is_none());

    if inst.is_output_layer || inst.challenge.rz_1.is_none() {
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &inst.challenge.rz_0,
            &F::ChallengeField::ONE,
            eq_evals_at_rz0,
            &mut inst.sp.eq_evals_first_half,
            &mut inst.sp.eq_evals_second_half,
        );
    } else {
        let alpha = inst.alpha.unwrap();
        let eq_evals_at_rx_previous = &inst.sp.eq_evals_at_rx;
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            inst.challenge.rz_1.as_ref().unwrap(),
            &alpha,
            eq_evals_at_rz0,
            &mut inst.sp.eq_evals_first_half,
            &mut inst.sp.eq_evals_second_half,
        );

        for i in 0..(1 << inst.challenge.rz_0.len()) {
            eq_evals_at_rz0[i] += eq_evals_at_rx_previous[i];
        }
    }

    for g in mul.iter() {
        let r = eq_evals_at_rz0[g.o_id] * g.coef;
        hg_vals[g.i_ids[0]] += r * vals[g.i_ids[1]];
        gate_exists[g.i_ids[0]] = true;
    }

    for g in add.iter() {
        hg_vals[g.i_ids[0]] += F::Field::from(eq_evals_at_rz0[g.o_id] * g.coef);
        gate_exists[g.i_ids[0]] = true;
    }

}

fn prepare_y_vals_one<'a, F: FieldEngine>(
    inst: &mut InstanceState<'a, F>,
    mpi_config: &impl MPIEngine,
) {
    let mut v_rx_rsimd_rw = inst.sp.mpi_var_v_evals[0];
    mpi_config.root_broadcast_f(&mut v_rx_rsimd_rw);

    let mul = &inst.layer.mul;
    let eq_evals_at_rz0 = &inst.sp.eq_evals_at_rz0;
    let eq_evals_at_rx = &mut inst.sp.eq_evals_at_rx;
    let gate_exists = &mut inst.sp.gate_exists;
    let hg_vals = &mut inst.sp.hg_evals;
    let fill_len = 1 << inst.rx.len();
    unsafe {
        std::ptr::write_bytes(hg_vals.as_mut_ptr(), 0, fill_len);
    }
    unsafe {
        std::ptr::write_bytes(gate_exists.as_mut_ptr(), 0, fill_len);
    }

    inst.sp.phase2_coef =
        EqPolynomial::<F::ChallengeField>::eq_vec(&inst.challenge.r_mpi, &inst.r_mpi_var)
            * inst.sp.eq_evals_at_r_simd0[0]
            * v_rx_rsimd_rw;

    EqPolynomial::<F::ChallengeField>::eq_eval_at(
        &inst.r_mpi_var,
        &F::ChallengeField::ONE,
        &mut inst.sp.eq_evals_at_r_mpi0,
        &mut inst.sp.eq_evals_first_half,
        &mut inst.sp.eq_evals_second_half,
    );

    EqPolynomial::<F::ChallengeField>::eq_eval_at(
        &inst.rx,
        &F::ChallengeField::ONE,
        eq_evals_at_rx,
        &mut inst.sp.eq_evals_first_half,
        &mut inst.sp.eq_evals_second_half,
    );

    EqPolynomial::<F::ChallengeField>::eq_eval_at(
        &inst.r_simd_var,
        &F::ChallengeField::ONE,
        &mut inst.sp.eq_evals_at_r_simd0,
        &mut inst.sp.eq_evals_first_half,
        &mut inst.sp.eq_evals_second_half,
    );

    for g in mul.iter() {
        hg_vals[g.i_ids[1]] +=
            F::Field::from(eq_evals_at_rz0[g.o_id] * eq_evals_at_rx[g.i_ids[0]] * g.coef);
        gate_exists[g.i_ids[1]] = true;
    }
}

// -----------------------------------------------------------------
// Free helpers ported from the single-instance file.
// -----------------------------------------------------------------

#[cfg(feature = "cuda")]
fn gpu_pack_poly_eval_results<F: FieldEngine>(
    per_lane: &[[u32; 9]; 16],
    pack_size: usize,
) -> [F::Field; 3] {
    let mut result = [F::Field::ZERO; 3];
    for component in 0..3usize {
        let result_ptr = &mut result[component] as *mut F::Field as *mut u32;
        for lane in 0..pack_size {
            for limb in 0..3usize {
                let val = per_lane[lane][component * 3 + limb];
                unsafe {
                    *result_ptr.add(limb * pack_size + lane) = val;
                }
            }
        }
    }
    result
}

#[cfg(feature = "cuda")]
fn gpu_challenge_to_limbs<F: FieldEngine>(r: &F::ChallengeField) -> [u32; 3] {
    let ptr = r as *const F::ChallengeField as *const u32;
    unsafe { [*ptr, *ptr.add(1), *ptr.add(2)] }
}

#[cfg(feature = "cuda")]
fn gpu_update_gate_exists(gate_exists: &mut [bool], eval_size: usize) {
    for i in 0..eval_size {
        gate_exists[i] = gate_exists[i * 2] || gate_exists[i * 2 + 1];
    }
}
