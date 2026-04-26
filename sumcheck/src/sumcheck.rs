use circuit::CircuitLayer;
use gkr_engine::{ExpanderDualVarChallenge, FieldEngine, MPIEngine, Transcript};

use crate::{
    prover_helper::{BatchedSumcheckGkrVanillaHelper, SumcheckGkrVanillaHelper},
    utils::transcript_io,
    ProverScratchPad,
};

/// The degree of the polynomial for sumcheck, which is 2 for non-SIMD/MPI variables
/// and 3 for SIMD/MPI variables.
pub const SUMCHECK_GKR_DEGREE: usize = 2;
pub const SUMCHECK_GKR_SIMD_MPI_DEGREE: usize = 3;

// FIXME
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
// essentially the prev level of challenge passes here, once this level is done, new challenge gets
// written back into the prev space
pub fn sumcheck_prove_gkr_layer<F: FieldEngine, T: Transcript>(
    layer: &CircuitLayer<F>,
    challenge: &mut ExpanderDualVarChallenge<F>,
    alpha: Option<F::ChallengeField>,
    transcript: &mut T,
    sp: &mut ProverScratchPad<F>,
    mpi_config: &impl MPIEngine,
    is_output_layer: bool,
) -> (F::ChallengeField, Option<F::ChallengeField>) {
    let mut helper =
        SumcheckGkrVanillaHelper::new(layer, challenge, alpha, sp, mpi_config, is_output_layer);

    helper.prepare_simd();
    helper.prepare_mpi();

    // gkr phase 1 over variable x
    helper.prepare_x_vals();
    for i_var in 0..helper.input_var_num {
        let evals = helper.poly_evals_at_rx(i_var, SUMCHECK_GKR_DEGREE, mpi_config);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        helper.receive_rx(i_var, r);
        log::trace!("x i_var={i_var} evals: {evals:?} r: {r:?}");
    }

    helper.prepare_simd_var_vals();
    for i_var in 0..helper.simd_var_num {
        let evals =
            helper.poly_evals_at_r_simd_var(i_var, SUMCHECK_GKR_SIMD_MPI_DEGREE, mpi_config);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        helper.receive_r_simd_var(i_var, r);
        log::trace!("SIMD i_var={i_var} evals: {evals:?} r: {r:?}");
    }

    helper.prepare_mpi_var_vals(mpi_config);
    for i_var in 0..mpi_config.world_size().trailing_zeros() as usize {
        let evals = helper.poly_evals_at_r_mpi_var(i_var, SUMCHECK_GKR_SIMD_MPI_DEGREE);
        let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
        helper.receive_r_mpi_var(i_var, r);
    }

    let vx_claim = helper.vx_claim();
    transcript.append_field_element(&vx_claim);

    // gkr phase 2 over variable y
    let mut vy_claim = None;
    if !layer.structure_info.skip_sumcheck_phase_two {
        helper.prepare_y_vals(mpi_config);
        for i_var in 0..helper.input_var_num {
            let evals = helper.poly_evals_at_ry(i_var, SUMCHECK_GKR_DEGREE, mpi_config);
            let r = transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
            helper.receive_ry(i_var, r);
        }
        vy_claim = Some(helper.vy_claim(mpi_config));
        transcript.append_field_element(&vy_claim.unwrap());
    }

    let rx = helper.rx;
    let ry = if !layer.structure_info.skip_sumcheck_phase_two {
        Some(helper.ry)
    } else {
        None
    };
    let r_simd = helper.r_simd_var;
    let r_mpi = helper.r_mpi_var;

    *challenge = ExpanderDualVarChallenge::new(rx, ry, r_simd, r_mpi);
    (vx_claim, vy_claim)
}

/// Batched-instance variant of [`sumcheck_prove_gkr_layer`].
///
/// Drives N independent GKR sumcheck proves through ONE layer in
/// lockstep. Each instance has its own:
///   - layer (must share the same shape across instances — same circuit)
///   - challenge (mutated in place; per-instance dual-var challenge state)
///   - alpha (per-instance from previous layer)
///   - transcript (per-instance, independent FS transcripts)
///   - scratch pad (per-instance bookkeeping)
///
/// At each round of phase 1 (and phase 2 if applicable):
///   1. ONE batched GPU dispatch covers all N instances' poly_eval_at.
///   2. Per-instance: absorb evals into transcript, squeeze challenge.
///   3. ONE batched GPU dispatch covers all N instances' receive_challenge.
///
/// Returns `Vec<(vx_claim, vy_claim)>` of length N. The challenges in
/// `challenge_per_instance` are written-through to the next layer's input.
///
/// Bit-exact correctness: each instance's CPU-side reductions
/// (`unpack_and_combine`, `coef_combine_vec`) are identical to the
/// single-instance helper's. The GPU kernel correctness is verified by
/// `cuda_dispatch::batched_correctness_tests`. Composing both gives
/// bit-exact equivalence to running `sumcheck_prove_gkr_layer` N times.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn sumcheck_prove_gkr_layer_batched<F: FieldEngine, T: Transcript>(
    layer_per_instance: Vec<&CircuitLayer<F>>,
    challenge_per_instance: &mut [ExpanderDualVarChallenge<F>],
    alpha_per_instance: Vec<Option<F::ChallengeField>>,
    transcript_per_instance: &mut [T],
    sp_per_instance: Vec<&mut ProverScratchPad<F>>,
    mpi_config: &impl MPIEngine,
    is_output_layer: bool,
) -> Vec<(F::ChallengeField, Option<F::ChallengeField>)> {
    let n = layer_per_instance.len();
    assert_eq!(challenge_per_instance.len(), n);
    assert_eq!(alpha_per_instance.len(), n);
    assert_eq!(transcript_per_instance.len(), n);
    assert_eq!(sp_per_instance.len(), n);

    // Clone challenges into the helper so the caller's slice stays
    // unborrowed for the writeback at the end.
    let challenges_owned: Vec<ExpanderDualVarChallenge<F>> =
        challenge_per_instance.iter().cloned().collect();

    let mut helper = BatchedSumcheckGkrVanillaHelper::new(
        layer_per_instance.clone(),
        challenges_owned,
        alpha_per_instance,
        sp_per_instance,
        mpi_config,
        is_output_layer,
    );

    helper.prepare_simd();
    helper.prepare_mpi();

    // gkr phase 1 over variable x
    helper.prepare_x_vals();
    let input_var_num = helper.instances[0].input_var_num;
    for i_var in 0..input_var_num {
        let evals_per_instance =
            helper.poly_evals_at_rx(i_var, SUMCHECK_GKR_DEGREE, mpi_config);
        let r_per_instance: Vec<F::ChallengeField> = evals_per_instance
            .iter()
            .zip(transcript_per_instance.iter_mut())
            .map(|(evals, transcript)| {
                transcript_io::<F::ChallengeField, T>(mpi_config, evals, transcript)
            })
            .collect();
        helper.receive_rx(i_var, &r_per_instance);
    }

    helper.prepare_simd_var_vals();
    let simd_var_num = helper.instances[0].simd_var_num;
    for i_var in 0..simd_var_num {
        let evals_per_instance =
            helper.poly_evals_at_r_simd_var(i_var, SUMCHECK_GKR_SIMD_MPI_DEGREE, mpi_config);
        let r_per_instance: Vec<F::ChallengeField> = evals_per_instance
            .iter()
            .zip(transcript_per_instance.iter_mut())
            .map(|(evals, transcript)| {
                transcript_io::<F::ChallengeField, T>(mpi_config, evals, transcript)
            })
            .collect();
        helper.receive_r_simd_var(i_var, &r_per_instance);
    }

    helper.prepare_mpi_var_vals(mpi_config);
    let mpi_logsize = mpi_config.world_size().trailing_zeros() as usize;
    for i_var in 0..mpi_logsize {
        let evals_per_instance = helper.poly_evals_at_r_mpi_var(i_var, SUMCHECK_GKR_SIMD_MPI_DEGREE);
        let r_per_instance: Vec<F::ChallengeField> = evals_per_instance
            .iter()
            .zip(transcript_per_instance.iter_mut())
            .map(|(evals, transcript)| {
                transcript_io::<F::ChallengeField, T>(mpi_config, evals, transcript)
            })
            .collect();
        helper.receive_r_mpi_var(i_var, &r_per_instance);
    }

    let vx_claim_per_instance = helper.vx_claim_per_instance();
    for (vx, transcript) in vx_claim_per_instance.iter().zip(transcript_per_instance.iter_mut()) {
        transcript.append_field_element(vx);
    }

    // gkr phase 2 over variable y (skipped per-instance flag follows
    // the layer; all instances share the same layer so the flag is
    // identical across them).
    let skip_phase_two = layer_per_instance[0].structure_info.skip_sumcheck_phase_two;
    let vy_claim_per_instance: Vec<Option<F::ChallengeField>> = if !skip_phase_two {
        helper.prepare_y_vals(mpi_config);
        for i_var in 0..input_var_num {
            let evals_per_instance =
                helper.poly_evals_at_ry(i_var, SUMCHECK_GKR_DEGREE, mpi_config);
            let r_per_instance: Vec<F::ChallengeField> = evals_per_instance
                .iter()
                .zip(transcript_per_instance.iter_mut())
                .map(|(evals, transcript)| {
                    transcript_io::<F::ChallengeField, T>(mpi_config, evals, transcript)
                })
                .collect();
            helper.receive_ry(i_var, &r_per_instance);
        }
        let vy = helper.vy_claim_per_instance(mpi_config);
        for (v, transcript) in vy.iter().zip(transcript_per_instance.iter_mut()) {
            transcript.append_field_element(v);
        }
        vy.into_iter().map(Some).collect()
    } else {
        vec![None; n]
    };

    // Drain per-instance final state. Take `instances` out of the
    // helper first so the borrow on `challenge_per_instance` (held via
    // helper.instances[*].challenge) is released before we re-borrow
    // mutably to write back the new challenges.
    let instances = std::mem::take(&mut helper.instances);
    drop(helper);

    let mut out = Vec::with_capacity(n);
    for (i, inst) in instances.into_iter().enumerate() {
        let rx = inst.rx;
        let ry = if !skip_phase_two { Some(inst.ry) } else { None };
        let r_simd = inst.r_simd_var;
        let r_mpi = inst.r_mpi_var;

        challenge_per_instance[i] = ExpanderDualVarChallenge::new(rx, ry, r_simd, r_mpi);
        out.push((vx_claim_per_instance[i], vy_claim_per_instance[i]));
    }
    out
}
