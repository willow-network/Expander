//! This module implements the core GKR IOP.

use circuit::Circuit;
use gkr_engine::{
    ExpanderDualVarChallenge, ExpanderSingleVarChallenge, FieldEngine, MPIConfig, MPIEngine,
    Transcript,
};
use sumcheck::{sumcheck_prove_gkr_layer, sumcheck_prove_gkr_layer_batched, ProverScratchPad};
use utils::timer::Timer;

#[allow(clippy::type_complexity)]
pub fn gkr_prove<F: FieldEngine>(
    circuit: &Circuit<F>,
    sp: &mut ProverScratchPad<F>,
    transcript: &mut impl Transcript,
    mpi_config: &MPIConfig,
) -> (F::ChallengeField, ExpanderDualVarChallenge<F>) {
    let layer_num = circuit.layers.len();

    let mut challenge: ExpanderDualVarChallenge<F> =
        ExpanderSingleVarChallenge::sample_from_transcript(
            transcript,
            circuit.layers.last().unwrap().output_var_num,
            mpi_config.world_size(),
        )
        .into();

    let mut alpha = None;

    let output_vals = &circuit.layers.last().unwrap().output_vals;
    let claimed_v = F::collectively_eval_circuit_vals_at_expander_challenge(
        output_vals,
        &challenge.challenge_x(),
        &mut sp.hg_evals,
        &mut sp.eq_evals_first_half, // confusing name here..
        mpi_config,
    );

    for i in (0..layer_num).rev() {
        let timer = Timer::new(
            &format!(
                "Sumcheck Layer {}, n_vars {}, one phase only? {}",
                i,
                &circuit.layers[i].input_var_num,
                &circuit.layers[i].structure_info.skip_sumcheck_phase_two,
            ),
            mpi_config.is_root(),
        );

        (_, _) = sumcheck_prove_gkr_layer(
            &circuit.layers[i],
            &mut challenge,
            alpha,
            transcript,
            sp,
            mpi_config,
            i == layer_num - 1,
        );

        if challenge.rz_1.is_some() {
            // TODO: try broadcast beta.unwrap directly
            let mut tmp = transcript.generate_field_element::<F::ChallengeField>();
            mpi_config.root_broadcast_f(&mut tmp);
            alpha = Some(tmp)
        } else {
            alpha = None;
        }
        timer.stop();
    }

    (claimed_v, challenge)
}

/// Batched analog of [`gkr_prove`]: drives N independent GKR proves
/// through the same circuit shape in lockstep, with the per-layer
/// sumcheck rounds dispatched as ONE batched GPU kernel per round
/// across all N instances.
///
/// Inputs:
/// - `circuit_per_instance`: N circuits with identical shape (same
///   layer count, same `input_var_num`/`output_var_num` per layer,
///   same `structure_info`). Witness values differ per instance.
/// - `sp_per_instance`: N owned scratch pads, mutated in place.
/// - `transcript_per_instance`: N independent FS transcripts.
/// - `mpi_config`: shared MPI config (single MPI universe; the batch
///   dimension is orthogonal to MPI ranks).
///
/// Returns `Vec<(claimed_v, final_challenge)>` of length N. Each
/// entry is bit-exactly equal to running `gkr_prove` independently
/// on instance i with the same inputs (verified by tests in this
/// crate's `tests/gkr_prove_batched.rs`).
#[allow(clippy::type_complexity)]
pub fn gkr_prove_batched<F: FieldEngine, T: Transcript>(
    circuit_per_instance: Vec<&Circuit<F>>,
    sp_per_instance: Vec<&mut ProverScratchPad<F>>,
    transcript_per_instance: &mut [T],
    mpi_config: &MPIConfig,
) -> Vec<(F::ChallengeField, ExpanderDualVarChallenge<F>)> {
    let n = circuit_per_instance.len();
    assert_eq!(sp_per_instance.len(), n);
    assert_eq!(transcript_per_instance.len(), n);
    assert!(n > 0, "batched gkr_prove needs at least one instance");

    let layer_num = circuit_per_instance[0].layers.len();
    for c in &circuit_per_instance {
        assert_eq!(c.layers.len(), layer_num, "all circuits must share shape");
    }

    // Per-instance initial challenge sampled from each transcript.
    let mut challenge_per_instance: Vec<ExpanderDualVarChallenge<F>> = transcript_per_instance
        .iter_mut()
        .zip(circuit_per_instance.iter())
        .map(|(t, c)| {
            ExpanderSingleVarChallenge::sample_from_transcript(
                t,
                c.layers.last().unwrap().output_var_num,
                mpi_config.world_size(),
            )
            .into()
        })
        .collect();

    let mut sp_per_instance = sp_per_instance;

    // Per-instance claimed_v at the output layer. Computed once up-front
    // using each instance's own scratch pad.
    let claimed_v_per_instance: Vec<F::ChallengeField> = sp_per_instance
        .iter_mut()
        .zip(circuit_per_instance.iter())
        .zip(challenge_per_instance.iter())
        .map(|((sp, c), ch)| {
            let output_vals = &c.layers.last().unwrap().output_vals;
            F::collectively_eval_circuit_vals_at_expander_challenge(
                output_vals,
                &ch.challenge_x(),
                &mut sp.hg_evals,
                &mut sp.eq_evals_first_half,
                mpi_config,
            )
        })
        .collect();

    // Per-instance alpha (carried across layers).
    let mut alpha_per_instance: Vec<Option<F::ChallengeField>> = vec![None; n];

    for i in (0..layer_num).rev() {
        let timer = Timer::new(
            &format!(
                "Batched sumcheck Layer {} (n={}), n_vars {}, one phase only? {}",
                i,
                n,
                &circuit_per_instance[0].layers[i].input_var_num,
                &circuit_per_instance[0].layers[i]
                    .structure_info
                    .skip_sumcheck_phase_two,
            ),
            mpi_config.is_root(),
        );

        // Snapshot per-layer references for this round.
        let layer_per_instance: Vec<&circuit::CircuitLayer<F>> = circuit_per_instance
            .iter()
            .map(|c| &c.layers[i])
            .collect();
        let alpha_clone = alpha_per_instance.clone();

        // Reborrow scratch pads as a fresh `Vec<&mut ProverScratchPad<F>>`
        // that the layer driver will consume by value.
        let sp_refs: Vec<&mut ProverScratchPad<F>> = sp_per_instance
            .iter_mut()
            .map(|sp_box| &mut **sp_box)
            .collect();

        let _ = sumcheck_prove_gkr_layer_batched(
            layer_per_instance,
            &mut challenge_per_instance,
            alpha_clone,
            transcript_per_instance,
            sp_refs,
            mpi_config,
            i == layer_num - 1,
        );

        // Per-instance: if rz_1 is set, sample a fresh alpha from each
        // transcript independently (mirrors single-instance gkr_prove).
        for (idx, ch) in challenge_per_instance.iter().enumerate() {
            if ch.rz_1.is_some() {
                let mut tmp =
                    transcript_per_instance[idx].generate_field_element::<F::ChallengeField>();
                mpi_config.root_broadcast_f(&mut tmp);
                alpha_per_instance[idx] = Some(tmp);
            } else {
                alpha_per_instance[idx] = None;
            }
        }

        timer.stop();
    }

    claimed_v_per_instance
        .into_iter()
        .zip(challenge_per_instance)
        .collect()
}
