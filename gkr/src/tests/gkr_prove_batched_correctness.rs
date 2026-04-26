//! Bit-exact correctness test: `gkr_prove_batched` on N copies of a
//! circuit with N independent transcripts must produce per-instance
//! `(claimed_v, final_challenge)` outputs that exactly match running
//! `gkr_prove` separately on each instance with the same inputs.
//!
//! This is the soundness contract for the batched layer driver. The
//! kernels themselves are tested in `sumcheck::cuda_dispatch`. This
//! test composes the kernels through the helper and the layer driver.

use circuit::Circuit;
use config_macros::declare_gkr_config;
use gkr_engine::{
    FieldEngine, FieldType, GKREngine, GKRScheme, M31x16Config, MPIConfig, MPIEngine, MPISharedMemory,
    Transcript,
};
use gkr_hashers::PoseidonFiatShamirHasher;
use mersenne31::M31x16;
use poly_commit::RawExpanderGKR;
use sumcheck::ProverScratchPad;
use transcript::BytesHashTranscript;

use std::time::Instant;

use crate::utils::*;
use crate::{gkr_prove, gkr_prove_batched};

#[test]
fn test_gkr_prove_batched_matches_single() {
    dev_env_data_setup();

    let universe = MPIConfig::init().unwrap();
    let world = universe.world();
    let mpi_config = MPIConfig::prover_new(Some(&universe), Some(&world));

    declare_gkr_config!(
        CfgM31Poseidon,
        FieldType::M31x16,
        FiatShamirHashType::Poseidon,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );

    do_test::<CfgM31Poseidon>(mpi_config);
}

/// Bench: measure single-prove vs batched-prove wall time on the
/// keccak M31 fixture for various batch sizes. Run with
/// `--ignored --nocapture` to see numbers.
#[test]
#[ignore]
fn bench_gkr_prove_batched_throughput() {
    dev_env_data_setup();

    let universe = MPIConfig::init().unwrap();
    let world = universe.world();
    let mpi_config = MPIConfig::prover_new(Some(&universe), Some(&world));

    declare_gkr_config!(
        CfgM31Poseidon,
        FieldType::M31x16,
        FiatShamirHashType::Poseidon,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );

    do_bench::<CfgM31Poseidon>(mpi_config);
}

fn do_bench<Cfg: GKREngine>(mpi_config: MPIConfig<'_>)
where
    Cfg::FieldConfig: FieldEngine,
{
    let circuit_path = "../".to_owned() + KECCAK_M31_CIRCUIT;
    let witness_path = "../".to_owned() + KECCAK_M31_WITNESS;

    // Single-prove baseline: time one gkr_prove call, repeated 3 times.
    let (mut c, mut w) = Circuit::<Cfg::FieldConfig>::prover_load_circuit::<Cfg>(
        &circuit_path,
        &mpi_config,
    );
    c.load_witness_allow_padding_testing_only(&witness_path, &mpi_config);
    c.evaluate();
    let max_in = c.layers.iter().map(|l| l.input_var_num).max().unwrap();
    let max_out = c.layers.iter().map(|l| l.output_var_num).max().unwrap();

    let mut single_times_us = vec![];
    for _ in 0..3 {
        let mut sp = ProverScratchPad::<Cfg::FieldConfig>::new(
            max_in,
            max_out,
            mpi_config.world_size(),
        );
        let mut t = <Cfg::TranscriptConfig as Transcript>::new();
        t.append_u8_slice(&[0u8]);
        let start = Instant::now();
        let _ = gkr_prove(&c, &mut sp, &mut t, &mpi_config);
        single_times_us.push(start.elapsed().as_micros() as u64);
    }
    let single_us = *single_times_us.iter().min().unwrap();
    println!("[BENCH single ] gkr_prove          : {} us (best of 3)", single_us);

    c.discard_control_of_shared_mem();
    mpi_config.free_shared_mem(&mut w);

    // Batched prove at N = 1, 2, 4, 8, 16 — load N circuits + scratch
    // pads + transcripts and time one gkr_prove_batched call.
    for &n in &[1usize, 2, 4, 8, 16] {
        let mut circuits: Vec<Circuit<Cfg::FieldConfig>> = Vec::with_capacity(n);
        let mut wins = Vec::with_capacity(n);
        for _ in 0..n {
            let (mut c, w) = Circuit::<Cfg::FieldConfig>::prover_load_circuit::<Cfg>(
                &circuit_path,
                &mpi_config,
            );
            c.load_witness_allow_padding_testing_only(&witness_path, &mpi_config);
            c.evaluate();
            circuits.push(c);
            wins.push(w);
        }

        let mut best_us: u64 = u64::MAX;
        for _ in 0..3 {
            let mut sps: Vec<ProverScratchPad<Cfg::FieldConfig>> = circuits
                .iter()
                .map(|c| {
                    let mi = c.layers.iter().map(|l| l.input_var_num).max().unwrap();
                    let mo = c.layers.iter().map(|l| l.output_var_num).max().unwrap();
                    ProverScratchPad::<Cfg::FieldConfig>::new(mi, mo, mpi_config.world_size())
                })
                .collect();
            let mut transcripts: Vec<Cfg::TranscriptConfig> = (0..n)
                .map(|i| {
                    let mut t = <Cfg::TranscriptConfig as Transcript>::new();
                    t.append_u8_slice(&[i as u8]);
                    t
                })
                .collect();
            let circuit_refs: Vec<&Circuit<Cfg::FieldConfig>> = circuits.iter().collect();
            let sp_refs: Vec<&mut ProverScratchPad<Cfg::FieldConfig>> =
                sps.iter_mut().collect();

            let start = Instant::now();
            let _ = gkr_prove_batched(circuit_refs, sp_refs, &mut transcripts, &mpi_config);
            let us = start.elapsed().as_micros() as u64;
            if us < best_us {
                best_us = us;
            }
        }
        let per_inst = best_us as f64 / n as f64;
        let speedup = (single_us as f64) / per_inst;
        println!(
            "[BENCH batched ] N={:2} gkr_prove_batched : {} us total ({:.0} us/inst, {:.2}x vs single)",
            n, best_us, per_inst, speedup,
        );

        for (c, mut w) in circuits.into_iter().zip(wins.into_iter()) {
            c.discard_control_of_shared_mem();
            mpi_config.free_shared_mem(&mut w);
        }
    }
}

fn do_test<Cfg: GKREngine>(mpi_config: MPIConfig<'_>)
where
    Cfg::FieldConfig: FieldEngine,
{
    let circuit_path = match <Cfg::FieldConfig as FieldEngine>::FIELD_TYPE {
        FieldType::M31x16 => "../".to_owned() + KECCAK_M31_CIRCUIT,
        _ => unreachable!("test only configured for M31x16"),
    };
    let witness_path = match <Cfg::FieldConfig as FieldEngine>::FIELD_TYPE {
        FieldType::M31x16 => "../".to_owned() + KECCAK_M31_WITNESS,
        _ => unreachable!(),
    };

    const N: usize = 2;

    // N independent circuit + scratch pad pairs.
    let mut circuits: Vec<Circuit<Cfg::FieldConfig>> = Vec::with_capacity(N);
    let mut shared_windows = Vec::with_capacity(N);
    for _ in 0..N {
        let (mut c, w) = Circuit::<Cfg::FieldConfig>::prover_load_circuit::<Cfg>(
            &circuit_path,
            &mpi_config,
        );
        c.load_witness_allow_padding_testing_only(&witness_path, &mpi_config);
        c.evaluate();
        circuits.push(c);
        shared_windows.push(w);
    }

    let mut sps: Vec<ProverScratchPad<Cfg::FieldConfig>> = circuits
        .iter()
        .map(|c| {
            let max_in = c.layers.iter().map(|l| l.input_var_num).max().unwrap();
            let max_out = c.layers.iter().map(|l| l.output_var_num).max().unwrap();
            ProverScratchPad::<Cfg::FieldConfig>::new(
                max_in,
                max_out,
                mpi_config.world_size(),
            )
        })
        .collect();

    // Per-instance transcript with distinct domain-separation byte.
    let make_transcript = |seed: u8| -> Cfg::TranscriptConfig {
        let mut t = <Cfg::TranscriptConfig as Transcript>::new();
        t.append_u8_slice(&[seed]);
        t
    };

    // === Reference: gkr_prove on each instance, sequentially ===
    let mut ref_results: Vec<(
        <Cfg::FieldConfig as FieldEngine>::ChallengeField,
        gkr_engine::ExpanderDualVarChallenge<Cfg::FieldConfig>,
    )> = Vec::with_capacity(N);
    let mut ref_transcripts: Vec<Cfg::TranscriptConfig> = Vec::with_capacity(N);
    for i in 0..N {
        let mut t = make_transcript(i as u8);
        let (cv, ch) = gkr_prove(&circuits[i], &mut sps[i], &mut t, &mpi_config);
        ref_results.push((cv, ch));
        ref_transcripts.push(t);
    }

    // Reset scratch pads for the batched run.
    for (i, c) in circuits.iter().enumerate() {
        let max_in = c.layers.iter().map(|l| l.input_var_num).max().unwrap();
        let max_out = c.layers.iter().map(|l| l.output_var_num).max().unwrap();
        sps[i] = ProverScratchPad::<Cfg::FieldConfig>::new(
            max_in,
            max_out,
            mpi_config.world_size(),
        );
    }

    // === gkr_prove_batched on the same N instances ===
    let mut batched_transcripts: Vec<Cfg::TranscriptConfig> =
        (0..N).map(|i| make_transcript(i as u8)).collect();
    let circuit_refs: Vec<&Circuit<Cfg::FieldConfig>> = circuits.iter().collect();
    let sp_refs: Vec<&mut ProverScratchPad<Cfg::FieldConfig>> = sps.iter_mut().collect();
    let batched_results =
        gkr_prove_batched(circuit_refs, sp_refs, &mut batched_transcripts, &mpi_config);

    // === Compare per-instance ===
    assert_eq!(batched_results.len(), N);
    for i in 0..N {
        let (ref_cv, ref_ch) = &ref_results[i];
        let (b_cv, b_ch) = &batched_results[i];

        assert_eq!(ref_cv, b_cv, "instance {i}: claimed_v mismatch");
        assert_eq!(ref_ch.rz_0, b_ch.rz_0, "instance {i}: rz_0 (rx) mismatch");
        assert_eq!(ref_ch.rz_1, b_ch.rz_1, "instance {i}: rz_1 (ry) mismatch");
        assert_eq!(ref_ch.r_simd, b_ch.r_simd, "instance {i}: r_simd mismatch");
        assert_eq!(ref_ch.r_mpi, b_ch.r_mpi, "instance {i}: r_mpi mismatch");

        // Transcript-state divergence check: pull a deterministic field
        // element from each transcript at the post-prove state. Equal
        // ⇒ same FS history.
        let ref_check = ref_transcripts[i]
            .generate_field_element::<<Cfg::FieldConfig as FieldEngine>::ChallengeField>(
        );
        let b_check = batched_transcripts[i]
            .generate_field_element::<<Cfg::FieldConfig as FieldEngine>::ChallengeField>(
        );
        assert_eq!(
            ref_check, b_check,
            "instance {i}: transcript-state hash mismatch"
        );
    }

    for (c, mut w) in circuits.into_iter().zip(shared_windows.into_iter()) {
        c.discard_control_of_shared_mem();
        mpi_config.free_shared_mem(&mut w);
    }
}
