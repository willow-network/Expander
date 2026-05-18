# Fork deviations from upstream

This is a fork of [PolyhedraZK/Expander](https://github.com/PolyhedraZK/Expander).
It carries the commits below on top of upstream `main`. Listed newest-first.

To compare against upstream:

```
git remote add upstream https://github.com/PolyhedraZK/Expander.git
git fetch upstream
git log --oneline upstream/main..main
```

## Currently carried

### Batched GPU dispatch across SIMD pack lanes

- `8310180` fix(sumcheck/cuda): correct bk_hg + lane stride in GPU→CPU transition
- `319c9fc` fix(sumcheck/cuda): cross-block read-write race in receive_challenge
- `17a14a1` test(sumcheck): batched-vs-single GPU dispatch correctness
- `8613d1c` feat(sumcheck): BatchedGpuSumcheckContext — N circuits in one kernel
- `7e68836` feat(sumcheck): batched GPU dispatch across SIMD pack lanes

Lane-batches the 16 SIMD pack lane CUDA launches into one kernel. Addresses
the ~1% GPU utilization observed with lane-sequential dispatch by amortizing
launch overhead across lanes. Measured 16% speedup on Willow's completeness
benchmarks.

Upstreaming: pending discussion with Polyhedra on whether they want this or
whether `gkr_prove_batch` supersedes it.

### Serdes 32-bit usize fix

- `274e2f4` fix(serdes): usize serializes as fixed-width u64 on 32-bit targets

Required for `wasm32-unknown-unknown` verifier builds (browser, mobile). Wire
format unchanged. Hand-written `ExpSerde for usize` impl that delegates to
`u64::ExpSerde` and `try_from`s on decode.

Upstreaming: [PolyhedraZK/Expander#328](https://github.com/PolyhedraZK/Expander/pull/328)
open. When that merges, drop this commit from the fork.

### Formatting / clippy

- `2eafd08` fix: inline format args to satisfy clippy
- `72ff560` chore: cargo fmt

Cosmetic. Can be upstreamed or dropped.

### CUDA GPU acceleration

- `736c6a1` feat: wire GPU dispatch into sumcheck prover hot path
- `8c80901` feat: add CUDA GPU acceleration for sumcheck, Merkle tree, and PCS transpose

Adds feature-gated `cuda` support to `sumcheck/`, `tree/`, `poly_commit/`.
Activated via per-crate `cuda` feature.

GPU is workload-shape-specific: wins on wide-batch workloads (folding,
cross-instance batching), regresses on narrow per-layer hand-built or
ECC-compiled circuits. See Willow's `docs/research/` for benchmark notes.

If upstream's Blake3 Merkle switch (`36c42cc`) is adopted, the Keccak
Merkle CUDA kernels (`tree/cuda/keccak256*.cu`) become irrelevant and the
GPU code shrinks.

Upstreaming: not pursued yet; depends on Polyhedra's interest.

## Parked work — recoverable from history

### Batched cross-instance GKR prover (`82ad772`)

Preserved on branch [`feat/gkr-prove-batched`](https://github.com/willow-network/Expander/tree/feat/gkr-prove-batched).

Built a `gkr_prove_batched` function (analog of `gkr_prove`) that drives N
independent GKR proves through one layer per round in lockstep, intended to
amortize GPU dispatch overhead across N circuit instances. Bit-exact
correctness verified at N=1, 2, 4 on the keccak M31 fixture.

**Why parked**: end-to-end bench on the keccak M31 fixture (RTX 5090) showed
essentially no whole-prove speedup at any N (0.99×–1.11× at N=1..16). For
ECC-compiled circuit shapes Willow currently runs, per-round GPU compute is
too small to amortize kernel launches, and most rounds run pure-CPU below the
dispatch threshold anyway. Bit-exact correctness is proven; the throughput
hypothesis for these circuit shapes is documented as "no".

**Revival**: `git cherry-pick 82ad772` to bring back. Re-validate against
current `cuda_dispatch.rs` (which has since been touched by `8310180` —
those fixes are now part of main). Most likely revival prompts: new circuit
shapes with wider layers where amortization wins, or new GPU hardware where
launch overhead is dwarfed by per-lane compute.

The bug fixes that surfaced while building this (`download_instance_bk_hg`,
shared `download_instance_ext3` helper) were split out into `8310180` and
remain on main, since they apply to the existing `BatchedGpuSumcheckContext`
that PR #322's lane-batched dispatch relies on.

## Goal

Shrink toward upstream over time. Each commit above either upstreams or
gets retired. When the fork carries zero deviations, drop it and pin
Willow directly to `PolyhedraZK/Expander`.
