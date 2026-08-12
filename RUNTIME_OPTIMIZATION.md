# Point-LIO Jetson Hot-Path Runtime Optimization

Goal: reduce per-frame wall-clock latency of the Point-LIO fork on the Jetson deployment
target (Orin NX), without changing its numerical output. This was scoped as a **serial
hot-path optimization pass**: same algorithm, same map structure (ikd-Tree), same
residual model, no parallelization added to the hot loop — just removing wasted work and
enabling compiler-level optimizations that were previously off. Branch: `speedup/hotpath2`
(off `fresh_main`), merged.

## Why this was needed

The bottleneck is single-core clock/IPC on the serial per-point EKF update loop — the
same code runs roughly 5x faster on an x86 desktop than on the Jetson's Cortex-A78AE
cores. Point-LIO's already-optimized baseline (branch `speedup/ekf`, merged earlier:
`prof_timing.h` instrumentation, `-DNDEBUG` + `-mcpu=cortex-a78ae`, a scalar fast path
for the common `dof_Measurement==1` case, removal of a wasted `h_x` zero-fill) still
wasn't enough — but critically, the ~30ms/frame figure being quoted for that baseline
was never *directly measured*: it was reconstructed by summing separately-profiled
per-operation means (`0.0074ms × ~3856 calls` + `map_incremental` + `predict`), which
misses fixed overhead and can't capture run-to-run tail variance. Step 0 below fixes
that before trusting any further optimization deltas.

## Step 0 — trusted frame-level timer

Added a single wall-clock span in `src/laserMapping.cpp`, from `sync_packages()`
success (ingress: a full LiDAR frame is in hand) to the odometry publish call (egress),
tagged `frame_e2e` in the existing `prof_timing.h` percentile-tracking instrumentation
(`POINTLIO_PROFILE` build flag). This is the number to trust — not a sum of per-op
means, which don't add up to wall-clock reality and hide tail behavior.

Also required environment hygiene before any numbers were meaningful: `nvpmodel -m 0`
(max-performance power mode) confirmed via `nvpmodel -q`, and — discovered *later*,
during unrelated detection-pipeline testing, not caught at the time this step was
first done — **`jetson_clocks` (the command that actually locks CPU governor/frequency
to max, not just selects the power mode) needs to be run explicitly**; without it,
`ondemand` governor leaves several cores well below `MaxFreq` even in `MAXN` mode. Any
Jetson timing number in this document assumes both were applied; if you see numbers
noticeably worse than reported here, check `jetson_clocks --show` first.

## Step 1 — reuse `dyn_share_modified` across calls (no realloc)

`update_iterated_dyn_share_modified()` and `update_iterated_dyn_share_IMU()` in
`include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp` each constructed a fresh
`dyn_share_modified<scalar_type>` on **every call** (i.e. every point). Its `h_x`/`z`
fields are `Eigen::Dynamic`-sized, so a freshly-stack-constructed object always
heap-allocates on first resize — even though the resize target (1×12 / 1×1) never
changes call to call. Fixed by hoisting it to a persistent private class member
(`dyn_share_buf_`), shared by both functions via reference. Safe because both are
single-threaded and called strictly sequentially, and both unconditionally overwrite
every field they read before reading it.

Verified bit-for-bit equivalent via a standalone (no-ROS) algebra test — this class of
change can't alter results, only allocation count, but the project's verification
policy treats "no ROS bag-replay comparisons, isolated algebra tests only" as a hard
constraint (bag replay is provably non-reproducible here — see `causal_vs_batch.py`'s
own comments on message-delivery timing jitter compounding through the recursive EKF;
a diff there would be meaningless either way).

## Step 2 — `.noalias()` on the scalar-path Kalman update

In the `dof_Measurement == 1` fast path, marked `.noalias()` on the two matrix products
where Eigen would otherwise allocate a temporary to guard against aliasing that isn't
actually possible: `PHT = P_.block<n,12>(0,0) * h_row.transpose()` and the final
covariance update. The covariance update itself (`P_ -= (K_*h_row) * P_.block<12,n>(0,0)`)
**is** genuinely self-aliasing — it reads `P_` via the block on the right while writing
all of `P_` on the left — so `.noalias()` can't be applied directly there; fixed by an
explicit pre-copy of the block (`P_top`) before the in-place subtraction, which *is*
alias-safe. Verified against 2000 randomized trials in a standalone test: max
difference `0.000e+00` — this is expected (the pre-copy makes the two formulations
compute identically, not just approximately), not a "close enough" result.

## Steps 3 & 4 — audited, found not applicable (documented as genuine no-ops)

The handoff plan's Step 3 (hoist per-frame constants — quaternion→rotation-matrix,
extrinsic transform assembly — out of the per-point loop) doesn't apply to this fork's
architecture: state updates continuously per point rather than once per frame, so
there's no frame-constant to hoist without changing the algorithm. Step 4 (exploit
measurement-Jacobian block sparsity) was already present — the existing
`block<n,12>`/`block<12,n>` slicing (inherited from upstream) already avoids dense
N×N work. Both are recorded here specifically so a future pass doesn't re-derive the
same conclusion from scratch, or worse, assume they were simply skipped.

## Step 5 — LTO/IPO

Enabled via CMake's own IPO support (`check_ipo_supported()` +
`CMAKE_INTERPROCEDURAL_OPTIMIZATION`), not a raw `-flto` flag, so it applies
consistently to both compile and link steps. Needed `cmake_policy(SET CMP0069 NEW)`
since this file pins `cmake_minimum_required` to 3.5 (older than CMP0069's 3.9
default-NEW version). Helps because the hot loop is dense with small function calls
across `Estimator.cpp`, `laserMapping.cpp`, and the header-only IKFoM/Eigen code they
call into — cross-translation-unit inlining matters here in a way it wouldn't for a
less call-heavy loop.

## Separately: `far_field_sampling.warmup_seconds` disabled (config, not code)

Not part of the hot-path optimization plan above, but resolves the same underlying
tension (compute cost vs. real-time budget) at the config level. `config/mid360.yaml`'s
`far_field_sampling.warmup_seconds` was `5.0` (full raw point density for the first 5s,
to fix a previously-diagnosed cold-start recall problem — sampling from frame 0 dropped
the causal detector's cold-start recall at a fixed test point from 0.99 to 0.35). That
warmup window's *own* full-density compute was found to make Point-LIO fall behind
real-time on the Jetson target during the warmup transient (confirmed via `--rate 1.0`
real-time bag playback vs. `--start-offset` skipping past it). Set to `0.0` — **a
deliberate, explicitly user-confirmed regression** of the earlier recall fix, trading
cold-start recall back down in exchange for no warmup-transient frame drops. If recall
matters more in a future revision, re-enabling this (or a lighter non-zero-density
warmup) is the fix — this is a live, known tradeoff, not an oversight.

## Results

**x86 (Ryzen 7 6800U), before any of this segment's work** — `frame_e2e`: mean 37.56ms,
p50 10.92ms, p95 134.00ms, p99 143.39ms, max 153.31ms. The mean badly understates the
tail — this is exactly the kind of distribution Step 0 exists to catch.

**Real Jetson Orin NX, after Steps 0/1/2/5 + the warmup-disable config change above**
(these landed together, so this is not a clean per-step attribution — see "Not yet
done" below): `frame_e2e` mean 6.60ms, p50 6.77ms, p95 8.16ms, p99 9.34ms, max 9.71ms.
Tight distribution, comfortably under a 90ms frame budget.

**Caveat surfaced later, not yet re-measured under it:** the detection-pipeline testing
that found the missing `jetson_clocks` issue (see Step 0) also found Point-LIO's own
`frame_e2e` running noticeably higher (~21-38ms mean, depending on how much of a
concurrent detection workload was also running on the same 8 cores) than the isolated
6.60ms figure above, even after applying `jetson_clocks`. The isolated 6.60ms number
is real but was measured with Point-LIO as the only significant CPU consumer; it is
**not** representative of Point-LIO's latency when a detection process is also
competing for the same cores. Re-measuring `frame_e2e` in isolation, freshly, with
`jetson_clocks` confirmed applied beforehand, would separate "did jetson_clocks alone
close most of the gap" from "is the rest genuinely CPU contention with a concurrent
process" — not yet done.

## Not yet done

- **Step 6 (system-level, not raw runtime but fixes variance):** core-pinning the hot
  thread (`taskset`, ideally an `isolcpus`-isolated core) and `SCHED_FIFO` for the
  ROS2 executor thread. The original plan flagged this as a *prerequisite* for
  trusting small per-step deltas above noise — not done here, and now additionally
  motivated by the concurrent-process contention observed above.
- **3-run variance report** for the Step 0 baseline (plan asked for "30±2ms or
  30±15ms"-style reporting) — only single runs have been done on both x86 and Jetson.
- **`-mcpu=cortex-a78ae` + LTO combined confirmation on Jetson** — self-reports via a
  CMake STATUS message during the Jetson build but hasn't been explicitly re-checked
  this segment (note: Jetson's default GCC 9.4.0 does not support `-mcpu=cortex-a78ae`
  in at least one toolchain check performed during unrelated work in this repo — worth
  confirming which compiler is actually in use for the Point-LIO build itself, not
  just the detection binding, which explicitly needed `g++-10`).

## Appendix: bigger levers considered, explicitly out of scope for this pass

Each of these relaxes one of the pass's hard constraints (no map-structure change, no
residual-model change, no point-count change) and was deliberately not pursued here —
recorded so a future pass doesn't have to re-derive the same landscape:

- **Cached-surfel map** (relaxes "no map-structure change"). `nn_search` was ~45% of
  reconstructed frame cost; a hash-voxel map that caches a pre-fit plane per voxel
  (from incremental point moments, not stored raw points) turns a query into an O(1)
  hash lookup with no per-query k-NN/PCA. A **naive** hash-voxel swap is not expected
  to help on its own — this is a documented finding from prior work (Surfel-LIO,
  LIO-GVM), not a guess: if the hash grid still gathers k-neighbors and re-fits a
  plane at query time, only insertion got cheaper. The win only comes from caching the
  fitted plane itself. A correct, standalone, tested (synthetic-data-verified)
  implementation of the caching idea exists at `include/surfel_map.h` — **not wired
  into the hot path**, by design; integration risk and real-hardware timing haven't
  been evaluated.
- **Same-plane clustering** (relaxes "no residual-model change"). Merges same-plane
  point-to-plane associations in the measurement Jacobian (see BA-LINS, MSC-LIO) to
  avoid recomputing the latest-IMU-pose Jacobian per point. Not attempted.
- **Fewer points** (relaxes "no point-count change", i.e. touches
  `far_field_sampling`/`preprocess` further). A linear lever, but double-edged here
  specifically because far-field points feed both the EKF *and* the detector's
  free-space evidence — cutting them for runtime helps Point-LIO but can cost detector
  recall, per the warmup discussion above. Not attempted beyond the warmup-window
  change already covered.
