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

## Follow-up pass — combined-workload latency, a branch-confusion bug, and detection backgrounding

The isolated 6.60ms `frame_e2e` figure above could not be reproduced this pass under
what was believed to be the same conditions (`jetson_clocks` confirmed applied,
`lidar_1_ros2` bag, `POINTLIO_PROFILE` on) — repeated measurements this pass gave
~26-34ms mean even with Point-LIO running alone, no detector. The gap is not reconciled;
the exact conditions behind the original 6.60ms figure aren't fully documented (which
bag, how many points/frame). **Treat the isolated 6.60ms number as unverified** and the
~26-34ms range below as the currently-trusted baseline until someone re-derives one
from the other.

### A session-long measurement was invalidated by a launch-file override

`mapping_mid360.launch.py` hardcodes `use_imu_as_input: false` in its inline parameter
dict, which — since ROS2 launch parameter lists are applied in order with later entries
winning — **overrides `mid360.yaml`'s own value**, regardless of what that file says.
`use_imu_as_input: false` means `h_model_output`/`kf_output` is the function that
actually runs every frame, not `h_model_input`/`kf_input`. A large fraction of this
pass's early work (OpenMP-parallelizing the per-point EKF measurement loop, validated
"safe under taskset-isolated combined load" after several rounds of tuning) was applied
to `h_model_input` — the dead one. Once corrected by porting the identical change to
the actually-active `h_model_output`, the result reversed: parallelizing this loop
**regresses** the real workload (frame_e2e 33-36ms mean vs. the serial baseline's
~30.75ms, p95/p99 43-46ms, one spike to 57ms), not a small win. Root cause: each call
to this loop has ~1 point on average in practice (`group_size` profiling metric,
confirmed independent of which function runs — see below), so there's essentially
nothing to parallelize, only thread-spawn/reduction-coordination overhead to pay. Both
functions' OpenMP pragmas are reverted; see the `HISTORY` comments directly above each
loop in `src/Estimator.cpp` for the full account.

**Lesson for future work on this fork:** before trusting *any* profiling number,
confirm which of `h_model_input`/`h_model_output` is actually executing (check the
live `--params-file` temp file a running node was launched with, not `mid360.yaml` in
isolation) — a launch file can silently override the config file.

### Combined workload (Point-LIO + detector running together) is what deployment latency actually depends on

Point-LIO's own `frame_e2e`, measured while a detector process (`causal_live.py`) is
also running concurrently on the same 8 Jetson cores: mean ~26-34ms across several
repeated runs (`--rate 1.0` real-time replay, `taskset` giving Point-LIO cores 0-3 and
the detector cores 4-7, both processes core-isolated from each other). This, not either
component benchmarked alone, is the number that matters for deployment.

### `jetson_clocks` does not persist

Confirmed this pass: the CPU governor silently reverts to `ondemand` (cores idling at
`MinFreq`) independent of any explicit action taken — observed after an unrelated
network/reconnect event, and again after unrelated periods of idle time. **Re-run
`sudo jetson_clocks` and check `jetson_clocks --show` immediately before every timing
run**, not just once per session — a stale "I already did this" assumption produces
numbers 20-40% worse than the true baseline with no obvious symptom other than the
numbers themselves looking bad.

### `-mcpu=cortex-a78ae` is not recognized by this Jetson's actual toolchain

Confirmed via a direct compiler probe (`g++ -mcpu=cortex-a78ae -x c++ -`) on the real
Jetson: GCC 9.4.0 (JetPack 5.1.1's default) does not know `cortex-a78ae` (too new a
core name for this GCC release) and was silently falling back to plain `-O3` with *no*
CPU-specific codegen at all — the `check_cxx_compiler_flag` guard in `CMakeLists.txt`
was doing its job, just landing on a worse fallback than necessary. `cortex-a76` *is*
recognized (same probe) and is a direct microarchitectural ancestor of Cortex-A78AE;
`CMakeLists.txt` now cascades `cortex-a78ae → cortex-a76 → plain -O3`. Combined with
removing a redundant double vector-clear in `ikd-Tree`'s `Nearest_Search` (called
~4750x/frame; see `include/ikd-Tree/ikd_Tree.cpp`), three repeated combined-workload
runs gave 30.98ms mean vs. a same-day 30.02ms baseline (two runs) — **no measurable
difference**, within this setup's normal run-to-run noise (individual runs ranged
25-38ms across this whole pass). Kept anyway: both changes are provably safe
(`cortex-a76` doesn't change program semantics; the removed clear operated on an
already-empty vector, verified by reading the surrounding code, not assumed) and
verified zero-impact on trajectory (9.82cm mean / 23cm max position diff against the
pre-change baseline, not growing over time — consistent with ordinary run-to-run
real-time-replay noise, not a regression) and detection recall (93% dynamic-point IoU).
Real upside unproven; real downside ruled out; the cortex-a76 targeting in particular
is likely to matter more once something else in the hot path is vectorizable NEON code,
which it mostly isn't yet.

### Two more relaxations of the "no residual-model change" constraint, both tried and reverted

Point-LIO's `use_imu_as_input: false` formulation processes points in groups
(`time_seq`) that in practice average ~1 point each (`group_size` profiling metric,
mean ≈1.0) — i.e. one full predict+`nn_search`+`esti_plane`+Kalman-update cycle *per
individual point*, not per frame or per batch. This is the structural reason
parallelizing the per-point loop (above) had nothing to parallelize. Two ways to
reduce the *number* of these cycles were tried:

- **`mapping.ekf_group_batch`** (`common_lib.h`'s `time_compressing()`): merge every N
  natural point-groups into one before the caller's EKF update, so N points share one
  predicted pose and one update instead of each getting their own. Tested at N=2:
  **causes severe trajectory divergence under real motion** (position estimate
  effectively froze ~48s into a 2-minute test, 5-8m off from baseline by the end) *and*
  made latency worse, not better (32.5ms mean, climbing to 34-36ms over the run, vs.
  the serial baseline). Root cause: points forced to share another point's pose breaks
  Point-LIO's per-point motion-distortion correction — the plane residual for the
  earlier point in a merged group gets computed against a pose that's really the later
  point's, which is only a small error at rest but large once the platform has real
  angular velocity (far points amplify small angular errors into large linear ones),
  and that bad residual then biases the EKF state itself, compounding frame to frame.
  Latency didn't improve either because the Kalman-update matrix cost scales with
  total matched points regardless of batching — halving the call count roughly doubles
  the per-call cost, no net win even before the divergence problem. **Do not revisit
  without addressing the shared-pose approximation itself** (e.g. per-point-interpolated
  residuals stacked into one shared solve, which is a different and much larger change).
- **`mapping.ekf_update_stride`**: keep every point's own correctly time-propagated
  pose (predict() still runs for every group, unconditionally) and keep publishing
  every point at full resolution (detection's input is completely unaffected), but
  only run the expensive measurement-correction step (`nn_search`+`esti_plane`+Kalman
  update) on every Nth group — the other groups' points get IMU-propagation-only
  poses, no plane-match correction. This does *not* have `ekf_group_batch`'s
  shared-pose bug (verified: `map_incremental()` was found to need a companion fix,
  `ekf_stride_skip_insert`, so skipped-group points are excluded from map insertion
  entirely rather than routed through a fallback path that turned out to cause runaway
  `ikd-Tree` growth — see that flag's comment in `Estimator.h`). Tested at N=2 with
  that fix in place: **frame_e2e genuinely halved** (~17.7ms mean vs. ~30.75ms
  baseline, stable, no runaway) — but caused a real, if much smaller, drift: ~30-85cm
  position error depending on test, growing over the run, consistent with ordinary IMU
  dead-reckoning drift between corrections (this mechanism is fundamentally different
  from `ekf_group_batch`'s shared-pose bug, and about an order of magnitude smaller in
  practice) — a real accuracy cost for a latency win the project's precision
  requirements ruled out. Both flags are wired end-to-end (parameters, both
  `h_model_input`/`h_model_output` branches, `map_incremental()`) and default to `1`
  (no-op, byte-for-byte original behavior) — kept as documented, working, but
  **not-to-be-enabled** infrastructure so a future revisit doesn't have to
  re-implement the plumbing from scratch, just decide whether the accuracy cost is
  ever acceptable.

### Detection-side win: background the detector's `run()` off the critical path

Unlike the two directions above, this doesn't touch Point-LIO or the residual model at
all, and has **zero accuracy cost, verified two ways**. Causally, the detector's
`segment()` for frame *i* only ever reads map state as of frame *i-1*'s `run()`
completing — never frame *i*'s own `run()`. So `run()` (map integration, measured at
~6-8ms) can execute on a background thread while the rest of frame *i*'s work and frame
*i+1*'s arrival proceed, and the only thing that can still cost a frame anything is
*waiting* for the previous frame's `run()` if it genuinely hasn't finished yet
(measured: ~0.02ms typical, i.e. essentially never blocks in practice — `run()`'s
~6-8ms comfortably fits inside the ~50ms inter-frame gap). Detector critical-path cost
dropped from ~8.2ms to ~0.8ms. Required releasing the GIL in the nanobind bindings
(`detection/dufomap_custom/src/bind.cpp`'s `gil_scoped_release` on both `run()` and
`segment()`) — without it, the background thread would be serialized behind the main
thread's own Python-level work by the GIL, largely negating the benefit. **Verified
bit-identical**: same fixed recorded input replayed through both the backgrounded and
non-backgrounded version of `causal_live.py` gives 100.0000% label agreement, 0 of
2,175,470 points differing, 0.000000m position diff — this isolates the comparison
from Point-LIO's own run-to-run real-time-replay noise (which alone produces ~10-20cm
of point-position "diff" between any two live runs, unrelated to this change — see the
`-mcpu` section above) and shows the ordering-preserving design is not just
theoretically sound but exactly reproduces the non-backgrounded output.

Combined with the above: end-to-end latency (Point-LIO `frame_e2e` + detector critical
path, both measured under the real combined workload) went from ~39ms to **~31.6ms**.

## Current overall status

- Combined-workload baseline (Point-LIO + detector, both running, `jetson_clocks`
  applied, cores split via `taskset`): **~31.6ms** end-to-end mean, up from ~39ms at
  the start of this pass, entirely from the detection-backgrounding change (zero
  accuracy cost) plus the `-mcpu`/redundant-clear fixes (unproven latency benefit, but
  verified zero cost either way).
- No safe path was found to close the remaining gap to a 30ms target. Both directions
  that would (point-count reduction, either EKF-update relaxation above) have a real,
  measured accuracy cost the project's requirements ruled out.
- Not yet done: Direction 2 from the correctness-preserving optimization discussion
  (batch multiple points' *already individually time-corrected* residuals into one
  Kalman solve, rather than either merging poses or skipping corrections) was proposed
  but not attempted — expected upper bound ~15-30% off the EKF-update portion only
  (not `nn_search`, so a smaller ceiling than either direction above), not yet
  validated to be free of the shared-pose bug `ekf_group_batch` hit. `isolcpus` /
  `SCHED_FIFO` (Step 6 above) also still not done — likely to help tail latency (p95/p99)
  more than the mean.
