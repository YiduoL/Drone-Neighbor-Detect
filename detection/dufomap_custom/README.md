# Custom dufomap binding for Jetson (aarch64)

## Why this exists

The published `dufomap` Python package (`pip install dufomap`) only ships wheels for
`manylinux_x86_64` / `musllinux_x86_64` / `win_amd64` (confirmed via
`pypi.org/pypi/dufomap/json`: no aarch64 wheel, no sdist at all). It has no source
distribution, so there is no official way to build it for Jetson.

The Python-binding source itself (the code that produces `dufomap_bind.*.so`) was
never published: neither `Kin-Zhang/dufomap`'s `main` branch nor its `feature/python`
branch contains any pybind11/nanobind code, and zero commits across the repo's history
mention "nanobind" (confirmed via the GitHub API, not guessed). The compiled `.so`'s
own embedded debug path (`/project/cpp/dufomap/dufomap.cpp`) doesn't match any file
layout in the public repo either. The wheel's own changelog says it uses **nanobind**,
not pybind11.

What **is** public is the underlying C++ library the official bindings wrap:
`UFO::Map`, a header-only octree occupancy-mapping library, included in
`Kin-Zhang/dufomap`'s `include/ufo/`. `src/bind.cpp` in this directory is a from-scratch
nanobind wrapper around that same library, exposing only the three calls the detection
code actually uses:

```python
dufomap(resolution, d_s, d_p, num_threads=0)
dm.run(points, pose, cloud_transform)      # integrate FAR-FIELD points into the map
dm.segment(points, pose, cloud_transform)  # -> per-point 0=static / 1=dynamic labels
```

`python_pkg/dufomap/` is the official package's pure-Python glue (`__init__.py`,
`utils/pose.py`, `utils/__init__.py`) copied with only the changes noted inline
(numpy.typing isn't available on Jetson's numpy 1.17, so that import is wrapped in a
try/except with a plain `np.ndarray` fallback — the types are never runtime-checked,
so this changes nothing observable).

## How `run()`/`segment()` map to UFO::Map calls

Reverse-engineered from `Kin-Zhang/dufomap`'s own reference CLI (`src/dufomap.cpp`),
not guessed:

- `run()` → `ufo::insertPointCloud(map, cloud, sensor_origin, frame_origin, params,
  propagate=true, need_transform=cloud_transform)`. `sensor_origin` must be the
  **world-frame** sensor position (`frame_origin.translation`) regardless of
  `cloud_transform` — UFO's own "FIXME: What is correct?" comment in
  `integration.hpp` shows this isn't transformed internally, so the caller has to
  already hand it a world-frame point. Originally hardcoded to `(0,0,0)`, which
  silently corrupted the ray-casting geometry (see Validation below).
- `segment()` → per-point `map.seenFree(world_point)`: a point is **dynamic** (label 1)
  if the map now believes that location is free space (something was there when
  captured, later rays saw through it); **static** (label 0) otherwise.
- Map type: `ufo::MapType::SEEN_FREE | REFLECTION | LABEL`, matching the reference CLI
  (LABEL/REFLECTION aren't queried here — no clustering/output-map support — but kept
  in the map type so the integration-cost profile matches the official tool as closely
  as possible, since the point of this binding is *representative timing*, not a
  lighter reimplementation).

## Build

See the header comment in `CMakeLists.txt` for exact commands. In short: `pip install
nanobind` (must be ≤2.6.0 on Python 3.8 — 2.14.0 dropped 3.8 support, discovered by
bisecting PyPI release history), `sudo apt install liblz4-dev liblzf-dev`
(`liblzf` is required, not optional — UFO's `point_cloud.hpp` unconditionally
`#include`s `<liblzf/lzf.h>` for a PCD reader this binding never calls), clone
`Kin-Zhang/dufomap` (`feature/python` branch) for the `include/ufo` headers.

**Two header patches were needed for aarch64** (not needed on x86, so easy to miss if
you only test there): `key.hpp`, `code.hpp`, and `ray_caster.hpp` all
unconditionally `#include <immintrin.h>` (x86 SSE/AVX intrinsics — doesn't exist on
ARM). The only things in those files that actually need it are gated behind
`#if defined(__BMI2__)`, which is never defined on an ARM build anyway (no `-mbmi2`
flag possible), so wrapping the *include* in `#if defined(__x86_64__) ||
defined(__i386__)` is a safe, zero-behavior-change fix — not a workaround that changes
what gets compiled on x86.

Also needs GCC ≥10 (`<concepts>` isn't available in GCC 9's libstdc++, even with
`-std=c++20`) — Jetson's default is GCC 9.4.0, but `g++-10` is already installed
(apparently for this exact reason — matches the upstream README's own "install
gcc-10/g++-10" instruction). Pass `-DCMAKE_CXX_COMPILER=g++-10` explicitly.

## Validation status

**Algorithm fidelity (confirmed good):** same recorded Point-LIO output, replayed
offline through both the official x86 package and this binding — 84.7%–99%+
correlation across two different bags of varying detection density, after fixing the
`sensor_origin` bug above (which alone caused ~10x under-detection before the fix).
Residual per-frame differences are consistent with independent-compile floating-point
summation-order noise at voxel boundaries, not a systematic bias (correlation
0.98–0.99, not a directional skew).

**Known open issue (unresolved as of this writing):** on a **full-length** (~400s,
~8000-frame) live run — Point-LIO + this binding consuming live topics via
`causal_live.py`, both on Jetson and reproduced on WSL/x86 to rule out a hardware-
specific cause — `segment()` returns **zero dynamic points for every single frame**,
including the frame range (near the end of the recording) where the official package,
tested on the identical underlying data, detects hundreds of dynamic points per frame.
Point-LIO's own point-cloud output matches between the two runs (99.5% exact match on
raw near-field point *counts*, confirming the input data and frame alignment are
correct) — so this is not a data-pipeline problem, it's specific to this binding's
`segment()`/`run()` behavior over a long run. Shorter runs (≤90s, ≤1500 frames) do
detect correctly and match the official package well (see Algorithm fidelity above),
so whatever is wrong is a function of run *duration* or cumulative call count, not a
one-off bug in the basic algorithm.

Leading hypotheses, not yet confirmed:
- `IntegrationParams::time` (a `float`, incremented by 1 every `run()` call) or some
  other piece of per-call accumulated state growing unboundedly across ~8000 calls and
  eventually pushing the map into a state where `seenFree()` always returns false.
- A difference from the reference CLI's usage pattern: the CLI calls
  `map.propagateModified()` **once**, at the very end of a whole batch; this binding
  calls it (`propagate=true`) on **every** `run()`, which is necessary for causal
  per-frame querying but is an unvalidated-at-scale usage pattern — worth checking
  whether repeated propagation over thousands of calls has a cost/correctness issue
  the official single-shot-at-the-end usage never exercises.
- Not yet ruled out: something specific to `num_threads=0` → `hardware_concurrency()`
  (8 threads on Jetson) combined with TBB (`UFO_PARALLEL`) over a long run — the
  earlier TBB-vs-sequential comparison only covered short (~600-frame) runs.

**Next step to isolate root cause:** reproduce this offline (via `causal_vs_batch.py`,
no live/ROS-timing confound) on a **full-length** recorded Point-LIO output bag,
comparing official vs. custom across the whole run, not just the first ~90s tested so
far. If the custom binding fails the same way offline, the bug is in this binding's
code/algorithm usage, independent of Jetson hardware or ROS live-subscription timing.
