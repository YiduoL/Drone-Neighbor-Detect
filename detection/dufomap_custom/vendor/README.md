# Vendored UFO::Map headers

`ufo_include/` is a copy of the header-only `include/ufo/` tree from
[Kin-Zhang/dufomap](https://github.com/Kin-Zhang/dufomap) (`feature/python` branch,
pulled 2026-08). Vendored (not fetched at build time) so the build doesn't depend on
that repo staying reachable or unchanged, and so the two patches below travel with the
source instead of needing to be reapplied by hand after every fresh clone. BSD 3-Clause
licensed -- see the license header preserved at the top of each file.

## Patches applied on top of upstream

1. **ARM portability (`key.hpp`, `code.hpp`, `ray_caster/ray_caster.hpp`)**: these files
   unconditionally `#include <immintrin.h>` (x86 SSE/AVX intrinsics header, doesn't
   exist on ARM). The only things in them that actually need it are gated behind
   `#if defined(__BMI2__)` (never defined on an ARM build -- no `-mbmi2` possible), so
   wrapping the *include* in `#if defined(__x86_64__) || defined(__i386__)` is a
   zero-behavior-change fix on x86, and lets the file compile at all on aarch64.

2. **Correctness fix, not just portability (`octree.hpp`, `toKey(coord_t, depth_t)`)**:
   the original code was
   ```cpp
   return static_cast<key_t>(std::floor(coord / node_size_[0])) + half_max_value_;
   ```
   `std::floor(...)` is negative for any coordinate on the negative side of the map
   origin, and casting a **negative floating-point value directly to an unsigned
   integer type is undefined behavior in C++** (not merely implementation-defined).
   x86's codegen for this UB happens to produce the value you'd naively expect, but
   AArch64's `FCVTZU` instruction is architecturally defined to *saturate*
   out-of-range (including all negative) float-to-unsigned conversions to 0 -- so on
   Jetson, every point with a negative local-frame coordinate on any axis silently
   collapsed to `half_max_value_` (the map origin key), corrupting voxel
   indexing/grouping for that point. Confirmed with a standalone test comparing
   `toKey()`/`toCode()` output between x86 and aarch64 builds for identical input
   coordinates: negative-coordinate points diverged, positive-coordinate points
   didn't -- and confirmed as the sole root cause of a real, reproducible bug: on the
   live Jetson pipeline, `dufomap_bind`'s `segment()` returned **zero dynamic points on
   every single frame** of a real multi-thousand-frame recording that the official
   x86 package correctly flags hundreds of dynamic points per frame on (same
   underlying data). Fixed by routing through a signed integer type first (float ->
   signed int truncation is well-defined on both platforms for in-range values), so
   the eventual signed -> unsigned step is a standard, always-well-defined modular
   conversion instead of a direct float -> unsigned cast:
   ```cpp
   return static_cast<key_t>(
              static_cast<std::int_fast64_t>(std::floor(coord / node_size_[0])) +
              static_cast<std::int_fast64_t>(half_max_value_));
   ```
   Verified bit-for-bit identical `toKey()`/`toCode()` output between x86 and aarch64
   builds after this fix, for the same test coordinates that diverged before it. After
   rebuilding with this fix, the live Jetson pipeline's detection output matches the
   x86 reference to r=0.9937 correlation on the same real encounter (previously 0 vs.
   ~400k total dynamic points across the run) -- see `detection/dufomap_custom/README.md`
   for the full investigation.

Both patches are pure C++ semantics fixes, not behavior changes conditioned on
anything other than target architecture / actual undefined behavior -- x86 output is
provably unchanged by either patch (verified directly, not assumed).

## Updating this vendor copy

If you need to pull a newer version of the upstream headers, re-apply both patches
above (search for `FIX (ARM` and `defined(__x86_64__) || defined(__i386__)` in this
tree for a template) -- and re-run the `toKey`/`toCode` standalone comparison test
(described in `detection/dufomap_custom/README.md`) on both an x86 and an aarch64
build before trusting the update, since this exact bug class (silent ARM/x86
numerical divergence, no compiler warning, no crash) is exactly the kind of thing
that's easy to reintroduce without noticing.
