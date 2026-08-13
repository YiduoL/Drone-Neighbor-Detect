// Minimal, from-scratch Python binding for the parts of the "dufomap" package that
// causal_vs_batch.py actually uses: dufomap(resolution, d_s, d_p, num_threads),
// .run(points, pose, cloud_transform), .segment(points, pose, cloud_transform).
//
// Why this file exists: the official PyPI "dufomap" package (wheel:
// dufomap_bind.cpython-*.so) has NO aarch64 build, and its nanobind-binding source is
// not published anywhere in Kin-Zhang/dufomap's public git history (checked: zero
// commits mention "nanobind" on either the main or feature/python branch; the .so's
// embedded debug path /project/cpp/dufomap/dufomap.cpp does not match this repo's
// layout on any branch). The underlying C++ library it wraps (UFO::Map, this repo's
// include/ufo/) IS public and header-only, so this file binds that library directly,
// using Kin-Zhang/dufomap's own src/dufomap.cpp (the reference CLI tool) as the ground
// truth for which calls reproduce the official run()/segment() semantics:
//   - run()     -> ufo::insertPointCloud(map, cloud, sensor_origin, frame_origin,
//                  params, propagate=true, need_transform=cloud_transform)
//                  (this is exactly the 3rd insertPointCloud overload in
//                  include/ufo/map/integration/integration.hpp)
//   - segment() -> per-point map.seenFree(world_point): a point is "dynamic" (label 1)
//                  if the map now believes that location is free space (i.e. something
//                  was there when the point was captured, but later rays saw through
//                  it); "static" (label 0) otherwise. This is the exact predicate
//                  src/dufomap.cpp's batch/final query pass uses
//                  ("if (!map.seenFree(p)) cloud_static.push_back(p);"), just evaluated
//                  causally per-frame (immediately after that frame's own run(), with
//                  propagate=true on every insert) instead of once at the end of a
//                  whole dataset.
//
// Map type (ufo::MapType::SEEN_FREE | REFLECTION | LABEL) and the resolution/d_s/d_p ->
// inflate_hits_dist/inflate_unknown parameter mapping are copied from the same
// reference CLI + include/ufo/map/integration/integration_parameters.hpp. LABEL/
// REFLECTION aren't queried by this binding (no clustering/output-map support, mirrors
// causal_vs_batch.py's actual usage) but are kept in the map type to match the official
// tool's integration-cost profile as closely as possible, since the point of this
// binding is a *representative* timing measurement, not a lighter reimplementation.
//
// One thing this binding does NOT try to replicate: the official wrapper's
// "hit_extension" constructor flag has no visible corresponding field in
// IntegrationParams as currently published (only inflate_hits_dist/inflate_unknown
// exist) -- interpreted here as a simple gate on whether inflate_hits_dist (d_s) is
// applied at all. causal_vs_batch.py never passes this argument (relies on the
// Python-side default), so this interpretation doesn't affect its behavior either way.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include <ufo/map/integration/integration.hpp>
#include <ufo/map/integration/integration_parameters.hpp>
#include <ufo/map/point_cloud.hpp>
#include <ufo/map/ufomap.hpp>
#include <ufo/math/pose6.hpp>

namespace nb = nanobind;

namespace {

using UfoMap = ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION |
                         ufo::MapType::LABEL>;

// pose is always the Python-side dufomap.utils.pose.pose_check() output:
// [x, y, z, qw, qx, qy, qz] (7 floats, translation then w-first quaternion) -- this is
// also exactly the argument order ufo::Pose6's 7-scalar constructor expects
// (t_x, t_y, t_z, r_w, r_x, r_y, r_z), see include/ufo/math/pose6.hpp.
//
// pose_check() returns a plain Python list (not a numpy array) when its input was
// already length-7 -- std::vector<double> is what nanobind's stl/vector.h caster
// accepts from a Python list (an ndarray parameter would reject a plain list, since
// list doesn't implement the buffer protocol).
ufo::Pose6f poseFromArray(std::vector<double> const &pose) {
    if (pose.size() != 7) {
        throw std::invalid_argument("pose must have length 7: [x,y,z,qw,qx,qy,qz]");
    }
    return ufo::Pose6f(static_cast<float>(pose[0]), static_cast<float>(pose[1]),
                        static_cast<float>(pose[2]), static_cast<float>(pose[3]),
                        static_cast<float>(pose[4]), static_cast<float>(pose[5]),
                        static_cast<float>(pose[6]));
}

ufo::PointCloud cloudFromArray(
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> const &points) {
    if (points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (N, 3)");
    }
    std::size_t n = points.shape(0);
    ufo::PointCloud cloud;
    cloud.reserve(n);
    auto v = points.view();
    for (std::size_t i = 0; i < n; ++i) {
        // CloudElement<Point>'s constructor takes a single Point (Vector3<float>)
        // argument, not 3 separate floats -- construct the Point first.
        cloud.emplace_back(ufo::Point(v(i, 0), v(i, 1), v(i, 2)));
    }
    return cloud;
}

class DufomapCustom {
public:
    DufomapCustom(double resolution, double d_s, double d_p, std::size_t num_threads,
                  bool hit_extension, bool ray_passthrough_hits)
        : map_(static_cast<ufo::node_size_t>(resolution), 17) {
        params_.inflate_hits_dist =
            hit_extension ? static_cast<float>(d_s) : 0.0f;
        params_.inflate_unknown = static_cast<std::size_t>(d_p);
        params_.ray_passthrough_hits = ray_passthrough_hits;
        // "0: auto" per the official docstring -- and the official package's own
        // startup log ("Set num_threads: 16" on a 16-thread machine, for
        // num_threads=0) confirms "auto" means "use the parallel path with a sane
        // default thread count", not "single-threaded". NOT reusing
        // IntegrationParams' own struct default here on purpose: that default is
        // `8 * std::thread::hardware_concurrency()` (128 on a 16-core machine),
        // which pathologically oversubscribes TBB's arena -- confirmed by hanging
        // for 60+s on ~60 frames of real data with the struct default vs completing
        // a single run() in ~7ms with an explicit sane thread count. Matches the
        // official package's own observed behavior (16 threads on a 16-thread
        // machine, i.e. plain hardware_concurrency(), not 8x it) more closely too.
        params_.num_threads = num_threads > 0
                                   ? num_threads
                                   : std::max<std::size_t>(1, std::thread::hardware_concurrency());
        params_.parallel = num_threads != 1;
        map_.reserve(20'000'000);
    }

    void run(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> points,
             std::vector<double> const &pose, bool cloud_transform) {
        ufo::PointCloud cloud = cloudFromArray(points);
        ufo::Pose6f frame_origin = poseFromArray(pose);
        // sensor_origin must be the WORLD-frame sensor position regardless of
        // cloud_transform: insertPointCloud's need_transform-aware overload transforms
        // `points` by frame_origin when need_transform is true, but does NOT transform
        // sensor_origin itself (see the "FIXME: What is correct?" comment in UFO's own
        // include/ufo/map/integration/integration.hpp right above where it forwards
        // sensor_origin unchanged) -- so the caller has to already hand it a world-frame
        // point. frame_origin.translation is exactly that (this is also correct when
        // cloud_transform=false: our near/far points are already world-frame, but the
        // sensor itself still physically sits at the pose's world position for
        // ray-casting purposes). Originally hardcoded to (0,0,0) here, which was wrong
        // and silently corrupted free-space ray geometry -- caught by comparing this
        // binding's dynamic-point counts against the real dufomap package on the same
        // bag (~10x under-detection before this fix, see the detection/dufomap_custom
        // validation notes).
        ufo::insertPointCloud(map_, std::move(cloud), frame_origin.translation,
                               frame_origin, params_, /*propagate=*/true,
                               cloud_transform);
    }

    nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>> segment(
        nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> points,
        std::vector<double> const &pose, bool cloud_transform) {
        ufo::PointCloud cloud = cloudFromArray(points);
        ufo::Pose6f frame_origin = poseFromArray(pose);
        if (cloud_transform) {
            ufo::applyTransform(cloud, frame_origin);
        }

        std::size_t n = cloud.size();
        std::uint8_t *labels = new std::uint8_t[n ? n : 1];
        for (std::size_t i = 0; i < n; ++i) {
            ufo::Point p(cloud[i].x, cloud[i].y, cloud[i].z);
            labels[i] = map_.seenFree(p) ? 1 : 0;
        }

        nb::capsule owner(labels, [](void *p) noexcept {
            delete[] static_cast<std::uint8_t *>(p);
        });
        std::size_t shape[1] = {n};
        return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>>(labels, 1, shape,
                                                                   owner);
    }

private:
    UfoMap map_;
    ufo::IntegrationParams params_;
};

}  // namespace

NB_MODULE(dufomap_bind, m) {
    nb::class_<DufomapCustom>(m, "_dufomap")
        .def(nb::init<double, double, double, std::size_t, bool, bool>(),
             nb::arg("resolution"), nb::arg("d_s"), nb::arg("d_p"),
             nb::arg("num_threads") = 0, nb::arg("hit_extension") = true,
             nb::arg("ray_passthrough_hits") = false)
        // gil_scoped_release: both methods do their real work entirely in C++ (a raw
        // nanobind ndarray view in, either nothing or a freshly-allocated numpy array
        // out via capsule) with no callback into Python in the hot loop, so releasing
        // the GIL here is safe on its own terms. This is what makes it possible to run
        // run() on a background Python thread and have it make real (not
        // GIL-serialized) progress concurrently with the main thread's own Python-level
        // work -- see causal_live.py's run() backgrounding. Correctness of run()/
        // segment() never actually overlapping in time on the shared map_ is enforced
        // by that Python-level synchronization (waiting for the previous frame's run()
        // to finish before the next frame's segment() starts), not by the GIL -- the
        // GIL was never doing that job for us even before this change (two separate
        // calls from two threads would already interleave at the Python bytecode level
        // without a race *within* a single call).
        .def("run", &DufomapCustom::run, nb::arg("points"), nb::arg("pose"),
             nb::arg("cloud_transform") = true, nb::call_guard<nb::gil_scoped_release>())
        .def("segment", &DufomapCustom::segment, nb::arg("points"), nb::arg("pose"),
             nb::arg("cloud_transform") = true, nb::call_guard<nb::gil_scoped_release>());
}
