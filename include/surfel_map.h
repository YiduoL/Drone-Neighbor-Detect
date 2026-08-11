#ifndef POINT_LIO_SURFEL_MAP_H
#define POINT_LIO_SURFEL_MAP_H

// Experimental: a hash-voxel map that caches a pre-fit plane ("surfel") per voxel,
// as an alternative to ikd-Tree's per-query k-NN-search + on-the-fly plane fit
// (Estimator.cpp's esti_plane()). NOT wired into the hot path -- this file is
// free-standing and isn't included anywhere in the current build. Exists so the
// idea can be evaluated (correctness first, then timing on the actual target
// hardware) before deciding whether it's worth the integration risk.
//
// Why this might help: ikdtree.Nearest_Search() + esti_plane() together were
// measured (Step 0 profiling, x86 dev machine) at roughly 38% of the per-point EKF
// update loop's own time (nn_search alone; esti_plane is smaller but not free either).
// Both re-do their work from scratch on every single query, even though the same
// local surface is queried by many nearby points across many frames.
//
// Why a naive "swap ikd-Tree for a hash grid" is NOT expected to help on its own
// (this is the documented Surfel-LIO / LIO-GVM finding, not a guess): if the hash
// grid still gathers k neighbors and re-fits a plane at query time, you've only
// made *insertion* cheaper -- query cost is unchanged or worse (scanning several
// neighbor voxels for a k-NN set costs more than one well-pruned kd-tree
// descent). The only way a voxel structure wins here is by caching the fitted
// plane *once*, incrementally, from a running set of moments -- so a query becomes
// a single hash lookup that returns an already-valid plane, no per-query fit at
// all. That's what this file does.
//
// How the surfel is fit: incrementally, from the first and second moments of the
// points assigned to a voxel (running sums, not stored raw points -- O(1) memory
// per voxel regardless of how many points have ever landed in it). The plane
// normal is the eigenvector of the point-covariance matrix with the smallest
// eigenvalue (the direction of least point-spread -- standard PCA plane fit,
// see e.g. any total-least-squares plane-fitting reference). This is a different
// numerical method than Estimator.cpp's esti_plane() (which solves
// A*x+B*y+C*z=-1 directly via colPivHouseholderQr(), implicitly assuming the
// plane doesn't pass through the local origin) -- the two aren't required to
// agree bit-for-bit, only to each independently be a correct least-squares plane
// fit through the same points. Verified against synthetic known-plane data with a
// standalone (no-ROS, not part of this package's build) test -- see the
// speedup/hotpath2 branch's commit that added this file for the exact test code
// and results.

#include <cstdint>
#include <unordered_map>

#include <Eigen/Dense>

namespace point_lio_experimental {

class SurfelMap {
public:
    // voxel_size: edge length of a cubic voxel (meters).
    // min_points_for_surfel: a voxel's surfel is considered valid (queryable) only
    // once at least this many points have landed in it -- below that, a 3-point (or
    // fewer) covariance-based fit is numerically unreliable (can't even uniquely
    // determine a plane from 2 points, and 3 exactly-fit points can be an
    // arbitrarily bad *local* plane estimate if they happen to be near-collinear).
    explicit SurfelMap(double voxel_size, int min_points_for_surfel = 5)
        : voxel_size_(voxel_size), min_points_(min_points_for_surfel) {}

    // O(1) amortized: update the running moments for this point's voxel. Does NOT
    // eagerly refit the plane on every insert (that would defeat the point) --
    // refit happens lazily, once, the first time a query lands on a voxel whose
    // point count has grown since its cached surfel (if any) was last computed.
    void insert(const Eigen::Vector3d &point) {
        VoxelStats &v = voxels_[voxel_key(point)];
        v.count += 1;
        v.sum += point;
        v.sum_outer += point * point.transpose();
        v.surfel_dirty = true;
    }

    // Returns true and fills `plane` (unit normal (a,b,c) + offset d, satisfying
    // a*x+b*y+c*z+d=0 for points on the fitted plane, normal direction otherwise
    // arbitrary-signed -- same convention esti_plane() uses before its own
    // implicit-normalization step) if this point's voxel has enough points for a
    // valid surfel. Returns false otherwise (caller should treat exactly like the
    // existing point_selected_surf[...] = false path: skip this point for the
    // update, not an error).
    bool query(const Eigen::Vector3d &point, Eigen::Vector4d &plane) const {
        auto it = voxels_.find(voxel_key(point));
        if (it == voxels_.end() || it->second.count < min_points_) {
            return false;
        }
        const VoxelStats &v = it->second;
        if (v.surfel_dirty) {
            // Lazy refit: only recompute if this voxel changed since we last fit it.
            // const_cast is safe here -- refresh_surfel only touches the cache
            // fields (cached_plane, surfel_dirty, surfel_valid), not the moment
            // accumulators (count/sum/sum_outer), so this doesn't change what
            // insert() has recorded, only how much of it we've bothered to
            // convert into a plane so far.
            refresh_surfel(const_cast<VoxelStats &>(v));
        }
        if (!v.surfel_valid) {
            return false; // e.g. degenerate/collinear point set, no well-defined plane
        }
        plane = v.cached_plane;
        return true;
    }

    size_t voxel_count() const { return voxels_.size(); }

private:
    struct VoxelStats {
        int count = 0;
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero(); // sum of p*p^T over points
        bool surfel_dirty = true;
        bool surfel_valid = false;
        Eigen::Vector4d cached_plane = Eigen::Vector4d::Zero();
    };

    int64_t voxel_key(const Eigen::Vector3d &p) const {
        // Signed voxel coordinates, offset so negative coordinates don't collide
        // with positive ones after the bit-packing below (a plain cast of a
        // negative double to int64_t and packing without an offset would make
        // e.g. voxel (-1,0,0) and (huge_positive,0,0) collide once truncated to
        // the packed field width).
        constexpr int64_t OFFSET = 1 << 20; // supports +-1M voxels/axis before wraparound
        int64_t ix = static_cast<int64_t>(std::floor(p.x() / voxel_size_)) + OFFSET;
        int64_t iy = static_cast<int64_t>(std::floor(p.y() / voxel_size_)) + OFFSET;
        int64_t iz = static_cast<int64_t>(std::floor(p.z() / voxel_size_)) + OFFSET;
        // 21 bits/axis (supports the +-1M range above), packed into 63 bits.
        return (ix << 42) | (iy << 21) | iz;
    }

    void refresh_surfel(VoxelStats &v) const {
        v.surfel_dirty = false;
        if (v.count < min_points_) {
            v.surfel_valid = false;
            return;
        }
        const double n = static_cast<double>(v.count);
        Eigen::Vector3d mean = v.sum / n;
        // Sample covariance: E[p p^T] - mean*mean^T, from the running moments --
        // this is the standard single-pass ("Welford-adjacent") covariance formula,
        // no need to revisit individual points.
        Eigen::Matrix3d cov = v.sum_outer / n - mean * mean.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        if (solver.info() != Eigen::Success) {
            v.surfel_valid = false;
            return;
        }
        // Eigen's SelfAdjointEigenSolver returns eigenvalues in ascending order --
        // the smallest-eigenvalue eigenvector is the direction of least point
        // spread, i.e. the plane normal (standard total-least-squares plane fit).
        Eigen::Vector3d normal = solver.eigenvectors().col(0);
        double smallest_eigenvalue = solver.eigenvalues()(0);
        double largest_eigenvalue = solver.eigenvalues()(2);
        // Flatness gate: if the "smallest spread" direction isn't meaningfully
        // smaller than the in-plane spread, these points don't actually describe a
        // plane (e.g. a corner, a sphere-ish cluster, or too few/degenerate
        // points) -- esti_plane()'s caller has an analogous check downstream
        // (the match_s * pd2 * pd2 residual gate in Estimator.cpp), but that only
        // catches individual bad points against an already-computed plane, not a
        // fundamentally non-planar voxel. Threshold is a free parameter, not yet
        // tuned against real data -- flagged here, not hidden.
        constexpr double kFlatnessRatioMax = 0.05;
        if (largest_eigenvalue <= 0.0 || smallest_eigenvalue / largest_eigenvalue > kFlatnessRatioMax) {
            v.surfel_valid = false;
            return;
        }
        normal.normalize();
        double d = -normal.dot(mean);
        v.cached_plane << normal(0), normal(1), normal(2), d;
        v.surfel_valid = true;
    }

    std::unordered_map<int64_t, VoxelStats> voxels_;
    double voxel_size_;
    int min_points_;
};

} // namespace point_lio_experimental

#endif // POINT_LIO_SURFEL_MAP_H
