#pragma once
// OS-Deskew C2: fixed-lag smoother that refines Point-LIO's causal state stream.
//
// This is a PURE ADD-ON LAYER: it never feeds back into the main IKFoM filter (kf_output /
// kf_input), it only post-processes the causal state stream Point-LIO already produces, to
// give near-field output points a second, higher-accuracy (but lagged) pose to be transformed
// through. See PIPELINE.md section 5.2 for the design rationale.
//
// It treats the causal state stream itself as a sequence of noisy observations of an
// underlying smooth trajectory, and runs a standard two-pass (forward Kalman filter +
// backward RTS) smoother over a bounded sliding window:
//   - position: per-axis constant-velocity model, measuring BOTH position and velocity
//     from the causal log (both are directly available and informative)
//   - rotation: per-tangent-axis random-walk model in the SO(3) log map relative to the
//     window's reference rotation (the window's first node), measuring the causal log-map
//     deviation
// This is a deliberate simplification of a literal joint IKFoM RTS derivation (which would
// require the filter's internal process Jacobians at every step) -- see PIPELINE.md's
// pilot-experiment section for why this class of approximation is an accepted first step.

#include <deque>
#include <Eigen/Dense>
#include <so3_math.h>

class PoseFixedLagSmoother {
public:
    void configure(double q_pos_accel, double r_pos, double r_vel, double q_rot, double r_rot) {
        q_pos_accel_ = q_pos_accel;
        r_pos_ = r_pos;
        r_vel_ = r_vel;
        q_rot_ = q_rot;
        r_rot_ = r_rot;
    }

    void add_node(double t, const Eigen::Matrix3d &R, const Eigen::Vector3d &p, const Eigen::Vector3d &v) {
        if (!nodes_.empty() && t <= nodes_.back().t) return;  // guard against out-of-order/dup
        Node n;
        n.t = t;
        n.R = R;
        n.p = p;
        n.v = v;
        nodes_.push_back(n);
        pass_valid_ = false;
    }

    double latest_time() const { return nodes_.empty() ? -1e18 : nodes_.back().t; }
    double earliest_time() const { return nodes_.empty() ? 1e18 : nodes_.front().t; }
    size_t size() const { return nodes_.size(); }

    // Drop nodes older than t_cutoff, bounding memory. Call after points needing them have
    // been finalized.
    void trim_before(double t_cutoff) {
        while (!nodes_.empty() && nodes_.front().t < t_cutoff) nodes_.pop_front();
        pass_valid_ = false;
    }

    // (Re)run the forward+backward smoothing pass over the whole current window. O(N) in the
    // number of held nodes (bounded by the retention window, typically ~L*1000 for a 1kHz
    // node rate). Call once per publish cycle, not once per queried point.
    void run_smoothing_pass() {
        const size_t n = nodes_.size();
        if (n == 0) { pass_valid_ = false; return; }
        if (n == 1) {
            nodes_[0].pos_smooth = nodes_[0].p;
            nodes_[0].vel_smooth = nodes_[0].v;
            nodes_[0].rot_smooth = nodes_[0].R;
            pass_valid_ = true;
            ref_R_ = nodes_[0].R;
            return;
        }

        ref_R_ = nodes_[0].R;

        // ---- position: 3 independent constant-velocity KF+RTS passes (x, y, z) ----
        for (int axis = 0; axis < 3; axis++) {
            std::vector<Eigen::Vector2d> x_pred(n), x_filt(n);
            std::vector<Eigen::Matrix2d> P_pred(n), P_filt(n);

            x_filt[0] << nodes_[0].p(axis), nodes_[0].v(axis);
            P_filt[0] = (Eigen::Matrix2d() << r_pos_, 0, 0, r_vel_).finished();
            x_pred[0] = x_filt[0];
            P_pred[0] = P_filt[0];

            for (size_t k = 1; k < n; k++) {
                const double dt = nodes_[k].t - nodes_[k - 1].t;
                Eigen::Matrix2d F;
                F << 1, dt, 0, 1;
                Eigen::Matrix2d Q;
                Q << q_pos_accel_ * dt * dt * dt / 3.0, q_pos_accel_ * dt * dt / 2.0,
                     q_pos_accel_ * dt * dt / 2.0, q_pos_accel_ * dt;

                x_pred[k] = F * x_filt[k - 1];
                P_pred[k] = F * P_filt[k - 1] * F.transpose() + Q;

                Eigen::Vector2d z(nodes_[k].p(axis), nodes_[k].v(axis));
                Eigen::Matrix2d R;
                R << r_pos_, 0, 0, r_vel_;
                Eigen::Matrix2d S = P_pred[k] + R;
                Eigen::Matrix2d K = P_pred[k] * S.inverse();
                x_filt[k] = x_pred[k] + K * (z - x_pred[k]);
                P_filt[k] = (Eigen::Matrix2d::Identity() - K) * P_pred[k];
            }

            std::vector<Eigen::Vector2d> x_smooth(n);
            std::vector<Eigen::Matrix2d> P_smooth(n);
            x_smooth[n - 1] = x_filt[n - 1];
            P_smooth[n - 1] = P_filt[n - 1];
            for (int k = (int) n - 2; k >= 0; k--) {
                const double dt = nodes_[k + 1].t - nodes_[k].t;
                Eigen::Matrix2d F;
                F << 1, dt, 0, 1;
                Eigen::Matrix2d C = P_filt[k] * F.transpose() * P_pred[k + 1].inverse();
                x_smooth[k] = x_filt[k] + C * (x_smooth[k + 1] - x_pred[k + 1]);
                P_smooth[k] = P_filt[k] + C * (P_smooth[k + 1] - P_pred[k + 1]) * C.transpose();
            }

            for (size_t k = 0; k < n; k++) {
                nodes_[k].pos_smooth(axis) = x_smooth[k](0);
                nodes_[k].vel_smooth(axis) = x_smooth[k](1);
            }
        }

        // ---- rotation: 3 independent random-walk KF+RTS passes on the tangent-space
        //      components of Log(ref_R^T * R_k) ----
        std::vector<Eigen::Vector3d> theta_meas(n);
        for (size_t k = 0; k < n; k++) {
            theta_meas[k] = Log<double>(ref_R_.transpose() * nodes_[k].R);
        }

        for (int axis = 0; axis < 3; axis++) {
            std::vector<double> x_pred(n), x_filt(n), P_pred(n), P_filt(n);
            x_filt[0] = theta_meas[0](axis);
            P_filt[0] = r_rot_;
            x_pred[0] = x_filt[0];
            P_pred[0] = P_filt[0];

            for (size_t k = 1; k < n; k++) {
                const double dt = nodes_[k].t - nodes_[k - 1].t;
                x_pred[k] = x_filt[k - 1];
                P_pred[k] = P_filt[k - 1] + q_rot_ * dt;

                const double z = theta_meas[k](axis);
                const double S = P_pred[k] + r_rot_;
                const double K = P_pred[k] / S;
                x_filt[k] = x_pred[k] + K * (z - x_pred[k]);
                P_filt[k] = (1.0 - K) * P_pred[k];
            }

            std::vector<double> x_smooth(n), P_smooth(n);
            x_smooth[n - 1] = x_filt[n - 1];
            P_smooth[n - 1] = P_filt[n - 1];
            for (int k = (int) n - 2; k >= 0; k--) {
                const double C = P_filt[k] / P_pred[k + 1];
                x_smooth[k] = x_filt[k] + C * (x_smooth[k + 1] - x_pred[k + 1]);
                P_smooth[k] = P_filt[k] + C * C * (P_smooth[k + 1] - P_pred[k + 1]);
            }

            for (size_t k = 0; k < n; k++) nodes_[k].theta_smooth(axis) = x_smooth[k];
        }
        for (size_t k = 0; k < n; k++) {
            nodes_[k].rot_smooth = ref_R_ * Exp<double>(Eigen::Vector3d(nodes_[k].theta_smooth));
        }

        pass_valid_ = true;
    }

    // Interpolate the smoothed pose at t_query. Requires run_smoothing_pass() to have been
    // called since the last add_node(), and t_query to fall within [earliest_time(),
    // latest_time()] (extrapolation beyond the window is not attempted).
    bool query(double t_query, Eigen::Matrix3d &R_out, Eigen::Vector3d &p_out) const {
        if (!pass_valid_ || nodes_.size() < 2) return false;
        if (t_query < nodes_.front().t || t_query > nodes_.back().t) return false;

        size_t lo = 0, hi = nodes_.size() - 1;
        while (hi - lo > 1) {
            size_t mid = (lo + hi) / 2;
            if (nodes_[mid].t <= t_query) lo = mid; else hi = mid;
        }
        const double dt = t_query - nodes_[lo].t;
        p_out = nodes_[lo].pos_smooth + nodes_[lo].vel_smooth * dt;

        const double span = nodes_[hi].t - nodes_[lo].t;
        const double frac = span > 1e-9 ? dt / span : 0.0;
        const Eigen::Vector3d theta_interp = (1.0 - frac) * nodes_[lo].theta_smooth + frac * nodes_[hi].theta_smooth;
        R_out = ref_R_ * Exp<double>(Eigen::Vector3d(theta_interp));
        return true;
    }

private:
    struct Node {
        double t;
        Eigen::Matrix3d R;
        Eigen::Vector3d p, v;
        Eigen::Vector3d pos_smooth, vel_smooth, theta_smooth;
        Eigen::Matrix3d rot_smooth;
    };
    std::deque<Node> nodes_;
    Eigen::Matrix3d ref_R_ = Eigen::Matrix3d::Identity();
    bool pass_valid_ = false;

    double q_pos_accel_ = 4.0;   // (m/s^2)^2 -- process noise for constant-velocity position model
    double r_pos_ = 4e-4;        // m^2       -- measurement noise treating causal position as an observation
    double r_vel_ = 1e-2;        // (m/s)^2   -- measurement noise treating causal velocity as an observation
    double q_rot_ = 0.5;         // (rad)^2/s -- process noise for random-walk rotation model
    double r_rot_ = 1e-4;        // rad^2     -- measurement noise treating causal rotation as an observation
};
