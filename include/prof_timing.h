#ifndef POINT_LIO_PROF_TIMING_H
#define POINT_LIO_PROF_TIMING_H

// Lightweight, opt-in per-frame timing/sample instrumentation for the Step-0 profiling
// pass (see prompt/optimization plan). Guarded entirely by the POINTLIO_PROFILE compile
// definition (set via the CMake option of the same name, default OFF) so that a normal
// build has none of this code present at all -- not just "disabled at runtime", but not
// compiled in, so it can't add overhead or change codegen/inlining in the hot path.
//
// Usage:
//   PROF_SCOPE("nn_search");           // times the rest of the enclosing block
//   PROF_SAMPLE("dof_measurement", n);  // records a raw (non-timing) value
//   prof::maybe_report();               // call once per frame; prints a summary every
//                                        // PROF_REPORT_EVERY_N_FRAMES frames and resets
#ifdef POINTLIO_PROFILE

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace prof {

struct Accumulator {
    std::vector<double> samples; // ms for timers, raw units for PROF_SAMPLE
};

inline std::unordered_map<std::string, Accumulator> &registry() {
    static std::unordered_map<std::string, Accumulator> r;
    return r;
}

// The per-point hot loop this is used inside of (Estimator.cpp's h_model_input/
// h_model_output) is now OpenMP-parallelized (see the #pragma omp parallel for there) --
// concurrent PROF_SCOPE/PROF_SAMPLE calls from multiple threads would otherwise race on
// this shared unordered_map (registry()[name] can trigger a rehash, which is never safe
// under concurrent access even to different keys). Only matters for POINTLIO_PROFILE
// builds -- a normal build has none of this code at all (see the #else below). The lock
// only guards the map mutation itself, not whatever's being timed.
inline std::mutex &registry_mutex() {
    static std::mutex m;
    return m;
}

inline void record(const std::string &name, double value) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[name].samples.push_back(value);
}

class ScopedTimer {
public:
    explicit ScopedTimer(std::string name) : name_(std::move(name)),
        t0_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0_).count();
        record(name_, ms);
    }
private:
    std::string name_;
    std::chrono::steady_clock::time_point t0_;
};

inline void percentiles(std::vector<double> v, double &mean, double &p50, double &p95,
                         double &p99, double &max_v) {
    if (v.empty()) { mean = p50 = p95 = p99 = max_v = 0.0; return; }
    std::sort(v.begin(), v.end());
    double sum = 0.0;
    for (double x : v) sum += x;
    mean = sum / v.size();
    auto at = [&](double frac) {
        size_t i = std::min(v.size() - 1, (size_t) (frac * v.size()));
        return v[i];
    };
    p50 = at(0.50);
    p95 = at(0.95);
    p99 = at(0.99);
    max_v = v.back();
}

constexpr int PROF_REPORT_EVERY_N_FRAMES = 200;

inline void maybe_report() {
    static int frame_count = 0;
    frame_count++;
    if (frame_count % PROF_REPORT_EVERY_N_FRAMES != 0) return;
    printf("[prof] ---- summary over last %d frames (n=samples per metric) ----\n",
           PROF_REPORT_EVERY_N_FRAMES);
    for (auto &kv : registry()) {
        double mean, p50, p95, p99, max_v;
        percentiles(kv.second.samples, mean, p50, p95, p99, max_v);
        printf("[prof] %-16s n=%6zu mean=%8.4f p50=%8.4f p95=%8.4f p99=%8.4f max=%8.4f\n",
               kv.first.c_str(), kv.second.samples.size(), mean, p50, p95, p99, max_v);
        kv.second.samples.clear();
    }
}

} // namespace prof

#define PROF_CONCAT_INNER(a, b) a##b
#define PROF_CONCAT(a, b) PROF_CONCAT_INNER(a, b)
#define PROF_SCOPE(name) ::prof::ScopedTimer PROF_CONCAT(_prof_scope_timer_, __LINE__)(name)
#define PROF_SAMPLE(name, value) ::prof::record((name), (double) (value))
#define PROF_MAYBE_REPORT() ::prof::maybe_report()

#else // !POINTLIO_PROFILE

#define PROF_SCOPE(name) do {} while (0)
#define PROF_SAMPLE(name, value) do {} while (0)
#define PROF_MAYBE_REPORT() do {} while (0)

#endif // POINTLIO_PROFILE

#endif // POINT_LIO_PROF_TIMING_H
