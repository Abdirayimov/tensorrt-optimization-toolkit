#include "trt_toolkit/accuracy/diff_reporter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace trt_toolkit::accuracy {

namespace {

constexpr double kEps = 1e-12;

}  // namespace

void DiffReport::format(std::ostream& os) const {
    os << std::scientific << std::setprecision(4) << "n=" << element_count
       << "  max_abs=" << max_abs_diff << "  mean_abs=" << mean_abs_diff
       << "  max_rel=" << max_rel_diff << "  mean_rel=" << mean_rel_diff
       << "  cos_sim=" << cosine_similarity;
}

DiffReport compare_arrays(const float* expected, const float* actual, std::size_t n) {
    if (expected == nullptr || actual == nullptr) {
        throw std::invalid_argument("compare_arrays: null pointer");
    }
    DiffReport r;
    r.element_count = n;
    if (n == 0) return r;

    double sum_abs = 0.0;
    double sum_rel = 0.0;
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double a = expected[i];
        const double b = actual[i];
        const double abs_diff = std::abs(a - b);
        sum_abs += abs_diff;
        if (abs_diff > r.max_abs_diff) r.max_abs_diff = abs_diff;

        const double denom = std::max({std::abs(a), std::abs(b), kEps});
        const double rel = abs_diff / denom;
        sum_rel += rel;
        if (rel > r.max_rel_diff) r.max_rel_diff = rel;

        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    r.mean_abs_diff = sum_abs / static_cast<double>(n);
    r.mean_rel_diff = sum_rel / static_cast<double>(n);

    const double denom = std::sqrt(na) * std::sqrt(nb);
    r.cosine_similarity = (denom > kEps) ? dot / denom : 0.0;
    return r;
}

DiffReport compare_arrays(const std::vector<float>& expected,
                          const std::vector<float>& actual) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("compare_arrays: size mismatch");
    }
    return compare_arrays(expected.data(), actual.data(), expected.size());
}

}  // namespace trt_toolkit::accuracy
