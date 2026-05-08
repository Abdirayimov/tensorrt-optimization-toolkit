#include "trt_toolkit/builder/dynamic_shapes.hpp"

#include <sstream>
#include <stdexcept>

namespace trt_toolkit::builder {

namespace {

std::string shape_to_string(const std::vector<std::int64_t>& s) {
    std::ostringstream o;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i > 0) o << 'x';
        o << s[i];
    }
    return o.str();
}

void check_compatible(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b,
                      const char* what) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(std::string("ShapeRange: ") + what + " rank mismatch");
    }
}

}  // namespace

bool ShapeRange::is_static() const noexcept {
    return min_shape == opt_shape && opt_shape == max_shape;
}

std::string ShapeRange::to_string() const {
    std::ostringstream o;
    o << input_name << ": min=" << shape_to_string(min_shape)
      << " opt=" << shape_to_string(opt_shape) << " max=" << shape_to_string(max_shape);
    return o.str();
}

ShapeRange static_shape(const std::string& input_name,
                        const std::vector<std::int64_t>& shape) {
    ShapeRange r;
    r.input_name = input_name;
    r.min_shape = shape;
    r.opt_shape = shape;
    r.max_shape = shape;
    return r;
}

ShapeRange batched_shape(const std::string& input_name,
                         const std::vector<std::int64_t>& base_shape,
                         std::int64_t min_batch, std::int64_t opt_batch,
                         std::int64_t max_batch) {
    if (base_shape.empty()) {
        throw std::invalid_argument("batched_shape: base_shape must include the batch slot");
    }
    ShapeRange r;
    r.input_name = input_name;
    r.min_shape = base_shape;
    r.opt_shape = base_shape;
    r.max_shape = base_shape;
    r.min_shape[0] = min_batch;
    r.opt_shape[0] = opt_batch;
    r.max_shape[0] = max_batch;
    return r;
}

void ShapeProfile::add(ShapeRange range) {
    if (range.input_name.empty()) {
        throw std::invalid_argument("ShapeRange.input_name must be non-empty");
    }
    check_compatible(range.min_shape, range.opt_shape, "min vs opt");
    check_compatible(range.opt_shape, range.max_shape, "opt vs max");
    for (std::size_t i = 0; i < range.min_shape.size(); ++i) {
        if (range.min_shape[i] > range.opt_shape[i] ||
            range.opt_shape[i] > range.max_shape[i]) {
            throw std::invalid_argument("ShapeRange: must satisfy min <= opt <= max in all axes");
        }
    }
    ranges_.push_back(std::move(range));
}

}  // namespace trt_toolkit::builder
