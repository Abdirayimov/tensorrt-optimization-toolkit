#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trt_toolkit::builder {

/// Single dynamic-shape profile entry: min / opt / max for one input.
///
/// Each shape vector must have the same rank; -1 in the underlying
/// engine input definition is replaced by the value at the
/// corresponding position here. Use static_shape() to build the trivial
/// profile when a network has no dynamic axes.
struct ShapeRange {
    std::string input_name;
    std::vector<std::int64_t> min_shape;
    std::vector<std::int64_t> opt_shape;
    std::vector<std::int64_t> max_shape;

    /// True when min == opt == max (no dynamic axes).
    bool is_static() const noexcept;

    /// Human-readable representation, e.g. "input: min=1x3x224x224
    /// opt=8x3x224x224 max=16x3x224x224".
    std::string to_string() const;
};

/// Convenience: a single static shape collapsed to a degenerate range.
ShapeRange static_shape(const std::string& input_name,
                        const std::vector<std::int64_t>& shape);

/// Convenience: build a profile that varies only the batch dimension.
ShapeRange batched_shape(const std::string& input_name,
                         const std::vector<std::int64_t>& base_shape,
                         std::int64_t min_batch, std::int64_t opt_batch,
                         std::int64_t max_batch);

/// Bundle of profiles, one per dynamic input. Engines may have several
/// optimization profiles; for now the toolkit emits one combining all
/// inputs. Multi-profile support is on the roadmap.
class ShapeProfile {
public:
    void add(ShapeRange range);

    const std::vector<ShapeRange>& ranges() const noexcept { return ranges_; }
    bool empty() const noexcept { return ranges_.empty(); }

private:
    std::vector<ShapeRange> ranges_;
};

}  // namespace trt_toolkit::builder
