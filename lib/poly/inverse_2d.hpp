// Inverse of a bivariate power series modulo 998244353: g = 1 / f mod (x^rows, y^cols), f_00 != 0.
// Newton iteration in x on the Kronecker substitution y = z, x = z^S. Design: lib/poly/notes.md.
//
//   const poly::Inverse2d plan(rows, cols);
//   poly::Arena arena(poly::Transform::words(plan.lg()) + poly::Arena::footprint(plan.f_words()) +
//                     poly::Arena::footprint(plan.g_words()) + poly::Arena::footprint(plan.scratch_words()));
//   poly::Transform t(arena, plan.lg());
//   auto f = arena.take(plan.f_words()), g = arena.take(plan.g_words());  // zero-filled
//   // f[plan.offset() + i * plan.stride() + j] = coefficient of x^i y^j, i < rows, j < cols
//   plan.run(t, f, g, arena.take(plan.scratch_words()));
//   // row i of 1 / f: cols coefficients at g + i * plan.stride() for i < plan.split(), else at
//   // f + plan.offset() + i * plan.stride()
//
// plan.cost() estimates the time, so a caller can compare Inverse2d(rows, cols) with the
// transposed Inverse2d(cols, rows).
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/inverse.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

class Inverse2d {
public:
    // rows, cols >= 1. Precisions 1, ..., ceil(rows / 4), ceil(rows / 2), rows; each step takes the
    // shortest transforms that stride 2 cols - 1 allows, and the stride is the largest all allow.
    Inverse2d(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), stride_(cols) {
        std::size_t precision[64], count = 0;  // rows, ceil(rows / 2), ..., 2
        for (std::size_t k = rows; k > 1; k = (k + 1) / 2) precision[count++] = k;
        for (std::size_t i = count; i-- > 0;) {
            const std::size_t to = precision[i], from = i + 1 < count ? precision[i + 1] : 1;
            const std::size_t words = from == 1 ? 2 * cols - 1 : (to - 1) * (2 * cols - 1) + cols;
            steps_[steps_count_++] = {from, to, length(words), 0, from == 1 ? Half::kLower : Half::kBoth};
        }
        if (rows > 2) {
            std::size_t stride = ~std::size_t(0);
            for (std::size_t i = 1; i < steps_count_; ++i)
                stride = std::min(stride, (steps_[i].length - cols) / (steps_[i].to - 1));
            stride_ = stride;
            for (std::size_t i = 1; i < steps_count_; ++i) {
                // Rows k .. k' - 1 of the products sit at shift + [k S, (k' - 1) S + cols): in the
                // upper half if a shift can move them there.
                Step& s = steps_[i];
                const std::size_t start = s.from * stride, end = (s.to - 1) * stride + cols;
                s.shift = start >= s.length / 2 ? 0 : s.length / 2 - start;
                if (end + s.shift <= s.length) s.half = Half::kUpper;
                else s.shift = 0;
            }
        }
        split_ = rows > 2 ? steps_[steps_count_ - 1].from : rows;
        for (std::size_t i = 0; i < steps_count_; ++i)
            if (!in_place(i)) work_ = std::max(work_, steps_[i].length);
    }

    // Words between the starts of consecutive rows, at least 2 cols - 1 when rows > 2.
    std::size_t stride() const { return stride_; }

    // Where row 0 of f starts in f's buffer; the last step's rows of 1 / f start there too.
    std::size_t offset() const { return rows_ > 2 ? steps_[steps_count_ - 1].shift : 0; }

    // Rows 0 .. split() - 1 of 1 / f end in g's buffer, the others in f's.
    std::size_t split() const { return split_; }

    // Words of f's buffer: f, then (rows > 2) the last step's work.
    std::size_t f_words() const {
        return rows_ > 2 ? std::max(offset() + size(rows_), steps_[steps_count_ - 1].length) : size(rows_);
    }

    std::size_t g_words() const { return size(split_); }

    // The Transform needs lg_max >= lg().
    int lg() const {
        int lg = inverse_log(cols_);
        for (std::size_t i = 0; i < steps_count_; ++i) lg = std::max(lg, std::countr_zero(steps_[i].length));
        return lg;
    }

    // The steps' transforms of g_k at offset 0, their work (but the last's) at footprint(work_).
    std::size_t scratch_words() const {
        const std::size_t last = steps_count_ ? steps_[steps_count_ - 1].length : 0;
        return std::max({Arena::footprint(work_) + work_, Arena::footprint(last), inverse_scratch(cols_)});
    }

    // Estimated time in ns on Zen 3 (transforms ~0.064 ns per word and level, leaf products ~0.7 ns
    // per word), to choose the variable of the iteration.
    double cost() const {
        double total = 0;
        for (std::size_t len = 64; len <= (std::size_t(1) << inverse_log(cols_)) && cols_ > detail::kInverseBase;
             len *= 2)
            total += step_cost(len);
        for (std::size_t i = 0; i < steps_count_; ++i) total += step_cost(steps_[i].length);
        return total;
    }

    // 1 / f mod (x^rows, y^cols) in the layout above. f: f_words() words, f from offset(), zero
    // between rows; overwritten when rows > 2. g: g_words() words, zero-filled. scratch:
    // scratch_words() words. t: lg_max >= lg(). Spans from an Arena; none overlap.
    void run(const Transform& t, std::span<std::uint32_t> f, std::span<std::uint32_t> g,
             std::span<std::uint32_t> scratch) const {
        const std::span<const std::uint32_t> rows = f.subspan(offset(), size(rows_));
        inverse(t, rows.first(cols_), g.first(cols_), scratch);
        const std::size_t room = Arena::footprint(work_);
        for (std::size_t i = 0; i < steps_count_; ++i) {
            const Step& s = steps_[i];
            const std::span<std::uint32_t> gt = scratch.first(s.length);
            if (s.from == 1) first_step(t, rows, g, gt, scratch.subspan(room, s.length));
            else if (in_place(i)) step(t, s, rows, g, gt, f.first(s.length));
            else step(t, s, rows, g, gt, scratch.subspan(room, s.length));
        }
    }

private:
    // Rows from .. to - 1 of g from rows 0 .. from - 1, with transforms of length `length`; the
    // products are computed shift words up, only their `half` half.
    struct Step {
        std::size_t from, to, length, shift;
        Half half;
    };

    static std::size_t length(std::size_t words) {
        return std::max(std::size_t(1) << Transform::kMinLog, std::bit_ceil(words));
    }

    static double step_cost(std::size_t len) {
        return double(len) * (5 * 0.064 * std::countr_zero(len) + 2 * 0.7);
    }

    std::size_t size(std::size_t rows) const { return (rows - 1) * stride_ + cols_; }

    // The last step of rows > 2 works in f's buffer (f is last read there) and leaves its rows there.
    bool in_place(std::size_t i) const { return rows_ > 2 && i + 1 == steps_count_; }

    // Row 1: e = f_1 g_0 mod y^cols, g_1 = -g_0 e mod y^cols, transforms of length >= 2 cols - 1.
    void first_step(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                    std::span<std::uint32_t> gt, std::span<std::uint32_t> work) const {
        t.forward(g.first(cols_), 0, gt);
        t.cyclic_product(f.subspan(stride_, cols_), 0, work, gt, Half::kLower);
        t.cyclic_product(work.first(cols_), 0, work, gt, Half::kLower, detail::kP - 1);
        std::copy_n(work.begin(), cols_, g.begin() + std::ptrdiff_t(stride_));
    }

    // Rows k .. k' - 1 from g_k = g mod x^k (k >= 2), with Z = z^S, transforms of length n, a = shift:
    //   e = z^a f g_k mod (z^n - 1), rows k .. k' - 1 (each row's words past cols cleared);
    //   rows k .. k' - 1 of g = rows k .. k' - 1 of -(Z^k e) g_k mod (z^n - 1).
    // Exact while n >= a + (k' - 1) S + cols: products have degree < a + (k + k' - 2) S + 2 cols - 1,
    // so the terms that wrap land below row k, and a row's terms (degree < 2 cols - 1 <= S in y)
    // stay in it. work may be f's buffer with f at offset a (in place); the rows then stay in work.
    void step(const Transform& t, const Step& s, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
              std::span<std::uint32_t> gt, std::span<std::uint32_t> work) const {
        const std::size_t k = s.from, rows = s.to - k, S = stride_, start = s.shift + k * S;
        t.forward(g.first((k - 1) * S + cols_), 0, gt);
        t.cyclic_product(f.first((s.to - 1) * S + cols_), s.shift, work, gt, s.half);
        std::uint32_t* const e = work.data() + start;
        for (std::size_t i = 0; i + 1 < rows; ++i) std::fill(e + i * S + cols_, e + (i + 1) * S, 0);
        t.cyclic_product(work.subspan(start, (rows - 1) * S + cols_), start, work, gt, s.half, detail::kP - 1);
        if (work.data() + s.shift == f.data()) return;
        for (std::size_t i = 0; i < rows; ++i) std::copy_n(e + i * S, cols_, g.begin() + std::ptrdiff_t((k + i) * S));
    }

    std::size_t rows_, cols_, stride_, split_ = 0, work_ = 0;  // work_: the longest work outside f
    std::array<Step, 64> steps_{};
    std::size_t steps_count_ = 0;
};

}  // namespace poly
