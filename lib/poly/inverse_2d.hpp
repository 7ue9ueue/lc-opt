// Inverse of a bivariate power series modulo 998244353: g = 1 / f mod (x^rows, y^cols), f_00 != 0.
// Newton iteration in x on the Kronecker substitution y = z, x = z^S, or for few rows one row at a
// time from products of rows' transforms. Design: lib/poly/notes.md.
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

namespace detail {

// Inverse bottom (Transform::inverse_with) for the sum of `terms` >= 1 leaf products a[k] b[k] of
// transforms, read by index. Each product's windows are filled one product ahead.
struct LeafProductSum {
    static constexpr bool kForward = false, kInverse = true;
    const std::uint32_t* roots;
    const std::uint32_t* inverse_roots;
    const std::uint32_t* const* a;
    const std::uint32_t* const* b;
    std::size_t terms;

    [[gnu::always_inline]] void prepare(std::size_t g, std::size_t k, Window (&window)[4]) const {
        const std::uint32_t* leaves = a[k] + 32 * g;
        const Vec v[4] = {load(leaves), load(leaves + 8), load(leaves + 16), load(leaves + 24)};
        fill_windows(window, v, roots, g);
    }

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        Window window[2][4];
        std::size_t slot = 0;
        prepare(first, 0, window[0]);
        for (std::size_t j = 0; j < count; ++j, out += 32) {
            const std::size_t g = first + j;
            Vec f[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
            for (std::size_t k = 0; k < terms; ++k, slot ^= 1) {
                if (k + 1 < terms) prepare(g, k + 1, window[slot ^ 1]);
                else if (j + 1 < count) prepare(g + 1, 0, window[slot ^ 1]);
                const std::uint32_t* leaves = b[k] + 32 * g;
#pragma GCC unroll 4
                for (int t = 0; t < 4; ++t) f[t] = low(add(f[t], leaf_product(window[slot][t], leaves + 8 * t)));
            }
            inverse_h1(f, Group(inverse_roots, g));
#pragma GCC unroll 4
            for (int t = 0; t < 4; ++t) store(out + 8 * t, f[t]);
        }
    }
};

}  // namespace detail

class Inverse2d {
public:
    static constexpr std::size_t kMaxRowByRow = 32;

    // kAuto: row by row when allowed and its cost is lower.
    enum class Method { kAuto, kNewton, kRowByRow };

    // rows, cols >= 1. Newton: precisions 1, ..., ceil(rows / 4), ceil(rows / 2), rows; each step
    // takes the shortest transforms that stride 2 cols - 1 allows, and the stride is the largest all
    // allow. Row by row (3 <= rows <= kMaxRowByRow): transforms of length >= 2 cols - 1, stride cols.
    Inverse2d(std::size_t rows, std::size_t cols, Method method = Method::kAuto)
        : rows_(rows), cols_(cols), stride_(cols) {
        std::size_t precision[64], count = 0;  // rows, ceil(rows / 2), ..., 2
        for (std::size_t k = rows; k > 1; k = (k + 1) / 2) precision[count++] = k;
        for (std::size_t i = count; i-- > 0;) {
            const std::size_t to = precision[i], from = i + 1 < count ? precision[i + 1] : 1;
            const std::size_t words = from == 1 ? 2 * cols - 1 : (to - 1) * (2 * cols - 1) + cols;
            steps_[steps_count_++] = {from, to, length(words), 0, from == 1 ? Half::kLower : Half::kBoth};
        }
        row_length_ = length(2 * cols - 1);
        const bool allowed = rows >= 3 && rows <= kMaxRowByRow;
        if (allowed && (method == Method::kRowByRow || (method == Method::kAuto && row_by_row_cost() < newton_cost()))) {
            row_by_row_ = true;
            steps_count_ = 0;
            split_ = rows;
            return;
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

    // Words between the starts of consecutive rows, at least 2 cols - 1 for Newton with rows > 2.
    std::size_t stride() const { return stride_; }

    // Where row 0 of f starts in f's buffer; the last step's rows of 1 / f start there too.
    std::size_t offset() const { return in_place(steps_count_ - 1) ? steps_[steps_count_ - 1].shift : 0; }

    // Rows 0 .. split() - 1 of 1 / f end in g's buffer, the others in f's.
    std::size_t split() const { return split_; }

    // Words of f's buffer: f, then (Newton, rows > 2) the last step's work.
    std::size_t f_words() const {
        return in_place(steps_count_ - 1) ? std::max(offset() + size(rows_), steps_[steps_count_ - 1].length)
                                          : size(rows_);
    }

    std::size_t g_words() const { return size(split_); }

    // The Transform needs lg_max >= lg().
    int lg() const {
        int lg = inverse_log(cols_);
        if (row_by_row_) lg = std::max(lg, std::countr_zero(row_length_));
        for (std::size_t i = 0; i < steps_count_; ++i) lg = std::max(lg, std::countr_zero(steps_[i].length));
        return lg;
    }

    // Newton: the steps' transforms of g_k at offset 0, their work (but the last's) at
    // footprint(work_). Row by row: the transforms of rows 1 .. rows - 1 of f, 0 .. rows - 2 of g,
    // and a sum.
    std::size_t scratch_words() const {
        if (row_by_row_) return std::max((2 * rows_ - 1) * Arena::footprint(row_length_), inverse_scratch(cols_));
        const std::size_t last = steps_count_ ? steps_[steps_count_ - 1].length : 0;
        return std::max({Arena::footprint(work_) + work_, Arena::footprint(last), inverse_scratch(cols_)});
    }

    // Estimated time in ns on Zen 3 (transforms ~0.064 ns per word and level, leaf products ~0.7 ns
    // per word), to choose the variable of the iteration.
    double cost() const { return row_by_row_ ? row_by_row_cost() : newton_cost(); }

    // 1 / f mod (x^rows, y^cols) in the layout above. f: f_words() words, f from offset(), zero
    // between rows; overwritten by Newton with rows > 2. g: g_words() words, zero-filled. scratch:
    // scratch_words() words. t: lg_max >= lg(). Spans from an Arena; none overlap.
    void run(const Transform& t, std::span<std::uint32_t> f, std::span<std::uint32_t> g,
             std::span<std::uint32_t> scratch) const {
        const std::span<const std::uint32_t> rows = f.subspan(offset(), size(rows_));
        inverse(t, rows.first(cols_), g.first(cols_), scratch);
        if (row_by_row_) return row_by_row(t, rows, g, scratch);
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

    static double transform_cost(std::size_t len) { return double(len) * 0.064 * std::countr_zero(len); }
    static double product_cost(std::size_t len) { return double(len) * 0.7; }

    // The base inverse of f(0, y).
    double base_cost() const {
        double total = 0;
        for (std::size_t len = 64; len <= (std::size_t(1) << inverse_log(cols_)) && cols_ > detail::kInverseBase;
             len *= 2)
            total += 5 * transform_cost(len) + 2 * product_cost(len);
        return total;
    }

    double newton_cost() const {
        double total = base_cost();
        for (std::size_t i = 0; i < steps_count_; ++i)
            total += 5 * transform_cost(steps_[i].length) + 2 * product_cost(steps_[i].length);
        return total;
    }

    // 5 (rows - 1) transforms and (rows - 1)(rows + 2) / 2 leaf products of the row length.
    double row_by_row_cost() const {
        const double r = double(rows_);
        return base_cost() + 5 * (r - 1) * transform_cost(row_length_) + (r - 1) * (r + 2) / 2 * product_cost(row_length_);
    }

    std::size_t size(std::size_t rows) const { return (rows - 1) * stride_ + cols_; }

    // The last step of Newton with rows > 2 works in f's buffer (f is last read there) and leaves
    // its rows there.
    bool in_place(std::size_t i) const { return !row_by_row_ && rows_ > 2 && i + 1 == steps_count_; }

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

    // g_i = -g_0 (sum over 0 < t <= i of f_t g_(i-t)) mod y^cols for i = 1 .. rows - 1, with the
    // transforms F_t, G_t of length n >= 2 cols - 1 (products of two rows do not wrap): the sum by
    // one inverse of a sum of leaf products, then a cyclic product with G_0.
    void row_by_row(const Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                    std::span<std::uint32_t> scratch) const {
        const std::size_t n = row_length_, room = Arena::footprint(n), rows = rows_, cols = cols_;
        const auto transform = [&](std::size_t slot) { return scratch.subspan(slot * room, n); };  // F_t: t - 1; G_i: rows - 1 + i
        const std::span<std::uint32_t> sum = transform(2 * rows - 2), g0 = transform(rows - 1);
        for (std::size_t i = 1; i < rows; ++i) t.forward(f.subspan(i * cols, cols), 0, transform(i - 1));
        t.forward(g.first(cols), 0, g0);
        std::array<const std::uint32_t*, kMaxRowByRow> a, b;
        for (std::size_t i = 1; i < rows; ++i) {
            for (std::size_t k = 0; k < i; ++k) a[k] = transform(k).data(), b[k] = transform(rows - 1 + i - 1 - k).data();
            t.inverse_with(detail::LeafProductSum{t.roots(), t.inverse_roots(), a.data(), b.data(), i}, sum, Half::kLower);
            t.cyclic_product(sum.first(cols), 0, sum, g0, Half::kLower, detail::kP - 1);
            std::copy_n(sum.begin(), cols, g.begin() + std::ptrdiff_t(i * cols));
            if (i + 1 < rows) t.forward(g.subspan(i * cols, cols), 0, transform(rows - 1 + i));
        }
    }

    std::size_t rows_, cols_, stride_, split_ = 0, work_ = 0;  // work_: the longest work outside f
    std::size_t row_length_ = 0;
    bool row_by_row_ = false;
    std::array<Step, 64> steps_{};
    std::size_t steps_count_ = 0;
};

}  // namespace poly
