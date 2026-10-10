// Conversion from the monomial basis to the Newton basis modulo 998244353 by a product tree.
// x86-64 with AVX2. Design: lib/poly/notes.md (Newton basis).
//
//   poly::Arena arena(poly::to_newton_words(n));
//   poly::to_newton(arena, f, points, c);  // f = sum_k c_k prod_(i < k) (x - points[i]), n of each
//
// With F = rev(f) (n coefficients) and Q_t = prod_(i < t) (1 - p_i x), c_k = [x^(n-1-k)] F / Q_(k+1).
// Node v of points [lo, hi) gets the window W_v = (F / Q_hi)[n - hi, n - lo); split at mid, its
// right child [mid, hi) gets W_v[0, hi - mid) and its left child (W_v Q_r)[hi - mid, hi - lo),
// Q_r = prod_(mid <= i < hi) (1 - p_i x): one middle product per node, where multipoint
// evaluation's descent has two. The root's window is F / Q mod x^n (evaluation.hpp's divide); a
// point's is c_k.
//
// NewtonTree: the trees of evaluation.hpp's PointTree, but lane l holds points [l C, (l + 1) C),
// C = m8 / 8, so every node covers consecutive points (padding points at the end add Newton
// coefficients 0), and only the right children's transforms are kept. The left child's state comes
// from the product, as in PointTree::descend; the right child's from the node's own state, halved
// to its lower half when the left child fills the upper half (no product).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>

#include "lib/poly/evaluation.hpp"

namespace poly {

namespace detail {

// The windows of the nodes of degree D (D vectors each) from their parents' (2D vectors each):
// left child 2i by the middle product with its sibling's product (D + 1 vectors), right child
// 2i + 1 the lower half of the parent's window.
template <std::size_t D>
void newton_level(const std::uint32_t* parents, const std::uint32_t* products, std::uint32_t* out, std::size_t pairs) {
    for (std::size_t i = 0; i < pairs; ++i) {
        const std::uint32_t *w = parents + 16 * D * i, *q = products + 8 * (D + 1) * (2 * i + 1);
        if constexpr (middle_block(D) == 0) middle_lanes(w, D, q, D, out + 16 * D * i);
        else middle_lanes<D, D, middle_block(D)>(w, q, out + 16 * D * i);
        std::copy_n(w, 8 * D, out + 16 * D * i + 8 * D);
    }
}

// The descent below a lane node of degree <= LaneLayout::kBase, slots [lo, hi), from the node's
// window w (hi - lo vectors): the windows of single slots into values[8k, 8k + 8) for slot k. As
// BaseDescent: products first (schoolbook), then the windows; a power of two of slots level by
// level, other counts split at the middle, recursively.
class NewtonBase {
public:
    NewtonBase(const PointSlots& slots, std::uint32_t* values) : slots_(slots), values_(values) {}

    void run(std::size_t lo, std::size_t hi, const std::uint32_t* w) {
        const std::size_t count = hi - lo;
        if (count == 1) return store(values_ + 8 * lo, load(w));
        if (std::has_single_bit(count)) return run_levels(lo, count, w);
        free_ = memory_;
        build(1, lo, hi);
        descend(1, lo, hi, w);
    }

private:
    // count = 2^k slots: products of degree 1 (the slots), 2, .. count / 2, level by level, but
    // node 0's (no right child needs it); then windows of degree count / 2, .. 1 (into values).
    void run_levels(std::size_t lo, std::size_t count, const std::uint32_t* w) {
        const std::uint32_t* products[6];
        std::uint32_t* at = memory_;
        for (std::size_t k = 0; k < count; ++k) slots_.load(lo + k, at + 16 * k);
        products[0] = at;
        at += 16 * count;
        for (std::size_t d = 1, j = 1; 2 * d < count; d *= 2, ++j) {
            with_degree(d, [&]<std::size_t D>() {
                if constexpr (D <= LaneLayout::kBase / 4)  // 2D < count <= kBase
                    product_level<D>(products[j - 1] + 16 * (D + 1), at + 8 * (2 * D + 1), count / (2 * D) - 1);
            });
            products[j] = at;
            at += 8 * (2 * d + 1) * (count / (2 * d));
        }
        std::uint32_t* const windows[2] = {at, at + 8 * count};
        const std::uint32_t* parents = w;
        for (std::size_t d = count / 2, j = std::countr_zero(count) - 1;; d /= 2, --j) {
            std::uint32_t* const out = d == 1 ? values_ + 8 * lo : windows[j & 1];
            with_degree(d, [&]<std::size_t D>() { newton_level<D>(parents, products[j], out, count / (2 * D)); });
            if (d == 1) return;
            parents = out;
        }
    }

    // The products of node v = [lo, hi) and of its descendants (heap order), as BaseDescent's.
    const std::uint32_t* build(std::size_t v, std::size_t lo, std::size_t hi) {
        std::uint32_t* const c = free_;
        if (hi - lo == 1) {
            slots_.load(lo, c);
            free_ += 16;
            return c;
        }
        const std::size_t mid = lo + (hi - lo) / 2;
        product_[2 * v] = build(2 * v, lo, mid);
        product_[2 * v + 1] = build(2 * v + 1, mid, hi);
        if (v == 1) return nullptr;
        std::uint32_t* const out = free_;
        multiply_lanes_any(product_[2 * v], mid - lo, product_[2 * v + 1], hi - mid, out);
        free_ += 8 * (hi - lo + 1);
        return out;
    }

    void descend(std::size_t v, std::size_t lo, std::size_t hi, const std::uint32_t* w) {
        if (hi - lo == 1) return store(values_ + 8 * lo, load(w));
        const std::size_t mid = lo + (hi - lo) / 2;
        std::uint32_t* const left = free_;
        free_ = left + 8 * (mid - lo);
        middle_lanes(w, hi - mid, product_[2 * v + 1], mid - lo, left);
        descend(2 * v, lo, mid, left);
        descend(2 * v + 1, mid, hi, w);
        free_ = left;
    }

    // Products: as BaseDescent's (<= 222 vectors for 32 slots); windows: at most 2 * 32 vectors.
    alignas(32) std::uint32_t memory_[8 * 320];
    const std::uint32_t* product_[64];
    std::uint32_t* free_ = memory_;
    const PointSlots& slots_;
    std::uint32_t* values_;
};

// A product tree's right children's transforms (its Keep, ProductTree's place_right): each in the
// room place() gives, found by its parent's split point. Also the root's product transform (in the
// tree's scratch).
class RightStore {
public:
    struct Keeper {
        RightStore* store;
        std::uint32_t* place_right(std::size_t, std::size_t mid, std::size_t, std::size_t words) const { return store->place(mid, words); }
        void root(std::span<const std::uint32_t> transform) const { store->root_ = transform; }
    };

    void reset(std::span<std::uint32_t> words, std::span<std::uint32_t> index) { words_ = words, used_ = 0, at_ = index.data(); }

    std::uint32_t* place(std::size_t mid, std::size_t words) {
        const std::size_t need = Arena::footprint(words);
        if (used_ + need > words_.size()) std::abort();
        at_[mid] = std::uint32_t(used_);
        used_ += need;
        return words_.data() + at_[mid];
    }

    // The transform of the right child of the node split at mid.
    const std::uint32_t* right(std::size_t mid) const { return words_.data() + at_[mid]; }

    std::span<const std::uint32_t> root() const { return root_; }

private:
    std::span<const std::uint32_t> root_;
    std::span<std::uint32_t> words_;
    std::size_t used_ = 0;
    std::uint32_t* at_ = nullptr;  // by split point: offsets in words_
};

// Words of the right children's transforms a tree keeps at node [lo, hi) and below: the same
// recursion as ProductTree::build.
template <class Layout, class Tree>
std::size_t right_words(const Tree& tree, std::size_t lo, std::size_t hi) {
    if (hi - lo == 1 || tree.degree(lo, hi) <= Layout::kBase) return 0;
    const std::size_t mid = tree.split(lo, hi), words = Layout::kWords * Layout::length(tree.degree(lo, hi));
    return Arena::footprint(words) + right_words<Layout>(tree, lo, mid) + right_words<Layout>(tree, mid, hi);
}

// The descent's state at node v of length L is f_v times the transform of a polynomial X whose
// coefficients [L - s_v, L) are v's window; u = 1 / f_v is passed down.

// In place: p's first half becomes 2 f_v times the transform of length L/2 of X[L/2, L):
// P_lo - T(X mod (x^(L/2) + 1)), the second term the forward of inverse_upper(P_hi, -1).
inline void halve_upper(const TreeTransform& t, std::uint32_t* p, std::size_t words) {
    const std::size_t h = words / 2;
    t.inverse_upper({p + h, h}, {p + h, h}, kP - 1);
    t.forward({p + h, h}, 0, {p + h, h});
    for (std::size_t i = 0; i < h; i += 8) store(p + i, add_mod(load(p + i), load(p + h + i)));
}

// In place: p's first half becomes 2 f_v times the transform of length L/2 of X mod x^(L/2):
// P_lo + T(X mod (x^(L/2) + 1)).
inline void halve_lower(const TreeTransform& t, std::uint32_t* p, std::size_t words) {
    const std::size_t h = words / 2;
    t.inverse_upper({p + h, h}, {p + h, h}, 1);
    t.forward({p + h, h}, 0, {p + h, h});
    for (std::size_t i = 0; i < h; i += 8) store(p + i, add_mod(load(p + i), load(p + h + i)));
}

inline std::uint32_t half(std::uint32_t u) { return ntt::detail::multiply_mod(u, (kP + 1) / 2); }

// In place: p becomes the coefficients of u f_v X: with A = inverse(P_lo, u / 2) and
// B = inverse_upper(P_hi, -u / 2), X_lo = A - B and X_hi = A + B (as PointTree::coefficients). A
// single vector (n = 8, standard layout) is its own transform.
inline void to_coefficients(const TreeTransform& t, std::uint32_t* p, std::size_t words, std::uint32_t u) {
    if (words == 8) return t.inverse({p, 8}, {p, 8}, u);
    const std::size_t h = words / 2;
    const std::uint32_t c = half(u);  // nonzero
    t.inverse({p, h}, {p, h}, c);
    t.inverse_upper({p + h, h}, {p + h, h}, kP - c);
    for (std::size_t i = 0; i < h; i += 8) {
        const Vec a = load(p + i), b = load(p + h + i);
        store(p + i, subtract_mod(a, b)), store(p + h + i, add_mod(a, b));
    }
}

// a[i] = 2 a[i] mod P for i < n (canonical).
inline void double_mod(std::uint32_t* a, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const Vec x = load_unaligned(a + i);
        store_unaligned(a + i, add_mod(x, x));
    }
    for (; i < n; ++i) a[i] = a[i] >= kP - a[i] ? a[i] - (kP - a[i]) : 2 * a[i];
}

}  // namespace detail

// The product trees of Q = prod (1 - a_i x) over points a_i in lane order (lane l holds points
// [l C, (l + 1) C)), padded with zeros to m8 = 8 C, with the right children's transforms kept, and
// the Newton descent.
//
// Memory: the scratch is [X][Y]. X holds the two trees' split tables, then their stacks (the lane
// tree's, then the top tree's, which ends with Q and its transform). Y holds the lane products for
// the top tree, then is free for the caller's division (division_scratch()); descend() uses X's
// stack part and Y.
class NewtonTree {
public:
    // Arena words a NewtonTree for n >= 1 points takes, with at least reserve words of
    // division_scratch().
    static std::size_t words(std::size_t n, std::size_t reserve) {
        const Sizes z(n, reserve);
        const std::size_t levels = std::bit_width(z.count) + 1;
        const std::size_t kept_lanes = levels * (16 * z.count + 16 * (z.count / 32 + 2));  // right children: 8 L <= 16 s words each
        const std::size_t kept_top = 7 * Arena::footprint(z.top_length);
        return TreeTransform::words(log_length(z.m8)) + Arena::footprint(z.m8) + Arena::footprint(z.x) + Arena::footprint(z.y) +
               Arena::footprint(kept_lanes) + Arena::footprint(z.count) + Arena::footprint(kept_top) + Arena::footprint(8);
    }

    // Points with padding for n points: 8 C for C slots per lane, C rounded up to a power of two
    // above 0.65 of it. The transforms then keep their lengths (nodes of more than half their
    // length), and every node takes the halving paths: faster above 0.65 (lib/poly/notes.md).
    static std::size_t padded_size(std::size_t n) {
        const std::size_t count = (n + 7) / 8, full = std::bit_ceil(count);
        return 8 * (20 * count > 13 * full ? full : count);
    }

    NewtonTree(Arena& arena, std::span<const std::uint32_t> points, std::size_t reserve)
        : m8_(padded_size(points.size())), count_(m8_ / 8), t_(arena, log_length(m8_)) {
        using namespace detail;
        const Sizes z(points.size(), reserve);
        points_ = arena.take(m8_);
        lane_order(points, points_);
        slots_ = {points_.data(), count_};
        const std::span<std::uint32_t> x = arena.take(z.x);
        y_ = arena.take(z.y);
        stack_ = x.subspan(z.lane_prefix + z.top_prefix);

        lanes_.emplace(t_, slots_, x, RightStore::Keeper{&lane_store_});
        lane_store_.reset(arena.take(right_words<LaneLayout>(*lanes_, 0, count_)), arena.take(count_));
        const LaneTree::Root lane_root = lanes_->root();

        const std::size_t stride = z.stride;
        std::uint32_t* const columns = y_.data();
        const std::uint32_t size = std::uint32_t(count_ + 1);
        const std::uint32_t sizes[8] = {size, size, size, size, size, size, size, size};
        std::uint32_t* const out[8] = {columns, columns + stride, columns + 2 * stride, columns + 3 * stride,
                                       columns + 4 * stride, columns + 5 * stride, columns + 6 * stride, columns + 7 * stride};
        lane_columns(lane_root.coefficients.data(), lane_root.length + 1, sizes, out);
        lane_products_ = {columns, stride, std::uint32_t(count_)};

        top_.emplace(t_, lane_products_, x.subspan(z.lane_prefix), RightStore::Keeper{&top_store_});
        top_store_.reset(arena.take(right_words<StandardLayout>(*top_, 0, 8)), arena.take(8));
        product_ = top_->root().coefficients.first(m8_ + 1);
    }

    NewtonTree(const NewtonTree&) = delete;
    NewtonTree& operator=(const NewtonTree&) = delete;

    // Points with padding: a multiple of 8.
    std::size_t size() const { return m8_; }

    // Q = prod (1 - a_i x) over size() points: size() + 1 coefficients in Montgomery form; and
    // its transform mod (x^L - 1), L = StandardLayout::length(size()). Valid until descend().
    std::span<const std::uint32_t> product() const { return product_; }
    std::span<const std::uint32_t> product_transform() const { return top_store_.root(); }

    // Scratch free until descend(): at least reserve words.
    std::span<std::uint32_t> division_scratch() const { return y_; }

    // values (size() words, canonical, word 8 k + l for point l C + k) = c times the points'
    // windows, from the root's window w (size() coefficients, canonical; overwritten). values may
    // be w.
    void descend(std::span<std::uint32_t> w, std::span<std::uint32_t> values, std::uint32_t c) {
        using namespace detail;
        Stack stack({stack_.data(), std::size_t(y_.data() + y_.size() - stack_.data())});  // X's stacks, then Y
        const std::size_t lane_length = LaneLayout::length(std::uint32_t(count_));
        const RootState root = root_state(w, c, stack);
        NewtonBase base(slots_, values.data());
        if (count_ <= LaneLayout::kBase) return base.run(0, count_, root.state + 8 * (lane_length - count_));
        lanes(0, count_, root.state, root.u, stack, base);
    }

    // out[l C + k] = values[8 k + l] for l C + k < out.size(): from lane order.
    void natural_order(const std::uint32_t* values, std::span<std::uint32_t> out) const {
        using namespace detail;
        const std::size_t n = out.size();
        for (std::size_t k = 0; k < count_; k += 8) {
            Vec r[8];
            for (std::size_t i = 0; i < 8; ++i) r[i] = k + i < count_ ? load(values + 8 * (k + i)) : _mm256_setzero_si256();
            transpose8(r);  // r[l]: points l C + k .. l C + k + 7
            for (std::size_t l = 0; l < 8; ++l) {
                const std::size_t first = l * count_ + k, here = std::min({std::size_t(8), count_ - k, n > first ? n - first : 0});
                if (here == 8) {
                    store_unaligned(out.data() + first, r[l]);
                } else {
                    alignas(32) std::uint32_t t[8];
                    store(t, r[l]);
                    std::copy_n(t, here, out.begin() + std::ptrdiff_t(first));
                }
            }
        }
    }

private:
    using LaneTree = ProductTree<LaneLayout, detail::PointSlots, detail::RightStore::Keeper>;
    using TopTree = ProductTree<StandardLayout, detail::LaneProducts, detail::RightStore::Keeper>;

    // Words of the scratch's parts for n points: X (split tables, then stacks for the build and
    // the descent), Y (the lane products, then the caller's reserve and the rest of the descent).
    struct Sizes {
        std::size_t m8, count, lane_length, top_length, stride, lane_prefix, top_prefix, x, y;  // stride: of the lane products

        Sizes(std::size_t n, std::size_t reserve)
            : m8(padded_size(n)),
              count(m8 / 8),
              lane_length(LaneLayout::length(std::uint32_t(count))),
              top_length(StandardLayout::length(std::uint32_t(m8))),
              stride((count + 8) / 8 * 8),
              lane_prefix(Arena::footprint(count + 1)),
              top_prefix(Arena::footprint(8 + 1)) {
            // A build's stack: the root's coefficients and transform, then one coefficient buffer
            // per level below (each left child is built in its parent's room): < 3 root lengths.
            const std::size_t lane_stack = 3 * Arena::footprint(8 * (lane_length + 1)) + 2048;
            const std::size_t top_stack = 3 * Arena::footprint(top_length + 1) + 2048;
            const std::size_t stack = std::max(lane_stack, top_prefix + top_stack);
            // The descent: the lane root's state and the top leaves (8 max(8, lane_length) words
            // each), the top root's state, products of the top levels (< 2 top lengths) and their
            // right children (< 1); in the lane tree, products (< 2 lane roots) and right children
            // (< 1).
            const std::size_t descent = 2 * Arena::footprint(8 * std::max<std::size_t>(8, lane_length)) +
                                        4 * Arena::footprint(top_length) + 3 * Arena::footprint(8 * lane_length) + 4096;
            x = lane_prefix + stack;
            y = std::max({8 * stride, reserve, descent > stack - top_prefix ? descent - (stack - top_prefix) : 0});
        }
    };

    // Largest transform: the top tree's root, or 8 lanes of the lane tree's root.
    static int log_length(std::size_t m8) {
        const std::size_t lanes = 8 * LaneLayout::length(std::uint32_t(m8 / 8)), top = StandardLayout::length(std::uint32_t(m8));
        return std::countr_zero(std::max(lanes, top));
    }

    // out[8 k + l] = point l C + k (0 past the points): PointTree's order, point 8 k + l in slot k
    // of lane l.
    static void lane_order(std::span<const std::uint32_t> points, std::span<std::uint32_t> out) {
        using namespace detail;
        const std::size_t n = points.size(), count = out.size() / 8;
        for (std::size_t k = 0; k < count; k += 8) {
            Vec r[8];
            for (std::size_t l = 0; l < 8; ++l) {
                const std::size_t first = l * count + k, here = std::min({std::size_t(8), count - k, n > first ? n - first : 0});
                if (here == 8) {
                    r[l] = load_unaligned(points.data() + first);
                } else {
                    alignas(32) std::uint32_t t[8] = {};
                    std::copy_n(points.begin() + std::ptrdiff_t(first), here, t);
                    r[l] = load(t);
                }
            }
            transpose8(r);  // r[i]: slot k + i of each lane
            for (std::size_t i = 0; i < 8 && k + i < count; ++i) store(out.data() + 8 * (k + i), r[i]);
        }
    }

    struct RootState {
        std::uint32_t* state;
        std::uint32_t u;
    };

    // The lane tree's root state from the root's window w (overwritten), as PointTree::lane_state
    // with the Newton descent through the top tree.
    RootState root_state(std::span<std::uint32_t> w, std::uint32_t c, detail::Stack& stack) {
        using namespace detail;
        const std::size_t lane_length = LaneLayout::length(std::uint32_t(count_)), stride = std::max<std::size_t>(8, lane_length);
        const bool transform = count_ > LaneLayout::kBase;
        std::uint32_t* const lanes = stack.take(8 * stride);
        const std::size_t mark = stack.mark();
        std::uint32_t* const columns = stack.take(8 * stride);
        if (!transform) std::fill_n(columns, 8 * stride, 0);
        const std::size_t length = StandardLayout::length(std::uint32_t(m8_));
        std::uint32_t* const root = stack.take(length);
        t_.forward(w, length - m8_, {root, length});
        std::uint32_t u = 1;
        top(0, 8, root, w.data(), c, stack, columns, lane_length, stride, transform ? &u : nullptr);
        if (transform) {
            t_.standard_to_lanes(columns, stride, lane_length, lanes);
        } else {
            for (std::size_t j = 0; j < stride; j += 8) {
                Vec r[8];
                for (std::size_t l = 0; l < 8; ++l) r[l] = load(columns + l * stride + j);
                transpose8(r);
                for (std::size_t i = 0; i < 8; ++i) store(lanes + 8 * (j + i), r[i]);
            }
        }
        stack.release(mark);
        return {lanes, u};
    }

    // Node [lo, hi) of the top tree with state s (overwritten) and u = 1 / f_v, as
    // PointTree::descend_top: the leaves' windows (right-aligned in lane_length) into columns, or
    // with leaf_u their transforms of length lane_length times 2^3 / c, 1 / that into leaf_u. window:
    // the coefficients of s's window if known (overwritten), else null; the right children's
    // windows are prefixes of it, so their states need one forward each.
    void top(std::size_t lo, std::size_t hi, std::uint32_t* s, std::uint32_t* window, std::uint32_t u, detail::Stack& stack,
             std::uint32_t* columns, std::size_t lane_length, std::size_t stride, std::uint32_t* leaf_u) {
        using namespace detail;
        const TopTree& tree = *top_;
        const std::size_t length = StandardLayout::length(tree.degree(lo, hi)), mid = tree.split(lo, hi);
        {
            const std::size_t inner = StandardLayout::length(tree.degree(lo, mid)), mark = stack.mark();
            std::uint32_t* const p = stack.take(length);
            t_.leaf_products(s, top_store_.right(mid), {p, length});
            if (mid - lo == 1 && leaf_u) {
                halve_upper(t_, p, length);
                std::copy_n(p, lane_length, columns + lo * stride);
                *leaf_u = half(u);
            } else if (mid - lo == 1) {
                to_coefficients(t_, p, length, u);
                std::copy_n(p + length - lane_length, lane_length, columns + lo * stride);
            } else if (2 * inner == length) {
                halve_upper(t_, p, length);
                top(lo, mid, p, nullptr, half(u), stack, columns, lane_length, stride, leaf_u);
            } else {
                to_coefficients(t_, p, length, u);
                std::uint32_t* const q = stack.take(inner);
                t_.forward({p + length - inner, inner}, 0, {q, inner});
                top(lo, mid, q, nullptr, 1, stack, columns, lane_length, stride, leaf_u);
            }
            stack.release(mark);
        }
        // The right child's window W_v[0, s_r) is at [L - s_v, L - s_l) in s's polynomial.
        const std::size_t sv = tree.degree(lo, hi), sl = tree.degree(lo, mid), sr = sv - sl, inner = StandardLayout::length(std::uint32_t(sr));
        const bool leaf = hi - mid == 1;
        if (leaf && !leaf_u) {
            to_coefficients(t_, s, length, u);
            std::copy_n(s + length - sv, sr, columns + mid * stride + lane_length - sr);
            return;
        }
        // The child's state carries 2 f_v, as a halving gives.
        std::uint32_t* out = leaf ? columns + mid * stride : nullptr;
        if (!window && 2 * sl == length && 2 * inner == length) {
            halve_lower(t_, s, length);
            if (leaf) std::copy_n(s, lane_length, out);
            else out = s;
        } else {
            if (window) {
                double_mod(window, sr);
            } else {
                to_coefficients(t_, s, length, 2);
                window = s + length - sv;
            }
            if (!leaf) out = stack.take(inner);
            t_.forward({window, sr}, inner - sr, {out, inner});
        }
        if (!leaf) return top(mid, hi, out, window, half(u), stack, columns, lane_length, stride, leaf_u);
        *leaf_u = half(u);
    }

    // Node [lo, hi) of the lane tree, of degree > kBase, with state s (overwritten) and u = 1 / f_v.
    void lanes(std::size_t lo, std::size_t hi, std::uint32_t* s, std::uint32_t u, detail::Stack& stack, detail::NewtonBase& base) {
        using namespace detail;
        const LaneTree& tree = *lanes_;
        const std::size_t words = 8 * LaneLayout::length(std::uint32_t(tree.degree(lo, hi))), mid = tree.split(lo, hi);
        {
            const std::size_t degree = tree.degree(lo, mid), inner = 8 * LaneLayout::length(std::uint32_t(degree)), mark = stack.mark();
            std::uint32_t* const p = stack.take(words);
            TreeTransform::pointwise_products(s, lane_store_.right(mid), {p, words});
            if (mid - lo == 1 || degree <= LaneLayout::kBase) {
                to_coefficients(t_, p, words, u);
                base.run(lo, mid, p + words - 8 * degree);
            } else if (2 * inner == words) {
                halve_upper(t_, p, words);
                lanes(lo, mid, p, half(u), stack, base);
            } else {
                to_coefficients(t_, p, words, u);
                std::uint32_t* const q = stack.take(inner);
                t_.forward({p + words - inner, inner}, 0, {q, inner});
                lanes(lo, mid, q, 1, stack, base);
            }
            stack.release(mark);
        }
        const std::size_t sv = tree.degree(lo, hi), sl = tree.degree(lo, mid), sr = sv - sl, inner = 8 * LaneLayout::length(std::uint32_t(sr));
        if (sr <= LaneLayout::kBase) {
            to_coefficients(t_, s, words, u);
            return base.run(mid, hi, s + words - 8 * sv);
        }
        if (16 * sl == words && 2 * inner == words) {
            halve_lower(t_, s, words);
            return lanes(mid, hi, s, half(u), stack, base);
        }
        to_coefficients(t_, s, words, 2);
        std::uint32_t* const q = stack.take(inner);
        t_.forward({s + words - 8 * sv, 8 * sr}, inner - 8 * sr, {q, inner});
        lanes(mid, hi, q, half(u), stack, base);
    }

    std::size_t m8_, count_;
    TreeTransform t_;
    std::span<std::uint32_t> points_, y_, stack_;  // stack_: X past the split tables
    detail::PointSlots slots_{};
    detail::LaneProducts lane_products_{};
    detail::RightStore lane_store_, top_store_;
    std::optional<LaneTree> lanes_;
    std::optional<TopTree> top_;
    std::span<const std::uint32_t> product_;
};

// Arena words to_newton() takes for n points.
inline std::size_t to_newton_words(std::size_t n) {
    const std::size_t m8 = NewtonTree::padded_size(n);
    const detail::Division sizes(n, m8);
    return NewtonTree::words(n, sizes.scratch_words()) + Transform::words(sizes.log()) + Arena::footprint(m8);
}

// c with f = sum_k c_k prod_(i < k) (x - points[i]), for n = f.size() = points.size() = c.size()
// >= 1, all canonical. arena: to_newton_words(n) words.
inline void to_newton(Arena& arena, std::span<const std::uint32_t> f, std::span<const std::uint32_t> points, std::span<std::uint32_t> c) {
    const std::size_t n = points.size(), m8 = NewtonTree::padded_size(n);
    const detail::Division sizes(n, m8);  // K = m8
    NewtonTree tree(arena, points, sizes.scratch_words());
    const Transform t(arena, sizes.log());
    const std::span<std::uint32_t> d = arena.take(m8);
    detail::divide(t, sizes, f, tree.product(), tree.product_transform(), d, tree.division_scratch());  // F / Q mod x^m8, times 2^-32
    tree.descend(d, d, detail::kR);
    tree.natural_order(d.data(), c);
}

}  // namespace poly
