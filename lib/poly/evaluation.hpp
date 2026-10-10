// Multipoint evaluation modulo 998244353 by the transposed product tree. x86-64 with AVX2.
// Design: lib/poly/notes.md (Multipoint evaluation).
//
//   poly::Arena arena(poly::evaluate_words(n, m));
//   poly::evaluate(arena, f, points, values);  // values[i] = f(points[i]), f of n coefficients
//
// PointTree: the product tree of Q = prod (1 - a_i x) over m points (padded with zeros to m8, a
// multiple of 8), every node's transform kept at its parent's length. Point 8k + l is slot k of
// lane l: one lane tree (LaneLayout) multiplies the 8 lanes side by side, a top tree
// (StandardLayout) the 8 lane products.
//
// Descent (the transpose of the sum of fractions sum_i v_i / (1 - a_i x)): for a series G and
// K >= m8, node v of degree s_v gets the window W_v = (G / Q_v)[K - s_v, K). A child c with
// sibling d gets W_c = (W_v Q_d)[s_d, s_v), a middle product that a cyclic product of length
// L >= s_v gives exactly (the stored transforms' length). The descent keeps each node's window as
// a transform of its length and halves the product's (PointTree::halve): per child one product
// of length L, one inverse_upper and one forward of length L/2. A single point's window is
// [x^(K-1)] G / (1 - a_i x). With G = x^(K - n) rev(f), that is f(a_i); the root's window needs
// 1 / Q mod x^K and one product.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/product_tree.hpp"

namespace poly {

namespace detail {

// Lanes: the words (canonical) of sums of terms products < P^2 / 2^32, the even lanes' in even,
// the odd lanes' in odd (64 bits each). Up to 12 terms: one Montgomery step (< 3.8P). Up to 17:
// first 2^32 h + l -> h (2^32 mod P) + l (< 2^62), then the step (< 2P + 1). As multiply_lanes.
[[gnu::always_inline]] inline Vec montgomery_sums(Vec even, Vec odd, std::size_t terms) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP), r = broadcast(kR);
    const auto fold = [r](Vec s) { return _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(s, 32), r), _mm256_blend_epi32(s, _mm256_setzero_si256(), 0xAA)); };
    if (terms > 12) even = fold(even), odd = fold(odd);
    even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
    odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
    return canonical(_mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA));
}

// Lanes: out[t] = sum_(j <= K) w[K + t - j] q[j] / 2^32 for t < C, the middle product of w (K + C
// vectors, canonical) and q (K + 1 vectors, canonical, Montgomery form), K <= 16; out canonical.
// Blocks of B outputs, their sums in registers; the odd lanes of w and q shifted once.
template <std::size_t K, std::size_t C, std::size_t B>
[[gnu::always_inline]] inline void middle_lanes(const std::uint32_t* w, const std::uint32_t* q, std::uint32_t* out) {
    static_assert(K <= 16);
    Vec wo[K + C], qo[K + 1];  // odd lanes moved to the even ones
#pragma GCC unroll 32
    for (std::size_t i = 0; i < K + C; ++i) wo[i] = _mm256_srli_epi64(load(w + 8 * i), 32);
#pragma GCC unroll 17
    for (std::size_t j = 0; j <= K; ++j) qo[j] = _mm256_srli_epi64(load(q + 8 * j), 32);
    for (std::size_t t0 = 0; t0 < C; t0 += B) {
        Vec even[B], odd[B];
#pragma GCC unroll 8
        for (std::size_t s = 0; s < B; ++s) even[s] = odd[s] = _mm256_setzero_si256();
#pragma GCC unroll 17
        for (std::size_t j = 0; j <= K; ++j) {
            const Vec y = load(q + 8 * j);
#pragma GCC unroll 8
            for (std::size_t s = 0; s < B && t0 + s < C; ++s) {
                even[s] = _mm256_add_epi64(even[s], _mm256_mul_epu32(load(w + 8 * (K + t0 + s - j)), y));
                odd[s] = _mm256_add_epi64(odd[s], _mm256_mul_epu32(wo[K + t0 + s - j], qo[j]));
            }
        }
#pragma GCC unroll 8
        for (std::size_t s = 0; s < B && t0 + s < C; ++s) store(out + 8 * (t0 + s), montgomery_sums(even[s], odd[s], K + 1));
    }
}

// The same for any k <= 16 and count, by loops. Not inlined: GCC's unrolled code for k = count = 16
// took 2.7 times as long (lib/poly/notes.md).
[[gnu::noinline]] inline void middle_lanes(const std::uint32_t* w, std::size_t k, const std::uint32_t* q, std::size_t count,
                                           std::uint32_t* out) {
    for (std::size_t t = 0; t < count; ++t) {
        Vec even = _mm256_setzero_si256(), odd = _mm256_setzero_si256();
        for (std::size_t j = 0; j <= k; ++j) {
            const Vec x = load(w + 8 * (k + t - j)), y = load(q + 8 * j);
            even = _mm256_add_epi64(even, _mm256_mul_epu32(x, y));
            odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32)));
        }
        store(out + 8 * t, montgomery_sums(even, odd, k + 1));
    }
}

// r[i] = word i of each of r[0 .. 8): an 8 x 8 transpose.
inline void transpose8(Vec (&r)[8]) {
    Vec t[8], u[8];
    for (int i = 0; i < 8; i += 2) t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]), t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
    for (int i = 0; i < 8; i += 4)
        for (int k = 0; k < 2; ++k)
            u[i + 2 * k] = _mm256_unpacklo_epi64(t[i + k], t[i + k + 2]), u[i + 2 * k + 1] = _mm256_unpackhi_epi64(t[i + k], t[i + k + 2]);
    for (int i = 0; i < 4; ++i)
        r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20), r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
}

// Lanes: slot k holds the factors 1 - a x of points[8k .. 8k + 8) (canonical), Montgomery form.
struct PointSlots {
    const std::uint32_t* points;
    std::size_t slots;

    std::size_t count() const { return slots; }
    std::uint32_t degree(std::size_t) const { return 1; }

    LaneLayout::Node load(std::size_t k, std::uint32_t* c) const {
        const Vec lead = to_montgomery(_mm256_sub_epi32(broadcast(kP), detail::load(points + 8 * k)));  // P - a in (0, P]
        store(c, broadcast(kR));
        store(c + 8, lead);
        return {broadcast(1), lead};
    }
};

// The 8 lane products as the top tree's leaves: lane l at c + l stride, degree + 1 coefficients.
struct LaneProducts {
    const std::uint32_t* c;
    std::size_t stride;
    std::uint32_t lane_degree;

    std::size_t count() const { return 8; }
    std::uint32_t degree(std::size_t) const { return lane_degree; }

    StandardLayout::Node load(std::size_t k, std::uint32_t* out) const {
        std::copy_n(c + k * stride, lane_degree + 1, out);
        return {lane_degree, out[lane_degree]};
    }
};

// A product tree's node transforms (its Keep): the tree builds each node's children in the room
// place() gives, one pair after another; pair(mid) finds those of the node split at mid. Also
// the root's product transform (in the tree's scratch).
class TransformStore {
public:
    struct Keeper {
        TransformStore* store;
        std::uint32_t* place(std::size_t, std::size_t mid, std::size_t, std::size_t words) const { return store->place(mid, words); }
        void root(std::span<const std::uint32_t> transform) const { store->root_ = transform; }
    };

    void reset(std::span<std::uint32_t> words, std::span<std::uint32_t> index) { words_ = words, used_ = 0, pair_ = index.data(); }

    std::uint32_t* place(std::size_t mid, std::size_t words) {
        const std::size_t need = 2 * Arena::footprint(words);
        if (used_ + need > words_.size()) std::abort();
        pair_[mid] = std::uint32_t(used_);
        used_ += need;
        return words_.data() + pair_[mid];
    }

    // The left child's transform; the right child's follows Arena::footprint(words) words later.
    const std::uint32_t* pair(std::size_t mid) const { return words_.data() + pair_[mid]; }

    std::span<const std::uint32_t> root() const { return root_; }

private:
    std::span<const std::uint32_t> root_;
    std::span<std::uint32_t> words_;
    std::size_t used_ = 0;
    std::uint32_t* pair_ = nullptr;  // by split point: count words
};

// Words of the transforms a tree keeps at node [lo, hi) and below: the same recursion as
// ProductTree::build.
template <class Layout, class Tree>
std::size_t kept_words(const Tree& tree, std::size_t lo, std::size_t hi) {
    if (hi - lo == 1 || tree.degree(lo, hi) <= Layout::kBase) return 0;
    const std::size_t mid = tree.split(lo, hi), words = Layout::kWords * Layout::length(tree.degree(lo, hi));
    return 2 * Arena::footprint(words) + kept_words<Layout>(tree, lo, mid) + kept_words<Layout>(tree, mid, hi);
}

// Lanes, one level of a perfect tree of nodes of degree d = D: the products of nodes pairs (2i,
// 2i + 1) of children (d + 1 vectors each) into out (2d + 1 vectors each).
template <std::size_t D>
void product_level(const std::uint32_t* children, std::uint32_t* out, std::size_t nodes) {
    for (std::size_t i = 0; i < nodes; ++i)
        multiply_lanes(children + 8 * (D + 1) * (2 * i), D, children + 8 * (D + 1) * (2 * i + 1), D, out + 8 * (2 * D + 1) * i);
}

// Output block sizes of middle_lanes<D, D, B> by D, fastest on lc-amd (lib/poly/notes.md); 0: the
// loops of the runtime middle_lanes.
constexpr std::size_t middle_block(std::size_t d) { return d >= 16 ? 0 : d == 8 ? 3 : d == 4 ? 4 : 2; }

// The windows of the nodes of degree D (D vectors each) from their parents' (2D vectors each):
// node i's from parent i / 2 and the product of its sibling i ^ 1 (D + 1 vectors each).
template <std::size_t D>
void middle_level(const std::uint32_t* parents, const std::uint32_t* products, std::uint32_t* out, std::size_t nodes) {
    for (std::size_t i = 0; i < nodes; ++i) {
        const std::uint32_t *w = parents + 16 * D * (i / 2), *q = products + 8 * (D + 1) * (i ^ 1);
        if constexpr (middle_block(D) == 0) middle_lanes(w, D, q, D, out + 8 * D * i);
        else middle_lanes<D, D, middle_block(D)>(w, q, out + 8 * D * i);
    }
}

// f.template operator()<D>() for d = D in 1, 2, 4, 8, 16.
template <class F>
void with_degree(std::size_t d, F&& f) {
    switch (d) {
        case 1: return f.template operator()<1>();
        case 2: return f.template operator()<2>();
        case 4: return f.template operator()<4>();
        case 8: return f.template operator()<8>();
        case 16: return f.template operator()<16>();
        default: std::abort();
    }
}

// The descent below a lane node of degree <= LaneLayout::kBase: slots [lo, hi) of degree 1 with
// the node's window w (hi - lo vectors). The products of every node but the top one come first
// (schoolbook, multiply_lanes), then the middle products down to single slots, whose windows are
// the values: values[8k, 8k + 8) for slot k. A power of two of slots runs level by level; other
// counts split nodes at the middle, recursively.
class BaseDescent {
public:
    BaseDescent(const PointSlots& slots, std::uint32_t* values) : slots_(slots), values_(values) {}

    void run(std::size_t lo, std::size_t hi, const std::uint32_t* w) {
        const std::size_t count = hi - lo;
        if (count == 1) return store(values_ + 8 * lo, load(w));
        if (std::has_single_bit(count)) return run_levels(lo, count, w);
        free_ = memory_;
        build(1, lo, hi);
        descend(1, lo, hi, w);
    }

private:
    // count = 2^k slots: products of degree 1 (the slots), 2, .. count / 2, level by level; then
    // windows of degree count / 2, .. 1 (into values), alternating between two buffers.
    void run_levels(std::size_t lo, std::size_t count, const std::uint32_t* w) {
        const std::uint32_t* products[6];
        std::uint32_t* at = memory_;
        for (std::size_t k = 0; k < count; ++k) slots_.load(lo + k, at + 16 * k);
        products[0] = at;
        at += 16 * count;
        for (std::size_t d = 1, j = 1; 2 * d < count; d *= 2, ++j) {
            with_degree(d, [&]<std::size_t D>() {
                if constexpr (D <= LaneLayout::kBase / 4) product_level<D>(products[j - 1], at, count / (2 * D));  // 2D < count <= kBase
            });
            products[j] = at;
            at += 8 * (2 * d + 1) * (count / (2 * d));
        }
        std::uint32_t* const windows[2] = {at, at + 8 * count};
        const std::uint32_t* parents = w;
        for (std::size_t d = count / 2, j = std::countr_zero(count) - 1;; d /= 2, --j) {
            std::uint32_t* const out = d == 1 ? values_ + 8 * lo : windows[j & 1];
            with_degree(d, [&]<std::size_t D>() { middle_level<D>(parents, products[j], out, count / D); });
            if (d == 1) return;
            parents = out;
        }
    }

    // The product of node v = [lo, hi) and of its descendants (heap order: children 2v, 2v + 1);
    // the top node's own product is not needed.
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
        std::uint32_t* const right = left + 8 * (mid - lo);
        free_ = right + 8 * (hi - mid);
        middle_lanes(w, hi - mid, product_[2 * v + 1], mid - lo, left);
        middle_lanes(w, mid - lo, product_[2 * v], hi - mid, right);
        descend(2 * v, lo, mid, left);
        descend(2 * v + 1, mid, hi, right);
        free_ = left;
    }

    // Products: at most 2 vectors per slot and deg + 1 per node below the top (<= 222 vectors for 32
    // slots); windows: at most 2 * 32 vectors.
    alignas(32) std::uint32_t memory_[8 * 320];
    const std::uint32_t* product_[64];
    std::uint32_t* free_ = memory_;
    const PointSlots& slots_;
    std::uint32_t* values_;
};

}  // namespace detail

// The product tree of Q = prod (1 - a_i x) over points a_i, padded with zeros to a multiple of 8,
// with every node's transform kept for descend().
class PointTree {
public:
    // Arena words a PointTree for m >= 1 points takes, with its tables.
    static std::size_t words(std::size_t m) {
        const std::size_t m8 = (m + 7) / 8 * 8, count = m8 / 8, levels = std::bit_width(count) + 1;
        const std::size_t top_length = StandardLayout::length(std::uint32_t(m8));
        const std::size_t kept_lanes = levels * (32 * count + 64 * (count / 32 + 1));  // 2 nodes of 8 L < 16 s words per inner node
        const std::size_t kept_top = 14 * Arena::footprint(top_length);  // 2 + 4 + 8 nodes, lengths <= the root's
        return TreeTransform::words(log_length(m8)) + Arena::footprint(m8) + Arena::footprint(LaneTree::scratch_words(count, count)) +
               Arena::footprint(kept_lanes) + Arena::footprint(count) +
               Arena::footprint(8 * ((count + 8) / 8 * 8)) + Arena::footprint(TopTree::scratch_words(8, m8)) +
               Arena::footprint(kept_top) + Arena::footprint(8);
    }

    // Scratch words descend() takes for m points.
    static std::size_t descend_words(std::size_t m) {
        const std::size_t m8 = (m + 7) / 8 * 8;
        const std::size_t lane_length = LaneLayout::length(std::uint32_t(m8 / 8)), top_length = StandardLayout::length(std::uint32_t(m8));
        return 2 * Arena::footprint(8 * std::max<std::size_t>(8, lane_length)) + 5 * Arena::footprint(top_length) +
               4 * Arena::footprint(8 * lane_length) + 4096;
    }

    PointTree(Arena& arena, std::span<const std::uint32_t> points)
        : m8_((points.size() + 7) / 8 * 8), count_(m8_ / 8), t_(arena, log_length(m8_)) {
        using namespace detail;
        std::uint32_t* const padded = arena.take(m8_).data();
        std::copy(points.begin(), points.end(), padded);  // the rest stays zero
        slots_ = {padded, count_};

        lanes_.emplace(t_, slots_, arena.take(LaneTree::scratch_words(count_, count_)), TransformStore::Keeper{&lane_store_});
        lane_store_.reset(arena.take(kept_words<LaneLayout>(*lanes_, 0, count_)), arena.take(count_));
        const LaneTree::Root lane_root = lanes_->root();

        const std::size_t stride = (count_ + 8) / 8 * 8;  // count + 1 coefficients, rounded up
        std::uint32_t* const columns = arena.take(8 * stride).data();
        const std::uint32_t size = std::uint32_t(count_ + 1);
        const std::uint32_t sizes[8] = {size, size, size, size, size, size, size, size};
        std::uint32_t* const out[8] = {columns, columns + stride, columns + 2 * stride, columns + 3 * stride,
                                       columns + 4 * stride, columns + 5 * stride, columns + 6 * stride, columns + 7 * stride};
        lane_columns(lane_root.coefficients.data(), lane_root.length + 1, sizes, out);
        lane_products_ = {columns, stride, std::uint32_t(count_)};

        top_.emplace(t_, lane_products_, arena.take(TopTree::scratch_words(8, m8_)), TransformStore::Keeper{&top_store_});
        top_store_.reset(arena.take(kept_words<StandardLayout>(*top_, 0, 8)), arena.take(8));
        product_ = top_->root().coefficients.first(m8_ + 1);
    }

    PointTree(const PointTree&) = delete;
    PointTree& operator=(const PointTree&) = delete;

    // Points with padding: a multiple of 8.
    std::size_t size() const { return m8_; }

    // Q = prod (1 - a_i x) over size() points: size() + 1 coefficients in Montgomery form.
    std::span<const std::uint32_t> product() const { return product_; }

    // The transform of Q mod (x^L - 1), L = StandardLayout::length(size()), Montgomery form.
    std::span<const std::uint32_t> product_transform() const { return top_store_.root(); }

    // values[i] = c [x^(m8 - 1)] w(x) prod_(j != i) (1 - a_j x) for i < m8 = size(), w (canonical) of
    // m8 coefficients: the window of the root. values (m8 words) canonical. scratch:
    // descend_words(m) words from an Arena.
    void descend(std::span<const std::uint32_t> w, std::span<std::uint32_t> values, std::uint32_t c, std::span<std::uint32_t> scratch) {
        using namespace detail;
        Stack stack(scratch);
        const std::size_t lane_length = LaneLayout::length(std::uint32_t(count_)), stride = std::max<std::size_t>(8, lane_length);
        std::uint32_t* const columns = stack.take(8 * stride);  // the lane products' windows, lane l at l stride
        std::fill_n(columns, 8 * stride, 0);
        const std::size_t length = StandardLayout::length(std::uint32_t(m8_));
        std::uint32_t* const root = stack.take(length);
        t_.forward(w, length - m8_, {root, length});
        descend_top(0, 8, root, c, stack, columns, lane_length, stride);  // c as the inverse of a factor
        std::uint32_t* const lanes = stack.take(8 * stride);  // word 8 i + l = coefficient i of lane l
        for (std::size_t j = 0; j < stride; j += 8) {
            Vec r[8];
            for (std::size_t l = 0; l < 8; ++l) r[l] = load(columns + l * stride + j);
            transpose8(r);
            for (std::size_t i = 0; i < 8; ++i) store(lanes + 8 * (j + i), r[i]);
        }
        BaseDescent base(slots_, values.data());
        if (count_ <= LaneLayout::kBase) return base.run(0, count_, lanes + 8 * (lane_length - count_));
        t_.forward({lanes, 8 * lane_length}, 0, {lanes, 8 * lane_length});
        descend_lanes(0, count_, lanes, 1, stack, base);
    }

private:
    using LaneTree = ProductTree<LaneLayout, detail::PointSlots, detail::TransformStore::Keeper>;
    using TopTree = ProductTree<StandardLayout, detail::LaneProducts, detail::TransformStore::Keeper>;

    // Largest transform: the top tree's root, or 8 lanes of the lane tree's root.
    static int log_length(std::size_t m8) {
        const std::size_t lanes = 8 * LaneLayout::length(std::uint32_t(m8 / 8)), top = StandardLayout::length(std::uint32_t(m8));
        return std::countr_zero(std::max(lanes, top));
    }

    // The descent's state at node v of length L: f_v times the transform of W'_v = x^(L - s_v) W_v
    // + (terms below x^(L - s_v)), the window right-aligned; f_v = 2^(halvings above v), and
    // u = 1 / f_v is passed down. A child x with sibling y: p = state times y's stored transform is
    // f_v times that of X = W'_v Q_y mod (x^L - 1), whose top s_x coefficients are W_x (the terms
    // below stay below: s_x + s_y <= L).

    // In place: p's first half becomes 2 f_v times the transform of X[L/2, L) of length L/2:
    // P_lo - T(X mod (x^(L/2) + 1)), the second term the forward of inverse_upper(P_hi, -1).
    void halve(std::uint32_t* p, std::size_t words) const {
        using namespace detail;
        const std::size_t h = words / 2;
        t_.inverse_upper({p + h, h}, {p + h, h}, kP - 1);
        t_.forward({p + h, h}, 0, {p + h, h});
        for (std::size_t i = 0; i < h; i += 8) store(p + i, add_mod(load(p + i), load(p + h + i)));
    }

    // In place: p becomes the coefficients of X, for u = 1 / f_v: with A = inverse(P_lo, u / 2)
    // = (X_lo + X_hi) / 2 and B = inverse_upper(P_hi, -u / 2) = -(X_lo - X_hi) / 2, X_lo = A - B
    // and X_hi = A + B. A single vector (n = 8, standard layout) is its own transform.
    void coefficients(std::uint32_t* p, std::size_t words, std::uint32_t u) const {
        using namespace detail;
        if (words == 8) return t_.inverse({p, 8}, {p, 8}, u);
        const std::size_t h = words / 2;
        const std::uint32_t c = half(u);  // nonzero
        t_.inverse({p, h}, {p, h}, c);
        t_.inverse_upper({p + h, h}, {p + h, h}, kP - c);
        for (std::size_t i = 0; i < h; i += 8) {
            const Vec a = load(p + i), b = load(p + h + i);
            store(p + i, subtract_mod(a, b)), store(p + h + i, add_mod(a, b));
        }
    }

    // Node [lo, hi) of the top tree with state s and u = 1 / f_v; the leaves' windows,
    // right-aligned in lane_length coefficients, into columns.
    void descend_top(std::size_t lo, std::size_t hi, const std::uint32_t* s, std::uint32_t u, detail::Stack& stack,
                     std::uint32_t* columns, std::size_t lane_length, std::size_t stride) {
        const TopTree& tree = *top_;
        const std::size_t length = StandardLayout::length(tree.degree(lo, hi)), mid = tree.split(lo, hi);
        const std::size_t sides[2][2] = {{lo, mid}, {mid, hi}};
        for (int side = 0; side < 2; ++side) {
            const std::size_t xlo = sides[side][0], xhi = sides[side][1], inner = StandardLayout::length(tree.degree(xlo, xhi));
            const std::size_t mark = stack.mark();
            std::uint32_t* const p = stack.take(length);
            t_.leaf_products(s, top_store_.pair(mid) + (1 - side) * Arena::footprint(length), {p, length});
            if (xhi - xlo == 1) {
                coefficients(p, length, u);
                std::copy_n(p + length - lane_length, lane_length, columns + xlo * stride);
            } else if (2 * inner == length) {
                halve(p, length);
                descend_top(xlo, xhi, p, half(u), stack, columns, lane_length, stride);
            } else {
                coefficients(p, length, u);
                std::uint32_t* const q = stack.take(inner);
                t_.forward({p + length - inner, inner}, 0, {q, inner});
                descend_top(xlo, xhi, q, 1, stack, columns, lane_length, stride);
            }
            stack.release(mark);
        }
    }

    // Node [lo, hi) of the lane tree with state s (8 lanes) and u = 1 / f_v.
    void descend_lanes(std::size_t lo, std::size_t hi, const std::uint32_t* s, std::uint32_t u, detail::Stack& stack,
                       detail::BaseDescent& base) {
        const LaneTree& tree = *lanes_;
        const std::size_t words = 8 * LaneLayout::length(std::uint32_t(tree.degree(lo, hi))), mid = tree.split(lo, hi);
        const std::size_t sides[2][2] = {{lo, mid}, {mid, hi}};
        for (int side = 0; side < 2; ++side) {
            const std::size_t xlo = sides[side][0], xhi = sides[side][1], degree = tree.degree(xlo, xhi);
            const std::size_t inner = 8 * LaneLayout::length(std::uint32_t(degree)), mark = stack.mark();
            std::uint32_t* const p = stack.take(words);
            TreeTransform::pointwise_products(s, lane_store_.pair(mid) + (1 - side) * Arena::footprint(words), {p, words});
            if (xhi - xlo == 1 || degree <= LaneLayout::kBase) {
                coefficients(p, words, u);
                base.run(xlo, xhi, p + words - 8 * degree);
            } else if (2 * inner == words) {
                halve(p, words);
                descend_lanes(xlo, xhi, p, half(u), stack, base);
            } else {
                coefficients(p, words, u);
                std::uint32_t* const q = stack.take(inner);
                t_.forward({p + words - inner, inner}, 0, {q, inner});
                descend_lanes(xlo, xhi, q, 1, stack, base);
            }
            stack.release(mark);
        }
    }

    static std::uint32_t half(std::uint32_t u) { return ntt::detail::multiply_mod(u, (detail::kP + 1) / 2); }

    std::size_t m8_, count_;
    TreeTransform t_;
    detail::PointSlots slots_{};
    detail::LaneProducts lane_products_{};
    detail::TransformStore lane_store_, top_store_;
    std::optional<LaneTree> lanes_;
    std::optional<TopTree> top_;
    std::span<const std::uint32_t> product_;
};

namespace detail {

// Below this many products n m, evaluate() uses Horner's rule.
inline constexpr std::size_t kDirectEvaluation = std::size_t(1) << 22;

// values[i] = f(points[i]) by Horner's rule, 32 points per step (4 chains of 8 lanes).
inline void evaluate_direct(std::span<const std::uint32_t> f, std::span<const std::uint32_t> points, std::span<std::uint32_t> values) {
    constexpr std::size_t kChains = 4;
    const Factor to_r(kR);
    for (std::size_t i = 0; i < points.size(); i += 8 * kChains) {
        alignas(32) std::uint32_t x[8 * kChains] = {}, v[8 * kChains];
        const std::size_t here = std::min<std::size_t>(8 * kChains, points.size() - i);
        std::copy_n(points.begin() + std::ptrdiff_t(i), here, x);
        Vec xm[kChains], acc[kChains];
        for (std::size_t c = 0; c < kChains; ++c) xm[c] = reduce(times(load(x + 8 * c), to_r), kP), acc[c] = _mm256_setzero_si256();
        for (std::size_t j = f.size(); j-- > 0;) {
            const Vec fj = broadcast(f[j]);
            for (std::size_t c = 0; c < kChains; ++c) acc[c] = reduce(add(montgomery(acc[c], xm[c]), fj), 2 * kP);  // < 2P
        }
        for (std::size_t c = 0; c < kChains; ++c) store(v + 8 * c, reduce(acc[c], kP));
        std::copy_n(v, here, values.begin() + std::ptrdiff_t(i));
    }
}

// The root's division for n coefficients and m8 points: K = max(n, m8) quotient coefficients in
// halves of k = ceil(K / 2) and K - k; products exact in a cyclic length L >= max(2k - 1, K).
struct Division {
    std::size_t big, k, length;

    Division(std::size_t n, std::size_t m8)
        : big(std::max(n, m8)), k((big + 1) / 2), length(std::max<std::size_t>(64, std::bit_ceil(std::max(2 * k - 1, big)))) {}

    int log() const { return std::max(inverse_log(k), std::countr_zero(length)); }

    // Scratch words of divide().
    std::size_t scratch_words() const {
        return Arena::footprint(big) + Arena::footprint(k) + Arena::footprint(inverse_scratch(k)) + 3 * Arena::footprint(length);
    }
};

// D = G / Q mod x^K for G = x^(K - n) rev(f) and q (m8 + 1 coefficients) = 2^32 Q, Q[0] = 1:
// D / 2^32 into d (K words), by Karp and Markstein's division: h = 1 / q mod x^k, q0 = G h mod
// x^k, r = (G - q q0)[k, K), q1 = h r mod x^(K - k), D = q0 + x^k q1. qt: the transform of q mod
// (x^L - 1) if its length is the division's L, else computed here. t: lg_max >= s.log();
// scratch: s.scratch_words() words from an Arena.
inline void divide(const Transform& t, const Division& s, std::span<const std::uint32_t> f, std::span<const std::uint32_t> q,
                   std::span<const std::uint32_t> qt, std::span<std::uint32_t> d, std::span<std::uint32_t> scratch) {
    const std::size_t big = s.big, k = s.k, rest = big - k, n = s.length;
    Stack stack(scratch);
    const std::span<std::uint32_t> g{stack.take(big), big}, h{stack.take(k), k};
    std::fill_n(g.begin(), big - f.size(), 0);
    std::reverse_copy(f.begin(), f.end(), g.begin() + std::ptrdiff_t(big - f.size()));
    inverse(t, q, h, {stack.take(inverse_scratch(k)), inverse_scratch(k)});
    const std::span<std::uint32_t> a{stack.take(n), n}, b{stack.take(n), n}, th{stack.take(n), n};
    const auto lower = [n](std::size_t size) { return size <= n / 2 ? Half::kLower : Half::kBoth; };
    t.forward(g.first(k), 0, a);
    t.forward(h, 0, th);
    t.inverse_product(a, th, a, lower(k));
    std::copy_n(a.begin(), k, d.begin());  // q0
    t.forward(d.first(k), 0, a);
    if (qt.size() != n) {
        const std::size_t terms = std::min(q.size(), n);  // q mod (x^L - 1): deg q <= K <= L
        std::copy_n(q.begin(), terms, b.begin());
        if (q.size() > n) b[0] = std::uint32_t((std::uint64_t(b[0]) + q[n]) % kP);
        t.forward(b.first(terms), 0, b);
        qt = b;
    }
    t.inverse_product(qt, a, a, k >= n / 2 ? Half::kUpper : Half::kBoth);  // (q q0)[k, K)
    for (std::size_t i = 0; i < rest; ++i) b[i] = g[k + i] >= a[k + i] ? g[k + i] - a[k + i] : g[k + i] + kP - a[k + i];
    t.cyclic_product(b.first(rest), 0, a, th, lower(rest));
    std::copy_n(a.begin(), rest, d.begin() + std::ptrdiff_t(k));  // q1
}

}  // namespace detail

namespace detail {

// Arena words evaluate_tree() takes for n coefficients and m points.
inline std::size_t evaluate_tree_words(std::size_t n, std::size_t m) {
    const std::size_t m8 = (m + 7) / 8 * 8;
    const Division sizes(n, m8);
    return PointTree::words(m) + Transform::words(sizes.log()) + Arena::footprint(sizes.big) +
           Arena::footprint(std::max(sizes.scratch_words(), PointTree::descend_words(m))) + Arena::footprint(m8);
}

// evaluate() by the tree, for any sizes: the root's window is (G / Q)[K - m8, K). The division
// and the descent share their scratch. The values go straight into a 32-byte aligned values when
// m is a multiple of 8.
inline void evaluate_tree(Arena& arena, std::span<const std::uint32_t> f, std::span<const std::uint32_t> points,
                          std::span<std::uint32_t> values) {
    const std::size_t m = points.size();
    PointTree tree(arena, points);
    const std::size_t m8 = tree.size();
    const Division sizes(f.size(), m8);
    const Transform t(arena, sizes.log());
    const std::span<std::uint32_t> d = arena.take(sizes.big);
    const std::span<std::uint32_t> scratch = arena.take(std::max(sizes.scratch_words(), PointTree::descend_words(m)));
    divide(t, sizes, f, tree.product(), tree.product_transform(), d, scratch);
    const bool direct = m8 == m && reinterpret_cast<std::uintptr_t>(values.data()) % 32 == 0;
    const std::span<std::uint32_t> all = direct ? values : arena.take(m8);
    tree.descend(d.subspan(sizes.big - m8, m8), all, kR, scratch);  // d carries 2^-32
    if (!direct) std::copy_n(all.begin(), m, values.begin());
}

}  // namespace detail

// Arena words evaluate() takes for n coefficients and m points.
inline std::size_t evaluate_words(std::size_t n, std::size_t m) {
    return n * m <= detail::kDirectEvaluation ? 0 : detail::evaluate_tree_words(n, m);
}

// values[i] = f(points[i]) for i < m = points.size() = values.size(), f of n = f.size() >= 1
// coefficients, all canonical. arena: evaluate_words(n, m) words.
inline void evaluate(Arena& arena, std::span<const std::uint32_t> f, std::span<const std::uint32_t> points,
                     std::span<std::uint32_t> values) {
    if (f.size() * points.size() <= detail::kDirectEvaluation) return detail::evaluate_direct(f, points, values);
    detail::evaluate_tree(arena, f, points, values);
}

}  // namespace poly
