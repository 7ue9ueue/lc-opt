// Products of many polynomials modulo 998244353 by product trees, with transform doubling.
// x86-64 with AVX2. Design: lib/poly/notes.md (Product tree).
//
//   poly::Arena arena(poly::TreeTransform::words(lg) + ...);
//   const poly::TreeTransform t(arena, lg);  // transforms of 2^3 .. 2^lg words
//   poly::ProductTree<poly::LaneLayout, Slots> lanes(t, slots, arena.take(words));
//   const auto root = lanes.root();          // 8 products, one per lane
//   poly::ProductTree<poly::StandardLayout, Items> top(t, items, arena.take(words));
//   const auto product = top.root();         // one product
//
// Coefficients are in Montgomery form (x 2^32 mod P) throughout: products of transforms then
// need no correction (leaf products and pointwise products both carry 2^-32).
//
// Layouts. Standard: a polynomial is its coefficients, the transforms are Transform's (n / 8
// leaves mod x^8 - w_p). Lanes: 8 polynomials side by side, word 8 j + l = coefficient j of
// polynomial l; the same transforms of 8 m words then act on each lane as a transform of length
// m with leaves of one coefficient (X = x^8 in Transform's terms is x here), so products are
// pointwise.
//
// Tree: leaves 0 .. count - 1, each one polynomial (a slot of 8 in lanes) of degree >= 1. A node
// covers leaves [lo, hi) and is split where the prefix sums of degree(k) come closest to halves.
// Its length L is a power of two >= its degree (>= 8 in the standard layout): its product p comes
// from the children's transforms of length L as p mod (x^L - 1); if deg p = L, p[L] is the
// product of the leading coefficients and p[0] is corrected by it. The transform of length L' > L
// that the parent needs is that of length L (kept) and transforms of p mod (x^m + 1) for
// m = L .. L' / 2 (forward_upper), from the coefficients of p. Per node of length L: one product
// pass, one inverse, one forward of length L. In lanes, nodes of degree <= 32 are multiplied out
// by schoolbook (64-bit sums, one Montgomery step per coefficient), then transformed.
//
// Nodes are computed depth first; scratch is a stack (ProductTree::scratch_words).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

inline Vec add_mod(Vec x, Vec y) { return reduce(add(x, y), kP); }                                      // x, y < P
inline Vec subtract_mod(Vec x, Vec y) { return reduce(_mm256_sub_epi32(add(x, broadcast(kP)), y), kP); }  // x, y < P

// x 2^32 mod P in [0, P), for x < 2^32.
inline Vec to_montgomery(Vec x) { return reduce(times(x, Factor(kR)), kP); }

// Block k of m vectors is a polynomial mod (X^m - r[k]^2), split into blocks 2k, 2k + 1 by
// X^(m/2) = +-r[k]: Transform's radix-2 steps, for transforms of 1, 2, 4 vectors (8 .. 32 words).
inline void forward_small(Vec* f, std::size_t m, std::size_t k, const std::uint32_t* roots) {
    if (m == 1) return;
    const std::size_t h = m / 2;
    const Factor s = entry(roots, k);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec lo = f[j], hi = reduce(times(f[j + h], s), kP);
        f[j] = add_mod(lo, hi), f[j + h] = subtract_mod(lo, hi);
    }
    forward_small(f, h, 2 * k, roots);
    forward_small(f + h, h, 2 * k + 1, roots);
}

// Inverse steps; the output is m times the input polynomial, canonical.
inline void inverse_small(Vec* f, std::size_t m, std::size_t k, const std::uint32_t* inverse_roots) {
    if (m == 1) return;
    const std::size_t h = m / 2;
    inverse_small(f, h, 2 * k, inverse_roots);
    inverse_small(f + h, h, 2 * k + 1, inverse_roots);
    const Factor s = entry(inverse_roots, k);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec a = f[j], b = f[j + h];
        f[j] = add_mod(a, b), f[j + h] = reduce(times(_mm256_sub_epi32(add(a, broadcast(kP)), b), s), kP);
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

// Montgomery product of scalars: x y / 2^32 mod P.
inline std::uint32_t montgomery_scalar(std::uint32_t x, std::uint32_t y) {
    constexpr std::uint32_t kInverseR = ntt::detail::power(kR, kP - 2);
    return ntt::detail::multiply_mod(ntt::detail::multiply_mod(x, y), kInverseR);
}

// Lanes: c = a b for polynomials of degrees m and k (m + k <= 32), m + 1 and k + 1 vectors in
// Montgomery form; c canonical. Coefficient sums of up to 17 products < P^2 in 64-bit lanes. Up to
// 12 products: one Montgomery step (< 3.8P). More: first 2^32 h + l -> h 2^32 + l mod P (< 2^62),
// then the step (< 2P + 1).
[[gnu::always_inline]] inline void multiply_lanes(const std::uint32_t* a, std::size_t m, const std::uint32_t* b, std::size_t k,
                                                  std::uint32_t* c) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP), r = broadcast(kR);
    Vec x[33], xo[33], y[33], yo[33];  // odd lanes moved to the even ones
#pragma GCC unroll 33
    for (std::size_t i = 0; i <= m; ++i) x[i] = load(a + 8 * i), xo[i] = _mm256_srli_epi64(x[i], 32);
#pragma GCC unroll 33
    for (std::size_t j = 0; j <= k; ++j) y[j] = load(b + 8 * j), yo[j] = _mm256_srli_epi64(y[j], 32);
    const auto fold = [r](Vec s) { return _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(s, 32), r), _mm256_blend_epi32(s, _mm256_setzero_si256(), 0xAA)); };
#pragma GCC unroll 65
    for (std::size_t t = 0; t <= m + k; ++t) {
        Vec even = _mm256_setzero_si256(), odd = _mm256_setzero_si256();
        const std::size_t first = t > k ? t - k : 0, last = std::min(t, m);
#pragma GCC unroll 17
        for (std::size_t i = first; i <= last; ++i) {
            even = _mm256_add_epi64(even, _mm256_mul_epu32(x[i], y[t - i]));
            odd = _mm256_add_epi64(odd, _mm256_mul_epu32(xo[i], yo[t - i]));
        }
        if (last - first >= 12) even = fold(even), odd = fold(odd);
        even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
        odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
        store(c + 8 * t, canonical(_mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA)));
    }
}

// multiply_lanes for m = k = 1, 2, 4, 8, 16 unrolled; others by the loops.
inline void multiply_lanes_any(const std::uint32_t* a, std::size_t m, const std::uint32_t* b, std::size_t k, std::uint32_t* c) {
    if (m == k) {
        switch (m) {
            case 1: return multiply_lanes(a, 1, b, 1, c);
            case 2: return multiply_lanes(a, 2, b, 2, c);
            case 4: return multiply_lanes(a, 4, b, 4, c);
            case 8: return multiply_lanes(a, 8, b, 8, c);
            case 16: return multiply_lanes(a, 16, b, 16, c);
        }
    }
    multiply_lanes(a, m, b, k, c);
}

// LIFO allocation from one span: spans as Arena's (32-byte aligned, 16 words after each).
class Stack {
public:
    explicit Stack(std::span<std::uint32_t> memory) : base_(memory.data()), size_(memory.size()) {}

    std::size_t mark() const { return top_; }
    void release(std::size_t mark) { top_ = mark; }

    std::uint32_t* take(std::size_t n) {
        const std::size_t need = Arena::footprint(n);
        if (top_ + need > size_) std::abort();
        std::uint32_t* const p = base_ + top_;
        top_ += need;
        return p;
    }

private:
    std::uint32_t* base_;
    std::size_t size_, top_ = 0;
};

}  // namespace detail

// Twiddle tables and transforms of 2^3 .. 2^lg_max words on the caller's spans, as Transform's
// (same leaves w_p), with cached scales; 8, 16 and 32 words by radix-2 steps.
class TreeTransform {
public:
    static constexpr std::size_t words(int lg_max) {
        return Arena::footprint(2 * ntt::detail::table_words(std::max(lg_max, Transform::kMinLog)));
    }

    TreeTransform(Arena& arena, int lg_max) : lg_max_(lg_max) {
        if (lg_max < 3 || lg_max > Transform::kMaxLog) std::abort();
        const int lg = std::max(lg_max, Transform::kMinLog);
        const std::size_t table = ntt::detail::table_words(lg), entries = (std::size_t(1) << lg) / 16;
        roots_ = arena.take(2 * table).data();
        inverse_roots_ = roots_ + table;
        ntt::detail::build_table(roots_, entries, ntt::detail::kRoots[0]);
        ntt::detail::build_table(inverse_roots_, entries, ntt::detail::kRoots[1]);
        for (int k = 3; k <= lg_max; ++k) leaf_scale_[k] = ntt::detail::power(std::uint32_t(1) << (k - 3), detail::kP - 2);
    }

    int lg_max() const { return lg_max_; }

    // out = the transform of x^shift in, n = out.size() words, as Transform::forward.
    void forward(std::span<const std::uint32_t> in, std::size_t shift, std::span<std::uint32_t> out) const {
        using namespace detail;
        const Source src = source(in, shift, out.size());
        if (out.size() < 64) return small(src, out, 0);
        run(out, src, ForwardBottom{roots_}, 0);
    }

    // out = the upper half of the transform of 2n words of x^shift in, n = out.size(), as
    // Transform::forward_upper.
    void forward_upper(std::span<const std::uint32_t> in, std::size_t shift, std::span<std::uint32_t> out) const {
        using namespace detail;
        check_length(2 * out.size());
        const Source src = source(in, shift, out.size());
        if (out.size() < 64) return small(src, out, 1);
        const Recursion recursion(roots_, inverse_roots_, ForwardBottom{roots_});
        const std::size_t nv = out.size() / 8;
        auto* v = reinterpret_cast<Vec*>(out.data());
        if (std::countr_zero(nv) % 2 == 0) {  // group 1 below a radix-2 level
            for (std::size_t j = 0; j < nv; ++j) v[j] = src(j);
            return recursion.visit(out.data(), nv, 1);
        }
        const std::size_t h = nv / 2;  // groups 2 and 3 below the radix-4 identity group
        const Factor z(roots_[1], roots_[9]);
        for (std::size_t j = 0; j < h; ++j) {
            const Vec a = src(j), zb = times(src(j + h), z);
            v[j] = add(a, zb), v[j + h] = diff(a, zb);
        }
        recursion.visit(out.data(), h, 2);
        recursion.visit(out.data() + 8 * h, h, 3);
    }

    // out = c times the coefficients of the transform in (n = out.size() words, c < P). in may
    // be out; otherwise they must not overlap.
    void inverse(std::span<const std::uint32_t> in, std::span<std::uint32_t> out, std::uint32_t c = 1) const {
        using namespace detail;
        check_length(out.size());
        if (out.size() < 64) {
            const std::size_t nv = out.size() / 8;
            Vec f[4];
            for (std::size_t j = 0; j < nv; ++j) f[j] = load(in.data() + 8 * j);
            inverse_small(f, nv, 0, inverse_roots_);
            const Factor s(scale(out.size(), c));
            for (std::size_t j = 0; j < nv; ++j) store(out.data() + 8 * j, reduce(times(f[j], s), kP));
            return;
        }
        run(out, Source(nullptr, 0, 0), InverseBottom{inverse_roots_, in.data()}, scale(out.size(), c));
    }

    // The inverse of forward_upper: out = c times the coefficients of a mod (x^n + 1) (canonical,
    // c < P) from in, the upper half of a's transform of length 2n (n = out.size() words). in may
    // be out; otherwise they must not overlap.
    void inverse_upper(std::span<const std::uint32_t> in, std::span<std::uint32_t> out, std::uint32_t c = 1) const {
        using namespace detail;
        check_length(2 * out.size());
        const std::size_t nv = out.size() / 8;
        auto* v = reinterpret_cast<Vec*>(out.data());
        const Factor s(scale(out.size(), c));
        if (out.size() < 64) {
            Vec f[4];
            for (std::size_t j = 0; j < nv; ++j) f[j] = load(in.data() + 8 * j);
            inverse_small(f, nv, 1, inverse_roots_);
            for (std::size_t j = 0; j < nv; ++j) v[j] = reduce(times(f[j], s), kP);
            return;
        }
        // The bottom reads leaves by their index in the transform of length 2n: in is its upper half.
        const Recursion recursion(roots_, inverse_roots_, InverseBottom{inverse_roots_, in.data() - out.size()});
        if (std::countr_zero(nv) % 2 == 0) {  // group 1: groups 4 .. 7 below its radix-4 butterfly
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) recursion.visit(out.data() + 8 * t * h, h, 4 + t);
            const Group w(inverse_roots_, 1);
            for (std::size_t j = 0; j < h; ++j) {
                Vec f[4] = {v[j], v[j + h], v[j + 2 * h], v[j + 3 * h]};
                inverse_h1(f, w);
                for (std::size_t t = 0; t < 4; ++t) v[j + t * h] = reduce(times(f[t], s), kP);
            }
            return;
        }
        // Groups 2 and 3 (moduli x^(n/2) -+ z, z^2 = -1), then u + w and (u - w) / z.
        const std::size_t h = nv / 2;
        recursion.visit(out.data(), h, 2);
        recursion.visit(out.data() + 8 * h, h, 3);
        const Factor sz(ntt::detail::multiply_mod(scale(out.size(), c), inverse_roots_[1]));
        for (std::size_t j = 0; j < h; ++j) {
            const Vec u = v[j], w = v[j + h];
            v[j] = reduce(times(add(u, w), s), kP), v[j + h] = reduce(times(diff(u, w), sz), kP);
        }
    }

    // Standard layout: out = a b leaf by leaf (canonical, times 2^-32), n = out.size() words.
    // out may be a or b.
    void leaf_products(const std::uint32_t* a, const std::uint32_t* b, std::span<std::uint32_t> out) const {
        using namespace detail;
        const std::size_t leaves = out.size() / 8;
        Window window[2];
        fill_window(window[0], load(a), leaf_weight(roots_, 0));
        for (std::size_t p = 0; p < leaves; ++p) {
            if (p + 1 < leaves) fill_window(window[(p + 1) & 1], load(a + 8 * p + 8), leaf_weight(roots_, p + 1));
            store(out.data() + 8 * p, reduce(leaf_product(window[p & 1], b + 8 * p), kP));
        }
    }

    // out (8 n words) = the lanes transform of the 8 polynomials whose standard transforms (n words
    // each, 8 <= n <= 2^(lg_max - 3)) are at in + l stride. Leaf p of a standard transform is the
    // polynomial mod (x^8 - w_p), whose roots are w_(8p) .. w_(8p+7); word 8 q + l of a lanes
    // transform is polynomial l at w_q. Per 8 vectors: an 8 x 8 transpose, the radix-2 step from
    // x^8 - w_p to x^4 - w_(2p), x^4 - w_(2p+1) (w_(2p) = r[p] = -w_(2p+1)), then the radix-4
    // groups 2p and 2p + 1 at h = 1, as the transforms' bottoms.
    void standard_to_lanes(const std::uint32_t* in, std::size_t stride, std::size_t n, std::uint32_t* out) const {
        using namespace detail;
        for (std::size_t p = 0; p < n / 8; ++p) {
            Vec f[8];
            for (std::size_t l = 0; l < 8; ++l) f[l] = load(in + l * stride + 8 * p);
            transpose8(f);
            const Factor z = entry(roots_, p);
            Vec lo[4], hi[4];
            for (std::size_t t = 0; t < 4; ++t) {
                const Vec zb = times(f[t + 4], z);
                lo[t] = add(f[t], zb), hi[t] = diff(f[t], zb);  // < 3P
            }
            forward_h1(lo, Group(roots_, 2 * p));  // < 4P
            forward_h1(hi, Group(roots_, 2 * p + 1));
            for (std::size_t t = 0; t < 4; ++t) store(out + 64 * p + 8 * t, canonical(lo[t])), store(out + 64 * p + 32 + 8 * t, canonical(hi[t]));
        }
    }

    // Standard layout: out = a b + c d leaf by leaf (canonical, times 2^-32), n = out.size()
    // words. out may be an operand.
    void leaf_product_sums(const std::uint32_t* a, const std::uint32_t* b, const std::uint32_t* c, const std::uint32_t* d,
                           std::span<std::uint32_t> out) const {
        using namespace detail;
        const std::size_t leaves = out.size() / 8;
        Window window[2][2];
        fill_window(window[0][0], load(a), leaf_weight(roots_, 0));
        fill_window(window[0][1], load(c), leaf_weight(roots_, 0));
        for (std::size_t p = 0; p < leaves; ++p) {
            if (p + 1 < leaves) {
                const Factor w = leaf_weight(roots_, p + 1);
                fill_window(window[(p + 1) & 1][0], load(a + 8 * p + 8), w);
                fill_window(window[(p + 1) & 1][1], load(c + 8 * p + 8), w);
            }
            const Vec ab = leaf_product(window[p & 1][0], b + 8 * p), cd = leaf_product(window[p & 1][1], d + 8 * p);  // < 2P each
            store(out.data() + 8 * p, canonical(add(ab, cd)));
        }
    }

    // Lanes: out = a b / 2^32 word by word (canonical). out may be a or b.
    static void pointwise_products(const std::uint32_t* a, const std::uint32_t* b, std::span<std::uint32_t> out) {
        using namespace detail;
        for (std::size_t i = 0; i < out.size(); i += 8) store(out.data() + i, reduce(montgomery(load(a + i), load(b + i)), kP));
    }

    // Lanes: out = (a b + c d) / 2^32 word by word (canonical), operands canonical: a b + c d +
    // (Montgomery term) < 2 P^2 + 2^32 P, so the quotient is below 1.47 P. out may be an operand.
    static void pointwise_product_sums(const std::uint32_t* a, const std::uint32_t* b, const std::uint32_t* c, const std::uint32_t* d,
                                       std::span<std::uint32_t> out) {
        using namespace detail;
        const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
        for (std::size_t i = 0; i < out.size(); i += 8) {
            const Vec x = load(a + i), y = load(b + i), z = load(c + i), w = load(d + i);
            Vec even = _mm256_add_epi64(_mm256_mul_epu32(x, y), _mm256_mul_epu32(z, w));
            Vec odd = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32)),
                                       _mm256_mul_epu32(_mm256_srli_epi64(z, 32), _mm256_srli_epi64(w, 32)));
            even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
            odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
            store(out.data() + i, reduce(_mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA), kP));
        }
    }

private:
    // The inverse's scale for n words, times c.
    std::uint32_t scale(std::size_t n, std::uint32_t c) const { return ntt::detail::multiply_mod(leaf_scale_[std::countr_zero(n)], c); }

    void check_length(std::size_t n) const {
        if (!std::has_single_bit(n) || n < 8 || std::countr_zero(n) > lg_max_) std::abort();
    }

    detail::Source source(std::span<const std::uint32_t> in, std::size_t shift, std::size_t n) const {
        check_length(n);
        if (shift > n || in.size() > n - shift) std::abort();
        return detail::Source(in.data(), in.size(), shift);
    }

    // Transforms of 1, 2 or 4 vectors: block (block = 0) or its upper neighbour (block = 1).
    void small(const detail::Source& src, std::span<std::uint32_t> out, std::size_t block) const {
        using namespace detail;
        const std::size_t nv = out.size() / 8;
        Vec f[4];
        for (std::size_t j = 0; j < nv; ++j) f[j] = src(j);
        forward_small(f, nv, block, roots_);
        for (std::size_t j = 0; j < nv; ++j) store(out.data() + 8 * j, f[j]);
    }

    // Top level, subtrees, the inverse top level with the scale (as Transform's run).
    template <class Bottom>
    void run(std::span<std::uint32_t> a, const detail::Source& in, const Bottom& bottom, std::uint32_t scale) const {
        using namespace detail;
        const Recursion recursion(roots_, inverse_roots_, bottom);
        const std::size_t nv = a.size() / 8;
        auto* v = reinterpret_cast<Vec*>(a.data());
        if (std::countr_zero(nv) % 2 == 0) {
            const std::size_t h = nv / 4;
            if constexpr (Bottom::kForward) forward_top4(in, v, h, roots_);
            for (std::size_t t = 0; t < 4; ++t) recursion.visit(a.data() + 8 * t * h, h, t);
            if constexpr (Bottom::kInverse) inverse_top4(v, h, Half::kBoth, inverse_roots_, Factor(scale));
        } else {
            const std::size_t h = nv / 2;
            if constexpr (Bottom::kForward) forward_top2(in, v, h);
            recursion.visit(a.data(), h, 0);
            recursion.visit(a.data() + 8 * h, h, 1);
            if constexpr (Bottom::kInverse) inverse_top2(v, h, Half::kBoth, scale);
        }
    }

    int lg_max_;
    std::uint32_t *roots_, *inverse_roots_;
    std::uint32_t leaf_scale_[Transform::kMaxLog + 1] = {};  // (2^k / 8)^-1: the inverse's scale for 2^k words
};

// Lanes: 8 polynomials per node, degrees and leading coefficients as vectors (one per lane).
struct LaneLayout {
    static constexpr std::size_t kWords = 8;  // words per coefficient
    static constexpr std::uint32_t kBase = 32;  // nodes up to this degree by schoolbook
    using Value = detail::Vec;
    struct Node {
        Value degree, lead;
    };

    static void multiply(const std::uint32_t* a, std::size_t m, const std::uint32_t* b, std::size_t k, std::uint32_t* c) {
        detail::multiply_lanes_any(a, m, b, k, c);
    }

    static std::size_t length(std::uint32_t degree) { return std::bit_ceil(std::max<std::uint32_t>(degree, 1)); }

    static void product(const TreeTransform&, const std::uint32_t* a, const std::uint32_t* b, std::span<std::uint32_t> out) {
        TreeTransform::pointwise_products(a, b, out);
    }

    // The leading coefficients: lane l of c[degree_l].
    static Value lead(const std::uint32_t* c, Value degree) {
        const Value index = _mm256_add_epi32(_mm256_slli_epi32(degree, 3), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        return _mm256_i32gather_epi32(reinterpret_cast<const int*>(c), index, 4);
    }

    static Value sum(Value x, Value y) { return _mm256_add_epi32(x, y); }
    static Value lead_product(Value x, Value y) { return detail::reduce(detail::montgomery(x, y), detail::kP); }

    // The leading coefficient where the degree is length, else 0.
    static Value top(Value degree, Value lead, std::size_t length) {
        return _mm256_and_si256(_mm256_cmpeq_epi32(degree, detail::broadcast(std::uint32_t(length))), lead);
    }

    static void subtract(std::uint32_t* c, Value x) { detail::store(c, detail::subtract_mod(detail::load(c), x)); }
    static void add(std::uint32_t* c, Value x) { detail::store(c, detail::add_mod(detail::load(c), x)); }
    static void put(std::uint32_t* c, Value x) { detail::store(c, x); }
};

// Standard: one polynomial per node.
struct StandardLayout {
    static constexpr std::size_t kWords = 1;
    static constexpr std::uint32_t kBase = 0;
    using Value = std::uint32_t;
    struct Node {
        Value degree, lead;
    };

    static void multiply(const std::uint32_t*, std::size_t, const std::uint32_t*, std::size_t, std::uint32_t*) {}

    static std::size_t length(std::uint32_t degree) { return std::max<std::size_t>(8, std::bit_ceil(degree)); }

    static void product(const TreeTransform& t, const std::uint32_t* a, const std::uint32_t* b, std::span<std::uint32_t> out) {
        t.leaf_products(a, b, out);
    }

    static Value lead(const std::uint32_t* c, Value degree) { return c[degree]; }

    static Value sum(Value x, Value y) { return x + y; }
    static Value lead_product(Value x, Value y) { return detail::montgomery_scalar(x, y); }
    static Value top(Value degree, Value lead, std::size_t length) { return degree == length ? lead : 0; }
    static void subtract(std::uint32_t* c, Value x) { *c = *c >= x ? *c - x : *c + detail::kP - x; }  // c, x < P
    static void add(std::uint32_t* c, Value x) { *c = *c + x >= detail::kP ? *c + x - detail::kP : *c + x; }
    static void put(std::uint32_t* c, Value x) { *c = x; }
};

// Does nothing with the nodes' transforms.
struct DropTransforms {
    void operator()(std::size_t, std::size_t, std::span<const std::uint32_t>) const {}
};

// The product tree over the leaves of a Leaves source:
//   std::size_t count() const;                 // leaves, >= 1
//   std::uint32_t degree(std::size_t k) const;  // degree of leaf k, >= 1 (lanes: the largest)
//   Layout::Node load(std::size_t k, std::uint32_t* c) const;
//       // writes the degree(k) + 1 coefficients of leaf k at c (kWords words each, Montgomery
//       // form, zero past a lane's degree) and returns its degrees and leading coefficients.
// Keep(lo, hi, transform) sees each node but the root once its transform (at its parent's
// length, Layout::kWords * length words) is final: transforms for later reuse. A Keep with
//   std::uint32_t* place(std::size_t lo, std::size_t mid, std::size_t hi, std::size_t words);
// instead gives the room where the children of node [lo, hi), split at mid, are built: the left
// one's transform (words words) at the result, the right one's Arena::footprint(words) words
// later. Its operator() is not called. A Keep with
//   std::uint32_t* place_right(std::size_t lo, std::size_t mid, std::size_t hi, std::size_t words);
// keeps the right children only: the room for the right child of node [lo, hi); the left one is
// built in the scratch (or in the unused upper part of its parent's room), as without a Keep. Its
// operator() is not called either. A Keep with root(transform) also sees the root's product
// transform (length words), which stays in the scratch.
template <class Layout, class Leaves, class Keep = DropTransforms>
class ProductTree {
public:
    using Value = typename Layout::Value;
    using Node = typename Layout::Node;
    static constexpr bool kPlaces = requires(Keep k, std::size_t i) { k.place(i, i, i, i); };
    static constexpr bool kPlacesRight = requires(Keep k, std::size_t i) { k.place_right(i, i, i, i); };
    static constexpr bool kSeesNodes = !kPlaces && !kPlacesRight;
    struct Root {
        std::span<std::uint32_t> coefficients;  // Layout::kWords * (length + 1) words, Montgomery form
        Node node;
        std::size_t length;  // coefficients past a lane's degree are zero
    };

    // Scratch words for count leaves whose degrees sum to total (lanes: the largest lane's sum).
    static std::size_t scratch_words(std::size_t count, std::size_t total) {
        return Arena::footprint(count + 1) + 8 * Arena::footprint(Layout::kWords * (Layout::length(std::uint32_t(total)) + 1)) + 4096;
    }

    ProductTree(const TreeTransform& t, const Leaves& leaves, std::span<std::uint32_t> scratch, Keep keep = {})
        : t_(t), leaves_(leaves), keep_(keep), stack_(scratch), count_(leaves.count()), prefix_(stack_.take(count_ + 1)) {
        prefix_[0] = 0;
        for (std::size_t k = 0; k < count_; ++k) prefix_[k + 1] = prefix_[k] + leaves.degree(k);
    }

    // The product of all leaves. Its coefficients live in the scratch.
    Root root() {
        const std::size_t count = count_, length = Layout::length(degree(0, count));
        constexpr std::size_t w = Layout::kWords;
        std::uint32_t* const c = stack_.take(w * (length + 1));
        if (count == 1 || degree(0, count) <= Layout::kBase) {
            std::fill_n(c, w * (length + 1), 0);
            return {{c, w * (length + 1)}, coefficients(0, count, c), length};
        }
        std::uint32_t* const a = stack_.take(w * length);
        const std::size_t mid = split(0, count);
        std::uint32_t *left_out = a, *right_out = c;
        if constexpr (kPlaces) left_out = keep_.place(0, mid, count, w * length), right_out = left_out + Arena::footprint(w * length);
        if constexpr (kPlacesRight) right_out = keep_.place_right(0, mid, count, w * length);
        const Node left = build(0, mid, left_out, length), right = build(mid, count, right_out, length);
        Layout::product(t_, left_out, right_out, {a, w * length});
        if constexpr (requires { keep_.root(std::span<const std::uint32_t>{}); }) keep_.root(std::span<const std::uint32_t>(a, w * length));
        t_.inverse({a, w * length}, {c, w * length});
        const Node node{Layout::sum(left.degree, right.degree), Layout::lead_product(left.lead, right.lead)};
        const Value top = Layout::top(node.degree, node.lead, length);
        Layout::subtract(c, top);
        Layout::put(c + w * length, top);
        return {{c, w * (length + 1)}, node, length};
    }

    // The leaf where node [lo, hi) splits.
    std::size_t split(std::size_t lo, std::size_t hi) const {
        const std::uint32_t target = prefix_[lo] + (prefix_[hi] - prefix_[lo]) / 2;
        std::size_t mid = std::size_t(std::lower_bound(prefix_ + lo + 1, prefix_ + hi - 1, target) - prefix_);  // in [lo + 1, hi - 1]
        if (mid > lo + 1 && prefix_[mid] >= target && target - prefix_[mid - 1] < prefix_[mid] - target) --mid;
        return mid;
    }

    std::uint32_t degree(std::size_t lo, std::size_t hi) const { return prefix_[hi] - prefix_[lo]; }

private:
    // The degree(lo, hi) + 1 coefficients of node [lo, hi) at c: a leaf, or a node of degree <=
    // Layout::kBase, multiplied out in consecutive pairs, level by level (any order gives the product).
    Node coefficients(std::size_t lo, std::size_t hi, std::uint32_t* c) const {
        if (hi - lo == 1) return leaves_.load(lo, c);
        constexpr std::size_t w = Layout::kWords, kMax = std::max<std::size_t>(Layout::kBase, 1);  // leaves have degree >= 1
        struct Part {
            std::uint32_t* c;
            std::size_t degree;
        };
        Part part[kMax];
        alignas(32) std::uint32_t memory[2][w * 2 * kMax];  // a level's coefficients: degree + parts <= 2 kBase
        std::size_t count = hi - lo;
        std::uint32_t* to = memory[0];
        Value degree{};
        for (std::size_t k = 0; k < count; ++k) {
            const Value d = leaves_.load(lo + k, to).degree;
            degree = k ? Layout::sum(degree, d) : d;
            part[k] = {to, leaves_.degree(lo + k)};
            to += w * (part[k].degree + 1);
        }
        for (std::size_t level = 1; count > 1; ++level) {
            to = count == 2 ? c : memory[level & 1];
            std::size_t next = 0;
            for (std::size_t k = 0; k + 1 < count; k += 2) {
                const Part &a = part[k], &b = part[k + 1];
                Layout::multiply(a.c, a.degree, b.c, b.degree, to);
                part[next++] = {to, a.degree + b.degree};
                to += w * (a.degree + b.degree + 1);
            }
            if (count % 2) {  // the odd one moves up as it is
                const Part& a = part[count - 1];
                std::copy_n(a.c, w * (a.degree + 1), to);
                part[next++] = {to, a.degree};
            }
            count = next;
        }
        return {degree, Layout::lead(c, degree)};
    }

    // out = the transform of node [lo, hi) of out_length coefficients (>= its length).
    Node build(std::size_t lo, std::size_t hi, std::uint32_t* out, std::size_t out_length) {
        constexpr std::size_t w = Layout::kWords;
        if (hi - lo == 1 || degree(lo, hi) <= Layout::kBase) {  // out_length > degree: not the root
            const Node node = coefficients(lo, hi, out);
            t_.forward({out, w * (degree(lo, hi) + 1)}, 0, {out, w * out_length});
            if constexpr (kSeesNodes) keep_(lo, hi, std::span<const std::uint32_t>(out, w * out_length));
            return node;
        }
        const std::size_t length = Layout::length(degree(lo, hi)), words = w * length, mark = stack_.mark();
        std::uint32_t *a, *b, *c;  // the children's transforms; c: p's coefficients
        std::size_t mid;
        if constexpr (kPlaces) {
            mid = split(lo, hi);
            a = keep_.place(lo, mid, hi, words), b = a + Arena::footprint(words), c = stack_.take(words + w);
        } else if constexpr (kPlacesRight) {
            a = out_length >= 2 * length ? out + words : stack_.take(words);
            mid = split(lo, hi);
            b = keep_.place_right(lo, mid, hi, words), c = stack_.take(words + w);
        } else {
            a = out_length >= 2 * length ? out + words : stack_.take(words);
            b = c = stack_.take(words + w);
            mid = split(lo, hi);
        }
        const Node left = build(lo, mid, a, length), right = build(mid, hi, b, length);
        const Node node{Layout::sum(left.degree, right.degree), Layout::lead_product(left.lead, right.lead)};
        Layout::product(t_, a, b, {out, words});  // separate passes: 1 cycle per vector faster than fused
        if (out_length > length) {
            // c = p mod (x^L - 1) = p + top (1 - x^L), top = p[L] or 0; first p mod (x^L + 1).
            t_.inverse({out, words}, {c, words});
            const Value top = Layout::top(node.degree, node.lead, length);
            Layout::subtract(c, top);
            Layout::subtract(c, top);
            t_.forward_upper({c, words}, 0, {out + words, words});
            Layout::add(c, top);
            Layout::put(c + words, top);
            for (std::size_t m = 2 * length; m < out_length; m *= 2) t_.forward_upper({c, words + w}, 0, {out + w * m, w * m});
        }
        stack_.release(mark);
        if constexpr (kSeesNodes) keep_(lo, hi, std::span<const std::uint32_t>(out, w * out_length));
        return node;
    }

    const TreeTransform& t_;
    const Leaves& leaves_;
    Keep keep_;
    detail::Stack stack_;
    std::size_t count_;
    std::uint32_t* prefix_;  // prefix_[k] = degree(0) + ... + degree(k - 1)
};

// Lane l of the vectors c[0 .. size[l]) into out[l][0 .. size[l]) for each lane, size[l] <= count
// (c holds count vectors). out[l] has room for size[l] rounded up to a multiple of 8.
inline void lane_columns(const std::uint32_t* c, std::size_t count, const std::uint32_t (&size)[8],
                         std::uint32_t* const (&out)[8]) {
    using namespace detail;
    const std::size_t longest = *std::max_element(size, size + 8);
    for (std::size_t j = 0; j < longest; j += 8) {
        Vec r[8];
        for (std::size_t i = 0; i < 8; ++i) r[i] = j + i < count ? load(c + 8 * (j + i)) : _mm256_setzero_si256();
        transpose8(r);  // r[l] = lane l of the vectors j .. j + 7
        for (int l = 0; l < 8; ++l)
            if (j < size[l]) store_unaligned(out[l] + j, r[l]);
    }
}

}  // namespace poly
