// Square root of a power series modulo 998244353: g = sqrt(f) mod x^n with g[0] = c, where
// c^2 = f[0] != 0. Design: lib/poly/notes.md.
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::sqrt_log(n)) + poly::sqrt_scratch(n) + ...);
//   poly::Transform t(arena, poly::sqrt_log(n));
//   poly::sqrt(t, f, c, g, arena.take(poly::sqrt_scratch(n)));  // g.size() == n
//
// The same in two parts, for callers that place the arrays (n > 64, m = 2^sqrt_log(n) < n):
//   poly::sqrt_steps(t, f, c, a, b, ht);            // a = g mod x^m; ht: T_m(h); a, b, ht: m words
//   t.forward(a, 0, gt);                            // gt = T_m(g mod x^m); may be a (in place)
//   poly::sqrt_last_step(t, f, gt, ht, work, hi);   // hi = g[m, n); may be f[m, n) (in place)
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <type_traits>

#include "lib/poly/calculus.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// Coefficients computed directly before the Newton steps; the first step uses transforms of
// length kSqrtBase.
inline constexpr std::size_t kSqrtBase = 64;

// g = sqrt(f) mod x^n with g[0] = c, n = g.size() <= kSqrtBase: 2 c g_i = f_i - sum_(0<j<i) g_j g_(i-j).
inline void sqrt_direct(std::span<const std::uint32_t> f, std::uint32_t c, std::span<std::uint32_t> g) {
    using ntt::detail::multiply_mod;
    const std::uint32_t inverse = ntt::detail::power(multiply_mod(2, c), kP - 2);  // 1 / (2 c)
    g[0] = c;
    for (std::size_t i = 1; i < g.size(); ++i) {
        std::uint64_t sum = i < f.size() ? f[i] : 0;  // fewer than kSqrtBase terms <= P
        for (std::size_t j = 1; j < i; ++j) sum += kP - std::uint64_t(g[j]) * g[i - j] % kP;
        g[i] = multiply_mod(std::uint32_t(sum % kP), inverse);
    }
}

// f[at, at + 8), zero past f.size().
inline Vec load_padded(std::span<const std::uint32_t> f, std::size_t at) {
    if (at + 8 <= f.size()) return load_unaligned(f.data() + at);
    alignas(32) std::uint32_t x[8] = {};
    if (at < f.size()) std::copy(f.begin() + std::ptrdiff_t(at), f.end(), x);
    return load(x);
}

// store_first and load_first: the lanes below count (all for count >= 8) at p, any alignment; the
// other lanes are not stored, or load as 0.
inline Vec first_lanes(std::size_t count) {
    return _mm256_cmpgt_epi32(broadcast(std::uint32_t(std::min<std::size_t>(count, 8))),
                              _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}
inline void store_first(std::uint32_t* p, Vec x, std::size_t count) {
    if (count >= 8) return store_unaligned(p, x);
    _mm256_maskstore_epi32(reinterpret_cast<int*>(p), first_lanes(count), x);
}
inline Vec load_first(const std::uint32_t* p, std::size_t count) {
    if (count >= 8) return load_unaligned(p);
    return _mm256_maskload_epi32(reinterpret_cast<const int*>(p), first_lanes(count));
}

// a^2 mod (x^8 - w) / 2^32 in [0, 2P) for a leaf a as a window (words <= P), as
// leaf_product(window, a) in transform.hpp computes it, with 12 products instead of 16: an odd
// output 2k + 1 pairs a_i a_j (i even, j odd) with a_j a_i (both or neither wrapped), so it is
// twice the sum over the even i. The doubled odd sums stay below 8 P^2, as the even ones.
[[gnu::always_inline]] inline Vec leaf_square(const Window& window) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP), p2 = broadcast(2 * kP);
    Vec even, odd, bi, t0, t1, t2;
    asm("vpbroadcastd 32(%[w]), %[bi]\n\t"
        "vpmuludq 36(%[w]), %[bi], %[odd]\n\t"
        "vpmuludq 32(%[w]), %[bi], %[even]\n\t"
        "vpbroadcastd 36(%[w]), %[bi]\n\t"
        "vpmuludq 28(%[w]), %[bi], %[t0]\n\t"
        "vpbroadcastd 40(%[w]), %[bi]\n\t"
        "vpmuludq 28(%[w]), %[bi], %[t1]\n\t"
        "vpmuludq 24(%[w]), %[bi], %[t2]\n\t"
        "vpaddq %[t0], %[even], %[even]\n\t"
        "vpbroadcastd 44(%[w]), %[bi]\n\t"
        "vpmuludq 20(%[w]), %[bi], %[t0]\n\t"
        "vpaddq %[t1], %[odd], %[odd]\n\t"
        "vpaddq %[t2], %[even], %[even]\n\t"
        "vpbroadcastd 48(%[w]), %[bi]\n\t"
        "vpmuludq 20(%[w]), %[bi], %[t1]\n\t"
        "vpmuludq 16(%[w]), %[bi], %[t2]\n\t"
        "vpaddq %[t0], %[even], %[even]\n\t"
        "vpbroadcastd 52(%[w]), %[bi]\n\t"
        "vpmuludq 12(%[w]), %[bi], %[t0]\n\t"
        "vpaddq %[t1], %[odd], %[odd]\n\t"
        "vpaddq %[t2], %[even], %[even]\n\t"
        "vpbroadcastd 56(%[w]), %[bi]\n\t"
        "vpmuludq 12(%[w]), %[bi], %[t1]\n\t"
        "vpmuludq 8(%[w]), %[bi], %[t2]\n\t"
        "vpaddq %[t0], %[even], %[even]\n\t"
        "vpbroadcastd 60(%[w]), %[bi]\n\t"
        "vpmuludq 4(%[w]), %[bi], %[t0]\n\t"
        "vpaddq %[t1], %[odd], %[odd]\n\t"
        "vpaddq %[t2], %[even], %[even]\n\t"
        "vpaddq %[t0], %[even], %[even]\n\t"
        "vpaddq %[odd], %[odd], %[odd]\n\t"
        // Montgomery: (s + (s / -P mod 2^32) P) / 2^32 in [0, 3P), then [0, 2P).
        "vpmuludq %[ni], %[even], %[t0]\n\t"
        "vpmuludq %[ni], %[odd], %[t1]\n\t"
        "vpmuludq %[p], %[t0], %[t0]\n\t"
        "vpmuludq %[p], %[t1], %[t1]\n\t"
        "vpaddq %[t0], %[even], %[even]\n\t"
        "vpaddq %[t1], %[odd], %[odd]\n\t"
        "vpsrlq $32, %[even], %[even]\n\t"
        "vpblendd $0xaa, %[odd], %[even], %[even]\n\t"
        "vpsubd %[p2], %[even], %[t0]\n\t"
        "vpminud %[t0], %[even], %[even]"
        : [even] "=&x"(even), [odd] "=&x"(odd), [bi] "=&x"(bi), [t0] "=&x"(t0), [t1] "=&x"(t1), [t2] "=&x"(t2)
        : [w] "r"(window.word), [ni] "x"(ni), [p] "x"(p), [p2] "x"(p2),
          "m"(*reinterpret_cast<const std::uint32_t(*)[17]>(window.word)));
    return even;
}

// InverseProductBottom for a times itself (leaf_square); out may be a.
struct InverseSquareBottom {
    static constexpr bool kForward = false, kInverse = true;
    const std::uint32_t* roots;
    const std::uint32_t* inverse_roots;
    const std::uint32_t* a;

    [[gnu::always_inline]] void prepare(std::size_t g, Window (&window)[4]) const {
        const std::uint32_t* leaves = a + 32 * g;
        const Vec f[4] = {load(leaves), load(leaves + 8), load(leaves + 16), load(leaves + 24)};
        fill_windows(window, f, roots, g);
    }

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        Window window[2][4];
        prepare(first, window[0]);
        for (std::size_t j = 0; j < count; ++j, out += 32) {
            if (j + 1 < count) prepare(first + j + 1, window[(j + 1) & 1]);
            Vec f[4];
#pragma GCC unroll 4
            for (int t = 0; t < 4; ++t) f[t] = leaf_square(window[j & 1][t]);
            inverse_h1(f, Group(inverse_roots, first + j));
            for (int t = 0; t < 4; ++t) store(out + 8 * t, f[t]);
        }
    }
};

// The top level of a transform of nv vectors is radix 4 (nv = 4^k) or radix 2 (nv = 2 4^k).
inline bool radix4(std::size_t nv) { return std::countr_zero(nv) % 2 == 0; }

// The levels below the top of a transform of nv vectors at a, for a bottom: radix-4 quarters or
// radix-2 halves. The caller runs the top levels.
template <class Bottom>
void subtrees(const Recursion<Bottom>& recursion, std::uint32_t* a, std::size_t nv) {
    if (!radix4(nv)) {
        recursion.visit(a, nv / 2, 0);
        return recursion.visit(a + 4 * nv, nv / 2, 1);
    }
    for (std::size_t q = 0; q < 4; ++q) recursion.visit(a + 2 * q * nv, nv / 4, q);
}

// A product's forward levels from its source in the lower half of a's nv vectors, in place:
// canonical, or for radix 4 already as the top level if written (forward_lower4). For radix 2 the
// radix-8 top level reads the source.
template <class Bottom>
void from_lower_source(const Transform& t, const Recursion<Bottom>& recursion, std::uint32_t* a, std::size_t nv,
                       bool written = true) {
    const Source in(a, 4 * nv, 0);
    if (!radix4(nv)) return radix2_halves(recursion, in, a, nv, t.roots());
    if (!written) forward_top4(in, reinterpret_cast<Vec*>(a), nv / 4, t.roots());
    subtrees(recursion, a, nv);
}

// A product's inverse top level in place (nv vectors at v), outputs canonical times scale; only
// output's half is computed.
inline void inverse_top(const Transform& t, Vec* v, std::size_t nv, Half output, std::uint32_t scale) {
    if (radix4(nv)) return inverse_top4(v, nv / 4, output, t.inverse_roots(), Factor(scale));
    inverse_top2(v, nv / 2, output, scale);
}

// The inverse top level of the nv vectors at v (subtree outputs < 2P), times scale, in [0, 2P):
// visit(radix4, j, h, lo, hi) for each column j, radix4 a std::bool_constant. The column's vectors
// of the lower half are lo[k] at vectors j + k h of the half (k < 2 for radix 4, k = 0 for
// radix 2; h = nv / 4 or nv / 2), those of the upper half hi[k]; only output's are computed. visit
// may write the column's vectors j + t h of v.
template <Half kOutput, class Visit>
[[gnu::always_inline]] inline void inverse_columns(const Transform& t, Vec* v, std::size_t nv, std::uint32_t scale,
                                                   Visit visit) {
    constexpr bool kLower = kOutput != Half::kUpper, kUpper = kOutput != Half::kLower;
    const Factor s(scale);
    if (radix4(nv)) {
        const std::size_t h = nv / 4;
        const Factor z(t.inverse_roots()[1], t.inverse_roots()[9]);
        for (std::size_t j = 0; j < h; ++j) {
            const Vec p0 = v[j], p1 = v[j + h], p2 = v[j + 2 * h], p3 = v[j + 3 * h];
            const Vec ab = low(add(p0, p1)), cd = low(add(p2, p3));
            const Vec amb = low(diff(p0, p1)), cmd = times(diff(p2, p3), z);
            Vec lo[2] = {}, hi[2] = {};
            if constexpr (kLower) lo[0] = times(add(ab, cd), s), lo[1] = times(add(amb, cmd), s);
            if constexpr (kUpper) hi[0] = times(diff(ab, cd), s), hi[1] = times(diff(amb, cmd), s);
            visit(std::true_type{}, j, h, lo, hi);
        }
        return;
    }
    const std::size_t h = nv / 2;
    for (std::size_t j = 0; j < h; ++j) {
        Vec lo[1] = {}, hi[1] = {};
        if constexpr (kLower) lo[0] = times(add(v[j], v[j + h]), s);
        if constexpr (kUpper) hi[0] = times(diff(v[j], v[j + h]), s);
        visit(std::false_type{}, j, h, lo, hi);
    }
}

// Column j of a radix-4 forward top level (quarters of h vectors at v) for a source in the lower
// half, quarters a and b < 2P; outputs < 4P.
[[gnu::always_inline]] inline void forward_lower4(Vec* v, std::size_t j, std::size_t h, Vec a, Vec b, const Factor& z) {
    const Vec zb = times(b, z);
    v[j] = add(a, b), v[j + h] = diff(a, b), v[j + 2 * h] = add(a, zb), v[j + 3 * h] = diff(a, zb);
}

// The same for a source in the upper half, quarters c (canonical) and d < 2P.
[[gnu::always_inline]] inline void forward_upper4(Vec* v, std::size_t j, std::size_t h, Vec c, Vec d, const Factor& z) {
    const Vec mc = _mm256_sub_epi32(broadcast(kP), c), zmd = times(_mm256_sub_epi32(broadcast(2 * kP), d), z);
    v[j] = add(c, d), v[j + h] = diff(c, d), v[j + 2 * h] = add(mc, zmd), v[j + 3 * h] = diff(mc, zmd);
}

// The transform tables' entry 1, z = r[1], for forward_lower4 and forward_upper4.
inline Factor top_root(const Transform& t) { return Factor(t.roots()[1], t.roots()[9]); }

inline Vec* vectors(std::span<std::uint32_t> a) { return reinterpret_cast<Vec*>(a.data()); }

// The scale of a product's inverse of length n (kInverseScales).
inline std::uint32_t product_scale(std::size_t n) { return kInverseScales[1][std::countr_zero(n)]; }

// (s - f[at, at + 8) - f[m + at, m + at + 8)) / 2, canonical, for s in [0, 2P). From s = g^2 mod
// (x^m - 1) at at, as g^2 = f mod x^m, this is (g^2 - f)[m + at, m + at + 8) / 2.
inline Vec half_residual(std::span<const std::uint32_t> f, std::size_t m, std::size_t at, Vec s) {
    const Factor half((kP + 1) / 2);
    const Vec fs = add(load_padded(f, at), load_padded(f, m + at));
    return reduce(times(_mm256_sub_epi32(add(s, broadcast(2 * kP)), fs), half), kP);  // < 4P before the halving
}

// out[i] = (s[i] - f[first + i] - f[m + first + i]) / 2 for i < count by half_residual: s canonical,
// 32-byte aligned, readable up to count rounded up to 8; out any alignment, may be s.
inline void half_residual(std::span<const std::uint32_t> f, std::size_t m, std::size_t first, const std::uint32_t* s,
                          std::uint32_t* out, std::size_t count) {
    for (std::size_t i = 0; i < count; i += 8) store_first(out + i, half_residual(f, m, first + i, load(s + i)), count - i);
}

// One Newton step from m to 2m. g: g mod x^m; h: h = -1 / g mod x^(m/2), at least m words. b: 2m
// words (32-byte aligned); ht: T_m(h mod x^(m/2)) in its first m words, 2m words. After it, g
// holds g mod x^2m, h holds h mod x^m and ht T_2m(h mod x^m). In transforms of length m and 2m
// (the products' top levels are passes shared with their neighbours):
//   G = T_m(g mod x^m) in b[0, m); e = (g h)[m/2, m), the upper half of G H, in b[m, 2m);
//   h[m/2, m) = (x^(m/2) e h mod (x^m - 1))[m/2, m), then T_2m(h mod x^m) in ht;
//   r = (g^2 - f)[m, 2m) / 2 from g^2 mod (x^m - 1) = G G in place of G;
//   g[m, 2m) = h r mod x^m, a product of length 2m in b.
inline void sqrt_step(const Transform& t, std::span<const std::uint32_t> f, std::size_t m, std::uint32_t* g,
                      std::uint32_t* h, std::span<std::uint32_t> b, std::span<std::uint32_t> ht) {
    const std::size_t half = m / 2, nv = m / 8;
    const std::span<std::uint32_t> gt = b.first(m), w = b.subspan(m, m), work = b.first(2 * m);
    const Factor z = top_root(t);
    const ProductBottom with_h{t.roots(), t.inverse_roots(), ht.data()};  // H, then T_2m(h)
    const std::uint32_t scale = product_scale(m), scale2 = product_scale(2 * m);
    t.forward(std::span<const std::uint32_t>(g, m), 0, gt);
    Vec* v = vectors(w);
    subtrees(Recursion(t.roots(), t.inverse_roots(), InverseProductBottom{with_h, gt.data()}), w.data(), nv);
    inverse_columns<Half::kUpper>(t, v, nv, scale, [&](auto radix4, std::size_t j, std::size_t q, const Vec*, const Vec* e) {
        if constexpr (radix4) forward_upper4(v, j, q, reduce(e[0], kP), e[1], z);
        else v[j] = e[0], v[j + q] = _mm256_sub_epi32(broadcast(2 * kP), e[0]);
    });
    subtrees(Recursion(t.roots(), t.inverse_roots(), with_h), w.data(), nv);
    inverse_columns<Half::kUpper>(t, v, nv, scale, [&](auto radix4, std::size_t j, std::size_t q, const Vec*, const Vec* x) {
        store_unaligned(h + half + 8 * j, reduce(x[0], kP));
        if constexpr (radix4) store_unaligned(h + half + 8 * (j + q), reduce(x[1], kP));
    });
    t.forward(std::span<const std::uint32_t>(h, m), 0, ht);
    // r in place of G: for m's radix 4 the canonical source, for radix 2 T_2m(r)'s radix-4 top level.
    v = vectors(gt);
    subtrees(Recursion(t.roots(), t.inverse_roots(), InverseSquareBottom{t.roots(), t.inverse_roots(), gt.data()}), gt.data(),
             nv);
    inverse_columns<Half::kBoth>(t, v, nv, scale, [&](auto radix4, std::size_t j, std::size_t q, const Vec* lo, const Vec* hi) {
        if constexpr (radix4) {
            v[j] = half_residual(f, m, 8 * j, lo[0]), v[j + q] = half_residual(f, m, 8 * (j + q), lo[1]);
            v[j + 2 * q] = half_residual(f, m, half + 8 * j, hi[0]);
            v[j + 3 * q] = half_residual(f, m, half + 8 * (j + q), hi[1]);
        } else {
            forward_lower4(v, j, q, half_residual(f, m, 8 * j, lo[0]), half_residual(f, m, half + 8 * j, hi[0]), z);
        }
    });
    from_lower_source(t, Recursion(t.roots(), t.inverse_roots(), with_h), work.data(), 2 * nv);
    inverse_columns<Half::kLower>(t, vectors(work), 2 * nv, scale2,
                                  [&](auto radix4, std::size_t j, std::size_t q, const Vec* d, const Vec*) {
                                      store_unaligned(g + m + 8 * j, reduce(d[0], kP));
                                      if constexpr (radix4) store_unaligned(g + m + 8 * (j + q), reduce(d[1], kP));
                                  });
}

}  // namespace detail

// Transform length sqrt uses for n coefficients: the Transform needs lg_max >= this.
inline int sqrt_log(std::size_t n) {
    return std::max(Transform::kMinLog, int(std::bit_width(std::max<std::size_t>(n, 2) - 1)) - 1);
}

// Scratch words for sqrt() of n coefficients.
inline std::size_t sqrt_scratch(std::size_t n) {
    if (n <= detail::kSqrtBase) return 0;
    return 3 * Arena::footprint(std::size_t(1) << sqrt_log(n));
}

// Newton steps from kSqrtBase coefficients: a = g mod x^m with g[0] = c (c^2 = f[0] != 0) and
// ht = T_m(h mod x^(m/2)), h = -1 / g, for m = a.size() = 2^sqrt_log(n), the last step's length
// (n > kSqrtBase). b: work. b and ht: m words, 32-byte aligned (from an Arena). Coefficients of f
// past f.size() are zero. None may overlap. Each step m -> 2m keeps h at m/2 and its transform of
// length m: 11 transforms of length m.
inline void sqrt_steps(const Transform& t, std::span<const std::uint32_t> f, std::uint32_t c, std::span<std::uint32_t> a,
                       std::span<std::uint32_t> b, std::span<std::uint32_t> ht) {
    using namespace detail;
    const std::size_t len = a.size();
    if (f.empty() || f[0] == 0 || ntt::detail::multiply_mod(c, c) != f[0] || len < kSqrtBase) std::abort();
    sqrt_direct(f, c, a.first(kSqrtBase));
    std::uint32_t* const h = a.data() + len / 2;  // h during the steps; g reaches it in the last one
    alignas(32) std::uint32_t h0[kSqrtBase / 2];
    inverse_direct(a.first(kSqrtBase / 2), h0);
    for (std::uint32_t& x : h0) x = x ? kP - x : 0;
    t.forward(h0, 0, ht.first(kSqrtBase));
    if (len > kSqrtBase) std::copy_n(h0, kSqrtBase / 2, h);
    for (std::size_t m = kSqrtBase; m < len; m *= 2) sqrt_step(t, f, m, a.data(), h, b.first(2 * m), ht.first(2 * m));
}

// The last Newton step: hi = g[m, n) for n = m + hi.size(), m < n <= 2m, from gt = T_m(g mod x^m)
// and ht = T_m(h mod x^(m/2)) of sqrt_steps (m = gt.size()). work: m words; gt, ht, work 32-byte
// aligned. hi may be f[m, n) (in place: f[m, n) is replaced); otherwise none may overlap.
// Transforms of length m only. With r = (g^2 - f)[m, n) / 2 from g^2 mod (x^m - 1) = G G and
// d = g[m, n): if n - m <= m/2, d = h r mod x^(n - m) (4 transforms in all). Else Karp and
// Markstein's split (8): d0 = h r0 mod x^(m/2), r0 = r mod x^(m/2); with g d = -r,
// d1 = h (r + g d0)[m/2, n - m) mod x^(n - 3m/2), where (g d0)[m/2, m) is the upper half of the
// cyclic product of g and d0; d = d0 + x^(m/2) d1.
inline void sqrt_last_step(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> gt,
                           std::span<const std::uint32_t> ht, std::span<std::uint32_t> work, std::span<std::uint32_t> hi) {
    using namespace detail;
    const std::size_t m = gt.size(), half = m / 2, rest = hi.size(), nv = m / 8;
    if (rest == 0 || rest > m) std::abort();
    const std::uint32_t scale = product_scale(m);
    const Factor z = top_root(t);
    const ProductBottom with_g{t.roots(), t.inverse_roots(), gt.data()}, with_h{t.roots(), t.inverse_roots(), ht.data()};
    const Recursion product_h(t.roots(), t.inverse_roots(), with_h);
    Vec* v = vectors(work);
    // A column of a lower-half source for from_lower_source: x(k, p) is the source's vector p of
    // the half from the column's vector k (p = j + k q).
    const auto source = [&](auto radix4, std::size_t j, std::size_t q, auto x) {
        if constexpr (radix4) forward_lower4(v, j, q, x(0, j), x(1, j + q), z);
        else v[j] = x(0, j);
    };
    // Each output vector of a column's half: put(out, count, p, x) writes vector p of the half to
    // out[8p, count) (masked, any alignment).
    const auto each = [](auto radix4, std::size_t j, std::size_t q, auto put) {
        put(0, j);
        if constexpr (radix4) put(1, j + q);
    };
    const auto put = [](std::uint32_t* out, std::size_t count, std::size_t p, Vec x) {
        if (8 * p < count) store_first(out + 8 * p, x, count - 8 * p);
    };
    // r = (g^2 - f)[m, n) / 2: r0 in place (the source of d0), r[m/2, n - m) to hi[m/2, n - m).
    // Separate passes: in the inverse top level's, f's streams 2^k words apart made it 2.7 times
    // slower at 2^17 (radix 4; lib/poly/notes.md).
    subtrees(Recursion(t.roots(), t.inverse_roots(), InverseSquareBottom{t.roots(), t.inverse_roots(), gt.data()}),
             work.data(), nv);
    const bool split = rest > half;
    inverse_top(t, v, nv, split ? Half::kBoth : Half::kLower, scale);
    half_residual(f, m, 0, work.data(), work.data(), std::min(rest, half));
    const std::size_t count = split ? rest - half : 0;      // coefficients of d1
    std::uint32_t* const upper = hi.data() + (rest - count);  // d1's place, hi + m/2 if split
    half_residual(f, m, half, work.data() + half, upper, count);
    from_lower_source(t, product_h, work.data(), nv, false);
    if (!split) {
        inverse_columns<Half::kLower>(t, v, nv, scale, [&](auto radix4, std::size_t j, std::size_t q, const Vec* d, const Vec*) {
            each(radix4, j, q, [&](int k, std::size_t p) { put(hi.data(), rest, p, reduce(d[k], kP)); });
        });
        return;
    }
    // d0 to hi[0, m/2) and in place, the source of g d0
    inverse_columns<Half::kLower>(t, v, nv, scale, [&](auto radix4, std::size_t j, std::size_t q, const Vec* d, const Vec*) {
        source(radix4, j, q, [&](int k, std::size_t p) {
            const Vec d0 = reduce(d[k], kP);
            store_unaligned(hi.data() + 8 * p, d0);
            return d0;
        });
    });
    // r[m/2, n - m) + (g d0)[m/2, m), canonical, the source of d1 (lanes from n - m on unused)
    from_lower_source(t, Recursion(t.roots(), t.inverse_roots(), with_g), work.data(), nv);
    inverse_columns<Half::kUpper>(t, v, nv, scale, [&](auto radix4, std::size_t j, std::size_t q, const Vec*, const Vec* x) {
        source(radix4, j, q, [&](int k, std::size_t p) {
            const std::size_t at = 8 * p;
            return canonical(add(x[k], load_first(upper + at, at < count ? count - at : 0)));
        });
    });
    from_lower_source(t, product_h, work.data(), nv);
    inverse_columns<Half::kLower>(t, v, nv, scale, [&](auto radix4, std::size_t j, std::size_t q, const Vec* d, const Vec*) {
        each(radix4, j, q, [&](int k, std::size_t p) { put(upper, count, p, reduce(d[k], kP)); });
    });
}

// g = sqrt(f) mod x^n for n = g.size() >= 1 with g[0] = c; c^2 = f[0] != 0. Coefficients of f
// past f.size() are zero. f and g must not overlap. scratch: sqrt_scratch(n) words, 32-byte
// aligned (from an Arena). t: lg_max >= sqrt_log(n).
inline void sqrt(const Transform& t, std::span<const std::uint32_t> f, std::uint32_t c, std::span<std::uint32_t> g,
                 std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = g.size();
    if (f.empty() || f[0] == 0 || ntt::detail::multiply_mod(c, c) != f[0]) std::abort();
    if (n <= kSqrtBase) return sqrt_direct(f, c, g);
    const std::size_t m = std::size_t(1) << sqrt_log(n), words = Arena::footprint(m);
    const std::span<std::uint32_t> b = scratch.first(m), ht = scratch.subspan(words, m), work = scratch.subspan(2 * words, m);
    sqrt_steps(t, f, c, g.first(m), b, ht);
    t.forward(g.first(m), 0, b);
    sqrt_last_step(t, f, b, ht, work, g.subspan(m));
}

}  // namespace poly
