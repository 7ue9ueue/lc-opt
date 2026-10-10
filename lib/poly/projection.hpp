// Power projection modulo 998244353: a[k] = [x^(n-1)] g^k for k < n and g[0] = 0, by Kinoshita
// and Li's algorithm: composition.hpp's levels, run forward together with the numerator.
// x86-64 with AVX2. Design: lib/poly/notes.md (Power projection).
//
//   const std::size_t n = ...;
//   poly::Arena arena(poly::Transform::words(poly::projection_log(n)) + poly::projection_scratch(n) + ...);
//   poly::Transform t(arena, poly::projection_log(n));
//   poly::power_projection(t, g, a, arena.take(poly::projection_scratch(n)));  // a.size() == n
//
// With m = 2^T >= n: sum_k a[k] y^k = [x^(m-1)] P_0 / Q_0 for P_0 = x^(m-n) and Q_0 = 1 - y g(x).
// Level s has L = m / 2^s coefficients in x, Q_s of degree Y = 2^s in y and P_s of degree below
// Y, and [x^(L-1)] P_s / Q_s is the same series. With A = P_s(x) Q_s(-x): P_(s+1)(x^2) x = the
// odd part of A mod x^L and Q_(s+1)(x^2) = Q_s(x) Q_s(-x) mod x^L (the denominator is even, and
// L - 1 is odd). At s = T, L = 1 and Q_T(0, y) = 1: a = P_T. Layouts and transforms as
// composition.hpp: Kronecker x = z, y = z^(2L) at length 4m; P_s fills rows 0 .. Y - 1, z below 2m.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/composition.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

// m = 2^T for n > kComposeBase outputs: at least 512, so that level T - 3's transforms (length
// m/4) have at least 128 words.
inline std::size_t projection_length(std::size_t n) { return std::bit_ceil(std::max<std::size_t>(n, 512)); }

// a[k] = [x^(n-1)] g^k for k < n = a.size() <= kComposeBase, from the powers of g.
inline void projection_direct(std::span<const std::uint32_t> g, std::span<std::uint32_t> a) {
    const std::size_t n = a.size();
    std::uint32_t power[kComposeBase] = {1}, next[kComposeBase];  // g^k mod x^n
    for (std::size_t k = 0; k < n; ++k) {
        a[k] = power[n - 1];
        for (std::size_t i = 0; i < n; ++i) {
            std::uint64_t c = 0;  // fewer than kComposeBase terms < P
            for (std::size_t j = 1; j <= i && j < g.size(); ++j) c += std::uint64_t(g[j]) * power[i - j] % kP;
            next[i] = std::uint32_t(c % kP);
        }
        std::copy_n(next, n, power);
    }
}

// [x1 x3 x5 x7 y1 y3 y5 y7].
inline Vec odd_lanes(Vec x, Vec y) {
    const Vec t = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), 0xDD));
    return _mm256_permute4x64_epi64(t, 0xD8);
}

// out[j] = op(b[2j + offset]) (kAdd: out[j] + op(...) mod P) for j < out.size(), where b[i] counts
// as 0 outside [lo, hi), hi <= b.size(). op maps canonical lanes to canonical lanes and 0 to 0.
template <bool kAdd, class Op>
void even_terms(std::span<const std::uint32_t> b, std::size_t lo, std::size_t hi, std::ptrdiff_t offset,
                std::span<std::uint32_t> out, const Op& op) {
    const auto scalar = [&op](std::uint32_t x) {
        return std::uint32_t(_mm_cvtsi128_si32(_mm256_castsi256_si128(op(_mm256_set1_epi32(int(x))))));
    };
    const auto put = [&out](std::size_t j, std::uint32_t x) {
        if constexpr (kAdd) out[j] = out[j] + x >= kP ? out[j] + x - kP : out[j] + x;
        else out[j] = x;
    };
    const auto at = [offset](std::size_t j) { return std::ptrdiff_t(2 * j) + offset; };
    const auto first = [&](std::size_t i) {  // the least j with 2j + offset >= i, at most out.size()
        const std::ptrdiff_t d = std::ptrdiff_t(i) - offset;
        return std::min(out.size(), d <= 0 ? std::size_t(0) : std::size_t(d + 1) / 2);
    };
    const std::size_t begin = first(lo), end = std::max(begin, first(hi));
    if constexpr (!kAdd) {
        std::fill(out.begin(), out.begin() + std::ptrdiff_t(begin), 0);
        std::fill(out.begin() + std::ptrdiff_t(end), out.end(), 0);
    }
    std::size_t j = begin;
    for (; j + 8 <= end && std::size_t(at(j)) + 16 <= b.size(); j += 8) {
        const std::uint32_t* from = b.data() + at(j);
        Vec x = op(even_lanes(load_unaligned(from), load_unaligned(from + 8)));
        if constexpr (kAdd) x = reduce(add(load_unaligned(out.data() + j), x), kP);
        store_unaligned(out.data() + j, x);
    }
    for (; j < end; ++j) put(j, scalar(b[std::size_t(at(j))]));
}

// out[i] = x[i] + y[i] mod P for i < out.size() (multiples of 8; aligned spans).
inline void add_into(std::span<const std::uint32_t> x, std::span<const std::uint32_t> y, std::span<std::uint32_t> out) {
    for (std::size_t i = 0; i < out.size(); i += 8) store(out.data() + i, reduce(add(load(x.data() + i), load(y.data() + i)), kP));
}

// The bottom of the forward transform of Q_s (length 4m): forward butterflies, then per pair of
// leaves 2k, 2k + 1 (moduli z^8 -+ s, s = r[k]) the leaves of V = Q_s(x) Q_s(-x) and of
// W = the odd part of P(x) Q_s(-x), from p, P's transform (canonical leaves): mod (u^4 -+ s),
// u = z^2, A and B by graeffe_leaf and product_leaf; leaf k of the transform of length 2m is
// (A + B) + u^4 (A - B) / s, which is twice the CRT of A and B. With the 2^-32 of the products,
// V and W come out times 2^-31, in [0, 2P); their inverse transforms undo it. Leaf k goes to
// v + 8k and w + 8k. Pairs go 8 at a time, one per lane, after their leaves are read, so v may be
// Q's array and w may be p: each writes below the leaves read so far.
struct ProjectionBottom {
    static constexpr bool kForward = true, kInverse = false;
    const Tables* tables;
    const std::uint32_t* p;
    std::uint32_t *v, *w;

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        ForwardBottom{tables->roots}(a, count, first);
        for (std::size_t i = 0; i < count; i += 4) pairs(a + 32 * i, 2 * (first + i));
    }

    // 8 pairs from Q's leaves at q, the first pair k (a multiple of 8). Flattened: GCC kept the
    // transposes as calls, with every vector spilled around them.
    [[gnu::flatten]] void pairs(const std::uint32_t* q, std::size_t k) const {
        const PairWeights weights(*tables, k);
        Vec va[4], vb[4], wa[4], wb[4];
        leaves<false>(q, p + 16 * k, weights.s, va, wa);
        leaves<true>(q + 8, p + 16 * k + 8, weights.s, vb, wb);
        combine_pairs(va, vb, weights.inverse, v + 8 * k);
        combine_pairs(wa, wb, weights.inverse, w + 8 * k);
    }

    // V and W mod (u^4 - t) for t = s (or -s if kNegative) of the 8 leaves at q and p, times 2^-32.
    template <bool kNegative>
    [[gnu::noinline, gnu::flatten]] static void leaves(const std::uint32_t* q, const std::uint32_t* p, const Factors& s,
                                                       Vec (&v)[4], Vec (&w)[4]) {
        Vec c[8], a[8], tc[8];
        load_leaves(q, c), load_leaves(p, a);
        scaled_leaves<kNegative>(c, s, 2, tc);
        graeffe_leaf(c, tc, v);
        product_leaf<1>(a, c, tc, w);
    }
};

// The outputs at 0 and h of inverse_h1 on the quarters of h vectors at a (x below L/2 of a block of
// 4h vectors), to out. Inputs and outputs below 2P.
inline void inverse_lower(const std::uint32_t* a, std::size_t h, const Group& w, std::uint32_t* out) {
    for (std::size_t i = 0; i < 8 * h; i += 8) {
        const Vec f0 = load(a + i), f1 = load(a + i + 8 * h), f2 = load(a + i + 16 * h), f3 = load(a + i + 24 * h);
        store(out + i, low(add(low(add(f0, f1)), low(add(f2, f3)))));
        store(out + i + 8 * h, low(add(times(diff(f0, f1), w.y), times(diff(f2, f3), w.z))));
    }
}

// forward_h1 on the quarters of h vectors at a from in (quarters 0 and 1; 2 and 3 are zero).
// Inputs below 4P, outputs below 4P.
inline void forward_lower(const std::uint32_t* in, std::size_t h, const Group& w, std::uint32_t* a) {
    for (std::size_t i = 0; i < 8 * h; i += 8) {
        const Vec u = low(load(in + i)), b = load(in + i + 8 * h), yb = times(b, w.y), zb = times(b, w.z);
        store(a + i, add(u, yb)), store(a + i + 8 * h, diff(u, yb));
        store(a + i + 16 * h, add(u, zb)), store(a + i + 24 * h, diff(u, zb));
    }
}

// From level s to level s + 1 without the y levels of the first half (lib/poly/notes.md, Power
// projection). a: 4m words; a[0, 2m) holds the leaves of level s's V (or W), the transform of
// length 2m at stride L: 2Y blocks of L words, block j the transform of V mod (u^L - w_j) by the
// levels within the block (group j at that size, in the transforms of length 2m and 4m). Level
// s + 1 (stride L, 4Y rows) needs the transform of length 4m of Q = V with x below L/2:
//  - its first half, Q mod (y^2Y - 1): per block, the inverse levels within the block (outputs
//    x < L/2 only, to a copy at a[2m, 4m)) and the forward levels within the block, with bottom;
//  - its second half, Q mod (y^2Y + 1): the inverse y levels of the copies (columns x < L/2) give
//    Q's rows mod (y^2Y - 1); for Q of a denominator, row 0 wraps row 2Y onto the true row 0 = 1, so
//    row 0 becomes 2 - row 0 (fix); then the forward of the upper half, with bottom.
// Against an inverse of length 2m and a forward of length 4m, the first half skips its y levels.
// Factors: the levels within a block multiply by L/8 and the y levels by 2Y (y_scale: (2Y)^-1),
// so the result is (L/8) c times the true transform when the leaves are c times V's.
template <class Bottom>
class Doubling {
public:
    // block = L/8 >= 4 vectors.
    Doubling(std::span<std::uint32_t> a, std::size_t block, const Tables& tables, const Bottom& bottom)
        : a_(a.data()), half_(a.size() / 2), block_(block), b_(std::size_t(std::countr_zero(block)) - 1),
          aligned_(std::countr_zero(block) % 2 == 0), tables_(tables), bottom_(bottom),
          inverse_(tables.roots, tables.inverse_roots, InverseBottom{tables.inverse_roots, a.data()}),
          forward_(tables.roots, tables.inverse_roots, bottom) {}

    // fix: 2c for a denominator whose true row 0 is c (the transform's factor), 0 for a numerator.
    void run(std::uint32_t y_scale, std::uint32_t fix) const {
        std::uint32_t* const copy = a_ + half_;
        const std::size_t nv = half_ / 8, chunk = std::max<std::size_t>(block_, 16);
        for (std::size_t c = 0; c < nv; c += chunk) {
            for (std::size_t j = c / block_; j < (c + chunk) / block_; ++j) {
                inverse_x(a_ + 8 * block_ * j, j, copy + 8 * block_ * j);
                forward_x(copy + 8 * block_ * j, j, a_ + 8 * block_ * j);
            }
            if (block_ < 16) bottom_(a_ + 8 * c, chunk / 4, c / 4);  // blocks of one or two groups of h = 1
        }
        inverse_y(copy, y_scale);
        if (fix) {
            for (std::size_t i = 0; i < 4 * block_; i += 8) store(copy + i, negate(load(copy + i)));
            copy[0] = fix >= kP - copy[0] ? fix - (kP - copy[0]) : fix + copy[0];
        }
        forward_upper();
    }

private:
    // Block j at a: its inverse levels, the outputs x < L/2 to out (below 2P).
    void inverse_x(std::uint32_t* a, std::size_t j, std::uint32_t* out) const {
        if (block_ == 4) {  // one group of h = 1
            inverse_.visit(a, 4, j);
            std::copy_n(a, 16, out);
        } else if (aligned_) {  // a radix-4 group
            const std::size_t h = block_ / 4;
            for (std::size_t t = 0; t < 4; ++t) inverse_.visit(a + 8 * t * h, h, 4 * j + t);
            inverse_lower(a, h, Group(tables_.inverse_roots, j), out);
        } else {  // half of radix-4 group j / 2: a radix-2 level (the lower output needs no twiddle)
            const std::size_t h = block_ / 2;
            inverse_.visit(a, h, 2 * j);
            inverse_.visit(a + 8 * h, h, 2 * j + 1);
            for (std::size_t i = 0; i < 8 * h; i += 8) store(out + i, low(add(load(a + i), load(a + i + 8 * h))));
        }
    }

    // Block j at a: its forward levels from in (x < L/2; the rest zero), with the bottom unless
    // the block holds fewer than 4 groups of h = 1 (run() calls it for 16 vectors).
    void forward_x(const std::uint32_t* in, std::size_t j, std::uint32_t* a) const {
        if (block_ == 4) {
            std::copy_n(in, 16, a);
            std::fill_n(a + 16, 16, 0);
        } else if (aligned_) {
            const std::size_t h = block_ / 4;
            forward_lower(in, h, Group(tables_.roots, j), a);
            if (h == 4) return bottom_(a, 4, 4 * j);
            for (std::size_t t = 0; t < 4; ++t) forward_.visit(a + 8 * t * h, h, 4 * j + t);
        } else {  // radix 2 with a zero upper input: both outputs are the lower input
            const std::size_t h = block_ / 2;
            for (std::size_t i = 0; i < 8 * h; i += 8) {
                const Vec x = load(in + i);
                store(a + i, x), store(a + i + 8 * h, x);
            }
            if (h == 4) return;
            forward_.visit(a, h, 2 * j);
            forward_.visit(a + 8 * h, h, 2 * j + 1);
        }
    }

    // The inverse levels of group k (nv vectors at a) above the blocks, columns x < L/2.
    void above(std::uint32_t* a, std::size_t nv, std::size_t k) const {
        if (nv == block_) return;
        if (nv == 2 * block_) {  // the radix-2 level of radix-4 group k: (u + v, (u - v) x)
            const Factor x = entry(tables_.inverse_roots, k);
            for (std::size_t i = 0; i < 4 * block_; i += 8) {
                const Vec u = load(a + i), v = load(a + i + 8 * block_);
                store(a + i, low(add(u, v))), store(a + i + 8 * block_, times(diff(u, v), x));
            }
            return;
        }
        const std::size_t h = nv / 4;
        for (std::size_t t = 0; t < 4; ++t) above(a + 8 * t * h, h, 4 * k + t);
        inverse_columns(a, h, b_, tables_.inverse_roots, k);
    }

    // The inverse y levels of the transform of length 2m at a (canonical output times scale, as
    // inverse_pruned's top). Pruned forwards run groups of at most 64 vectors unpruned, so for
    // b <= 5 the level holding b (h = 16 or less) reads the other columns too: they are zeroed.
    void inverse_y(std::uint32_t* a, std::uint32_t scale) const {
        const std::size_t nv = half_ / 8, width = std::size_t(1) << b_;
        const bool zero = b_ <= 5;
        Vec* const v = reinterpret_cast<Vec*>(a);
        const Factor s(scale);
        const auto scaled = [&s](Vec x) { return reduce(times(x, s), kP); };
        const Vec z0 = _mm256_setzero_si256();
        if (std::countr_zero(nv) % 2 == 0) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) above(a + 8 * t * h, h, t);
            const Factor z(tables_.inverse_roots[1], tables_.inverse_roots[9]);
            for (std::size_t c = 0; c < h; c += 2 * width) {
                for (std::size_t j = c; j < c + width; ++j) {
                    const Vec x0 = v[j], x1 = v[j + h], x2 = v[j + 2 * h], x3 = v[j + 3 * h];
                    const Vec ab = low(add(x0, x1)), cd = low(add(x2, x3));
                    const Vec amb = low(diff(x0, x1)), cmd = times(diff(x2, x3), z);
                    v[j] = scaled(add(ab, cd)), v[j + h] = scaled(add(amb, cmd));
                    v[j + 2 * h] = scaled(diff(ab, cd)), v[j + 3 * h] = scaled(diff(amb, cmd));
                }
                if (zero)
                    for (std::size_t j = c + width; j < c + 2 * width; ++j) v[j] = v[j + h] = v[j + 2 * h] = v[j + 3 * h] = z0;
            }
        } else {
            const std::size_t h = nv / 2;
            above(a, h, 0);
            above(a + 8 * h, h, 1);
            for (std::size_t c = 0; c < h; c += 2 * width) {
                for (std::size_t j = c; j < c + width; ++j) {
                    const Vec x0 = v[j], x1 = v[j + h];
                    v[j] = scaled(add(x0, x1)), v[j + h] = scaled(diff(x0, x1));
                }
                if (zero)
                    for (std::size_t j = c + width; j < c + 2 * width; ++j) v[j] = v[j + h] = z0;
            }
        }
    }

    // The second half of the transform of length 4m from the rows mod (y^2Y + 1) at a[2m, 4m):
    // the top level's outputs of that half (as forward_pruned's top), then pruned forwards.
    void forward_upper() const {
        const std::size_t nv = half_ / 4, width = std::size_t(1) << b_;  // vectors of length 4m
        const Pruned<Bottom> pruned(tables_, bottom_, b_);
        if (std::countr_zero(nv) % 2 == 0) {  // radix-4 identity group: outputs 2, 3 = r0 +- z r1
            const std::size_t h = nv / 4;
            Vec* const v = reinterpret_cast<Vec*>(a_);
            const Factor z(tables_.roots[1], tables_.roots[9]);
            for (std::size_t c = 0; c < h; c += 2 * width)
                for (std::size_t j = c; j < c + width; ++j) {
                    const Vec x = v[2 * h + j], y = times(v[3 * h + j], z);
                    v[2 * h + j] = add(x, y), v[3 * h + j] = diff(x, y);
                }
            pruned.forward(a_ + 16 * h, h, 2);
            pruned.forward(a_ + 24 * h, h, 3);
        } else {  // radix 2: the upper output is the input
            pruned.forward(a_ + 4 * nv, nv / 2, 1);
        }
    }

    std::uint32_t* a_;
    std::size_t half_, block_, b_;
    bool aligned_;
    const Tables& tables_;
    Bottom bottom_;
    Recursion<InverseBottom> inverse_;
    Recursion<Bottom> forward_;
};

// Levels 0 and 1, one-dimensional in x, into q and p (length 4m each), ending in level 2's
// layout (stride m/2: Q_2 rows 0 .. 4, P_2 rows 0 .. 3, x below m/4; the rest of each row is not
// read by the pruned forward transforms).
// Level 0: Q_0 = 1 - y g, P_0 = x^e (e = m - n); LevelBottom on the transform of g at 2m gives
// v = g(x) g(-x) as a series in u = x^2. Level 1, series in x of length m/2:
// Q_1 = 1 + y q1 + y^2 q2 with q1 = -2 ge (g = ge(x^2) + x go(x^2)) and q2 = v, and
// P_1 = p0 + y p1 with p0 = x^d if e = 2d + 1 (else 0) and p1[j] = -(-1)^i g_i for i = 2j + 1 - e
// (the sign is the same for all j).
// With G(a) = a(x) a(-x) and E(a, b), O(a, b) the even part and the odd part (over x) of
// a(x) b(-x), all series in u, and ae, ao the even and odd parts of a:
//   Q_2 = 1 + 2y q1e + y^2 (2 q2e + G(q1)) + 2y^3 E(q1, q2) + y^4 G(q2),
//   P_2 = p0o + y (p1o + O(p0, q1)) + y^2 (O(p1, q1) + O(p0, q2)) + y^3 O(p1, q2), mod u^(m/4).
// O(p0, b) is a shift of b: coefficient i is (-1)^j b[j] for j = 2i + 1 - d. The five products:
// FirstLevelProducts on the transforms of q1, q2, p1 at m, inverses at m/2 (lower half). In units
// of h = m/2: q holds the transforms of q2 and q1 (q[0, 4h)), the coefficients of p1 at q[4h, 5h),
// O(p1, q1), O(p1, q2) at q[5h, 7h) and the coefficients of q1 at q[7h, 8h); p the transform of
// p1 (p[0, 2h)), G(q1), G(q2), E(q1, q2) at p[4h, 7h) and q2 at p[7h, 8h). P_2 goes to p[0, 4h),
// then Q_2 to q[0, 5h).
inline void projection_first_levels(const Transform& t, const Tables& tables, std::span<const std::uint32_t> g,
                                    std::size_t n, std::span<std::uint32_t> q, std::span<std::uint32_t> p) {
    const std::size_t m = q.size() / 4, h = m / 2, quarter = m / 4, e = m - n, size = std::min(n, g.size());
    const auto identity = [](Vec x) { return x; };
    const auto negated = [](Vec x) { return negate(x); };
    const auto doubled = [](Vec x) { return twice_mod(x); };
    const auto minus_twice = [](Vec x) { return negate(twice_mod(x)); };
    const auto row = [h, quarter](std::span<std::uint32_t> a, std::size_t k) { return a.subspan(k * h, quarter); };

    if (size > 0) {
        q[0] = 0;
        std::copy(g.begin() + 1, g.begin() + std::ptrdiff_t(size), q.begin() + 1);
    }
    forward_with(q.first(2 * m), size, tables, LevelBottom{&tables, q.data()});
    inverse_with(q.first(m), tables, InverseBottom{tables.inverse_roots, q.data()}, graeffe_scale(std::countr_zero(m)),
                 Half::kBoth);  // v
    const std::span<std::uint32_t> q2 = p.subspan(7 * h, h), q1 = q.subspan(7 * h, h), p1 = q.subspan(4 * h, h);
    std::copy_n(q.begin(), h, q2.begin());
    even_terms<false>(g, 1, size, 0, q1, minus_twice);
    if (e % 2) even_terms<false>(g, 1, size, 1 - std::ptrdiff_t(e), p1, negated);
    else even_terms<false>(g, 1, size, 1 - std::ptrdiff_t(e), p1, identity);

    const std::span<std::uint32_t> tq2 = q.first(m), tq1 = q.subspan(m, m), tp1 = p.first(m);
    t.forward(tq2.first(h), 0, tq2);
    t.forward(q1, 0, tq1);
    t.forward(p1, 0, tp1);
    std::uint32_t* const g1 = p.data() + 4 * h;
    std::uint32_t* const g2 = p.data() + 5 * h;
    std::uint32_t* const e12 = p.data() + 6 * h;
    std::uint32_t* const o1 = q.data() + 5 * h;
    std::uint32_t* const o2 = q.data() + 6 * h;
    const FirstLevelProducts<true> products{&tables, tq1.data(), tq2.data(), tp1.data(), {g1, g2, e12, o1, o2}};
    for (std::size_t k = 0; k < m / 16; k += 8) products.pairs(k);
    const std::uint32_t scale = graeffe_scale(std::countr_zero(h));
    for (std::uint32_t* out : products.out)
        inverse_with(std::span<std::uint32_t>(out, h), tables, InverseBottom{tables.inverse_roots, out}, scale, Half::kLower);

    std::fill(p.begin(), p.begin() + std::ptrdiff_t(quarter), 0);
    if (e % 2 && (e - 1) / 2 % 2) p[(e - 3) / 4] = 1;
    even_terms<false>(p1, 0, h, 1, row(p, 1), identity);
    std::copy_n(o1, quarter, p.begin() + std::ptrdiff_t(2 * h));
    std::copy_n(o2, quarter, p.begin() + std::ptrdiff_t(3 * h));
    if (e % 2) {  // the shifts O(p0, q1) and O(p0, q2): (-1)^j b[j] for j = 2i + 1 - d
        const std::size_t d = (e - 1) / 2;
        const std::ptrdiff_t offset = 1 - std::ptrdiff_t(d);
        const std::span<const std::uint32_t> b1 = q1, b2 = q2;
        if (d % 2) {
            even_terms<true>(b1, 0, h, offset, row(p, 1), identity);
            even_terms<true>(b2, 0, h, offset, row(p, 2), identity);
        } else {
            even_terms<true>(b1, 0, h, offset, row(p, 1), negated);
            even_terms<true>(b2, 0, h, offset, row(p, 2), negated);
        }
    }

    std::fill(q.begin(), q.begin() + std::ptrdiff_t(quarter), 0);
    q[0] = 1;
    even_terms<false>(q1, 0, h, 0, row(q, 1), doubled);
    even_terms<false>(q2, 0, h, 0, row(q, 2), doubled);
    add_into(row(q, 2), std::span<const std::uint32_t>(g1, quarter), row(q, 2));
    for (std::size_t i = 0; i < quarter; i += 8) store(q.data() + 3 * h + i, twice_mod(load(e12 + i)));
    std::copy_n(g2, quarter, q.begin() + std::ptrdiff_t(4 * h));
}

// Scratch words of projection_last_levels' work span for m.
inline std::size_t projection_work(std::size_t m) {
    const std::size_t eighth = m / 8;
    return std::max(7 * Carve::words(eighth + 1) + 8 * Carve::words(eighth), Carve::words(m) + 2 * Carve::words(m / 2));
}

// Levels T - 3 .. T - 1, one-dimensional in y, from Q_(T-3) (rows 0 .. m/8 at stride 16 in q, x
// below 8) and P_(T-3) (rows 0 .. m/8 - 1 at stride 16 in p), into a. w: projection_work(m) words.
// Level T - 3 (as composition.hpp's third_last_level): Q = sum_(k < 8) x^k q_k(y), q_0 = 1,
// deg q_k <= m/8, and P = sum_(k < 8) x^k p_k(y), deg p_k < m/8. With c_k = (-1)^k q_k, level T - 2
// has c'_k = (-1)^k q'_k for Q' = 1 + x q'_1 + x^2 q'_2 + x^3 q'_3 and P' = sum_(j < 4) x^j p'_j:
//   c'_1 = c1^2 - 2 c2,  c'_2 = c2^2 - 2 c1 c3 + 2 c4,  c'_3 = c3^2 + 2 c1 c5 - 2 c2 c4 - 2 c6,
//   p'_j = sum over a + b = 2j + 1 of p_a c_b (c_0 = 1),
// products of length m/4 (1 + 3 + 5 + 7 for P'). c'_k has degree <= m/4 and no y^0 term: its
// y^(m/4) wraps onto y^0. Levels T - 2 and T - 1: a = r3 + r1 (c'_1^2 - 2 c'_2) for
// r1 = p'_1 + p'_0 c'_1 and r3 = p'_3 + p'_2 c'_1 + p'_1 c'_2 + p'_0 c'_3 (products of length
// m/2), the last product of length m; c'_1^2 has no y^0 term, so its y^(m/2) wraps onto y^0.
// Memory: the columns go to w; the transforms of length m/4 and the twice-scaled ones of c3, c4,
// c5 to q and p, then c' (in q) and p' (in p) after them; the next transforms from the starts of q
// and p (below c' and p'), the rest to w.
inline void projection_last_levels(const Transform& t, const Tables& tables, std::span<std::uint32_t> q,
                                   std::span<std::uint32_t> p, std::span<std::uint32_t> w, std::span<std::uint32_t> a) {
    const std::size_t m = q.size() / 4, eighth = m / 8, quarter = m / 4, h = m / 2;
    std::span<std::uint32_t> c[8], pc[8];  // columns; c[0] unused
    {
        Carve carve(w);
        for (std::size_t k = 1; k < 8; ++k) c[k] = carve.take(eighth + 1);
        for (std::size_t k = 0; k < 8; ++k) pc[k] = carve.take(eighth);
    }
    for (std::size_t r = 0; r < eighth; r += 8) {  // rows r .. r + 7, transposed
        Vec x[8];
#pragma GCC unroll 8
        for (std::size_t i = 0; i < 8; ++i) x[i] = load(q.data() + 16 * (r + i));
        transpose(x);
#pragma GCC unroll 7
        for (std::size_t k = 1; k < 8; ++k) store(c[k].data() + r, k % 2 ? negate(x[k]) : x[k]);
    }
    for (std::size_t k = 1; k < 8; ++k) {
        const std::uint32_t x = q[16 * eighth + k];
        c[k][eighth] = k % 2 && x ? kP - x : x;
    }
    for (std::size_t r = 0; r < eighth; r += 8) {
        Vec x[8];
#pragma GCC unroll 8
        for (std::size_t i = 0; i < 8; ++i) x[i] = load(p.data() + 16 * (r + i));
        transpose(x);
#pragma GCC unroll 8
        for (std::size_t k = 0; k < 8; ++k) store(pc[k].data() + r, x[k]);
    }

    // Level T - 3.
    Carve carve_q(q), carve_p(p);
    std::span<std::uint32_t> tc[8], tp[7], twice[3], cn[4], pn[4];  // cn[k] = c'_k, pn[j] = p'_j
    for (std::size_t k = 1; k < 8; ++k) {
        tc[k] = carve_q.take(quarter);
        t.forward(c[k], 0, tc[k]);
    }
    for (std::size_t k = 0; k < 7; ++k) {
        tp[k] = carve_p.take(quarter);
        t.forward(pc[k], 0, tp[k]);
    }
    for (auto& s : twice) s = carve_q.take(quarter);  // 2 c5, -2 c3, -2 c4
    multiply_constant(tc[5], 2, twice[0]), multiply_constant(tc[3], kP - 2, twice[1]), multiply_constant(tc[4], kP - 2, twice[2]);
    for (std::size_t k = 1; k < 4; ++k) cn[k] = carve_q.take(quarter + 1);
    for (auto& s : pn) s = carve_p.take(quarter);
    {
        const Transform::Pair first[1] = {{tc[1], tc[1]}}, second[2] = {{tc[2], tc[2]}, {tc[1], twice[1]}},
                              third[3] = {{tc[3], tc[3]}, {tc[1], twice[0]}, {tc[2], twice[2]}};
        t.inverse_product_sum(first, cn[1].first(quarter));
        t.inverse_product_sum(second, cn[2].first(quarter));
        t.inverse_product_sum(third, cn[3].first(quarter));
    }
    const std::uint32_t weight[4] = {0, kP - 2, 2, kP - 2};
    for (std::size_t k = 1; k < 4; ++k) {  // c'_k[i] = square[i mod m/4] + weight[k] c_2k[i]
        const std::span<const std::uint32_t> linear = c[2 * k];
        const Factor f(weight[k]);
        for (std::size_t i = 0; i < eighth; i += 8)
            store(cn[k].data() + i, canonical(add(load(cn[k].data() + i), reduce(times(load(linear.data() + i), f), kP))));
        const std::uint64_t last = cn[k][eighth] + std::uint64_t(weight[k]) * linear[eighth];
        cn[k][eighth] = std::uint32_t(last % kP);
        cn[k][quarter] = cn[k][0];
        cn[k][0] = 0;
    }
    {
        Transform::Pair pairs[7];
        for (std::size_t j = 0; j < 4; ++j) {  // p'_j: pairs (p_a, c_b), a + b = 2j + 1, b >= 1
            for (std::size_t b = 1; b <= 2 * j + 1; ++b) pairs[b - 1] = {tp[2 * j + 1 - b], tc[b]};
            const std::span<const Transform::Pair> sum(pairs, 2 * j + 1);
            if (j < 2) t.inverse_product_sum(sum, pn[j]);
            else if (j == 2) inverse_product_sum<5>(tables, pairs, pn[j], Half::kBoth);
            else inverse_product_sum<7>(tables, pairs, pn[j], Half::kBoth);
            add_into(pn[j].first(eighth), pc[2 * j + 1], pn[j].first(eighth));
        }
    }

    // Levels T - 2 and T - 1.
    Carve next_q(q), next_p(p), next_w(w);
    std::span<std::uint32_t> tcn[4], tpn[3];
    for (std::size_t k = 1; k < 4; ++k) {
        tcn[k] = next_q.take(h);
        t.forward(cn[k], 0, tcn[k]);
    }
    for (std::size_t j = 0; j < 3; ++j) {
        tpn[j] = next_p.take(h);
        t.forward(pn[j], 0, tpn[j]);
    }
    const std::span<std::uint32_t> square = next_q.take(h), negative_q = next_w.take(m);  // c'_1^2 - 2 c'_2
    t.inverse_product(tcn[1], tcn[1], square);
    for (std::size_t i = 0; i < h; i += 8) {
        const Vec x = i < quarter ? negate(twice_mod(load(cn[2].data() + i))) : _mm256_setzero_si256();
        store(negative_q.data() + i, reduce(add(x, load(square.data() + i)), kP));
    }
    {  // c'_2 has degree <= m/4: its top term; square's y^(m/2) wrapped onto y^0, where the product is zero
        const std::uint32_t x = negative_q[quarter] + minus_twice(cn[2][quarter]);
        negative_q[quarter] = x >= kP ? x - kP : x;
        negative_q[h] = negative_q[0];
        negative_q[0] = 0;
        std::fill(negative_q.begin() + std::ptrdiff_t(h + 1), negative_q.end(), 0);
    }
    const std::span<std::uint32_t> r1 = next_w.take(h), r3 = next_w.take(h);
    t.inverse_product(tpn[0], tcn[1], r1);
    add_into(r1.first(quarter), pn[1], r1.first(quarter));
    const Transform::Pair pairs[3] = {{tpn[2], tcn[1]}, {tpn[1], tcn[2]}, {tpn[0], tcn[3]}};
    t.inverse_product_sum(pairs, r3);
    add_into(r3.first(quarter), pn[3], r3.first(quarter));
    t.forward(negative_q);
    const std::span<std::uint32_t> product = next_q.take(m);
    t.cyclic_product(r1, 0, product, negative_q);
    std::size_t k = 0;
    for (; k + 8 <= std::min(a.size(), h); k += 8)
        store_unaligned(a.data() + k, reduce(add(load(r3.data() + k), load(product.data() + k)), kP));
    for (; k < a.size(); ++k) {
        const std::uint32_t x = (k < h ? r3[k] : 0) + product[k];
        a[k] = x >= kP ? x - kP : x;
    }
}

}  // namespace detail

// Transform length power_projection() uses for n outputs: the Transform needs lg_max >= this.
inline int projection_log(std::size_t n) {
    if (n <= detail::kComposeBase) return Transform::kMinLog;
    return std::countr_zero(detail::projection_length(n));
}

// Scratch words for power_projection() of n outputs.
inline std::size_t projection_scratch(std::size_t n) {
    using detail::Carve;
    if (n <= detail::kComposeBase) return 0;
    const std::size_t m = detail::projection_length(n);
    return Arena::footprint(detail::Tables::words(m)) + 2 * Carve::words(4 * m) + Carve::words(detail::projection_work(m));
}

// a[k] = [x^(n-1)] g^k for k < n = a.size() >= 1, g[0] = 0 (if g is not empty); coefficients
// past g.size() are zero. scratch: projection_scratch(n) words, 32-byte aligned (from an Arena).
// t: lg_max >= projection_log(n). a must not overlap g.
//
// Levels as above: 0 and 1 (one-dimensional in x, projection_first_levels), 2 .. T - 4 (Kronecker
// layout: transforms of P_s and Q_s at 4m, the second with ProjectionBottom; inverses of V and W
// at 2m), T - 3 .. T - 1 (one-dimensional in y, projection_last_levels).
inline void power_projection(const Transform& t, std::span<const std::uint32_t> g, std::span<std::uint32_t> a,
                             std::span<std::uint32_t> scratch) {
    using namespace detail;
    const std::size_t n = a.size();
    if (n <= kComposeBase) return projection_direct(g, a);
    const std::size_t m = projection_length(n);
    const int lg = std::countr_zero(m);
    const Tables tables(scratch.first(Tables::words(m)), m);
    Carve carve(scratch.subspan(Arena::footprint(Tables::words(m))));
    const std::span<std::uint32_t> q = carve.take(4 * m), p = carve.take(4 * m), w = carve.take(projection_work(m));
    projection_first_levels(t, tables, g, n, q, p);
    const ProjectionBottom bottom{&tables, p.data(), q.data(), p.data()};
    const std::uint32_t half_scale = ntt::detail::power(kP / 2 + 1, 31);  // 2^-31: ProjectionBottom's factor
    std::uint32_t c = 1;  // level s's transforms are c times the true ones
    for (int s = 2; s + 3 < lg; ++s) {
        const std::size_t stride = 2 * (m >> s), rows = std::size_t(1) << s;  // Q_s: rows 0 .. Y; P_s: 0 .. Y - 1
        const std::size_t pad = std::size_t(std::countr_zero(stride / 16));  // vector bit of x = L
        if (s == 2) {
            forward_pruned(p, rows * stride, pad, tables, ForwardBottom{tables.roots});
            forward_pruned(q, (rows + 1) * stride, pad, tables, bottom);
        }
        if (s + 4 < lg) {  // level s + 1 <= T - 4: its transforms by doubling
            const std::size_t block = stride / 16;
            const std::uint32_t next =
                ntt::detail::multiply_mod(ntt::detail::multiply_mod(ntt::detail::multiply_mod(c, c), half_scale), std::uint32_t(block));
            const std::uint32_t y_scale = kInverseScales[0][s + 4];  // (2^(s+1))^-1
            Doubling<ForwardBottom>(p, block, tables, ForwardBottom{tables.roots}).run(y_scale, 0);
            Doubling<ProjectionBottom>(q, block, tables, bottom).run(y_scale, next + next >= kP ? next + next - kP : next + next);
            c = next;
        } else {  // level T - 4: only x < L/2 is kept, the next forwards do not read the rest of each row
            const std::uint32_t inverse_c = ntt::detail::power(c, kP - 2);
            const std::uint32_t scale = ntt::detail::multiply_mod(graeffe_scale(lg + 1), ntt::detail::multiply_mod(inverse_c, inverse_c));
            inverse_pruned(q.first(2 * m), pad - 1, tables, InverseBottom{tables.inverse_roots, q.data()}, scale);
            inverse_pruned(p.first(2 * m), pad - 1, tables, InverseBottom{tables.inverse_roots, p.data()}, scale);
            next_level(q, stride / 2, 2 * rows);
        }
    }
    projection_last_levels(t, tables, q, p, w, a);
}

}  // namespace poly
