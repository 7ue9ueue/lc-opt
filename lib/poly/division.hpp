// Division with remainder of polynomials modulo 998244353: f = q g + r with deg r < deg g.
// x86-64 with AVX2. Design: lib/poly/notes.md (Division).
//
//   poly::Arena arena(poly::divide_words(n, m));
//   poly::divide(arena, f, g, q, r);  // f: n coefficients, g: m >= 1 with g[m - 1] != 0,
//                                     // q: quotient_size(n, m), r: remainder_size(n, m)
//
// With d = m - 1, k = n - d, F = rev(f) and G = rev(g) (G[0] = g[d]): rev(q) = F / G mod x^k and
// rev(r) = (F - G rev(q))[k, n).
// - Quotient in blocks of s coefficients, F padded in front to a multiple of s: with
//   h = 1 / G mod x^s, Q_j = h R_j mod x^s for the residual R_j = (F - G Q)[js, js + s) of the
//   blocks before. If d < s, R_j is F's block minus (G T)[d, 2d) in its first d coefficients, T
//   the last d coefficients of Q_(j-1): a cyclic product of length >= 2d ("tails"). Else
//   R_j = F_j - sum_t (W_t Q_(j-t))[s, 2s) over the windows W_t = G[(t-1)s, (t+1)s), from stored
//   transforms of length 2s (as log.hpp's blocks).
// - Remainder: r = f - q g mod x^d. The coefficients of q g past d are f's, so q g mod (x^L - 1)
//   for any L >= d gives it; for L < d < 2L, (q g)[L, d) comes from a middle product with the
//   same transform of q. With tails, rev(r) = (F - G Q)[k, n) is one more tail product. For a
//   short quotient, directly.
// - s and the remainder's method minimize a cost model of the transforms.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "lib/poly/inverse.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

// Coefficients of the quotient and of the remainder of f (n coefficients) by g (m >= 1).
inline std::size_t quotient_size(std::size_t n, std::size_t m) { return n >= m ? n - m + 1 : 0; }
inline std::size_t remainder_size(std::size_t n, std::size_t m) { return std::min(n, m - 1); }

namespace detail {

// Window products per residual: InverseProductSumBottom<K> for K up to this.
inline constexpr std::size_t kDivisionTerms = 8;

// Long division when k d is at most this.
inline constexpr std::size_t kLongDivision = std::size_t(1) << 12;

// The remainder may be computed directly for quotients of up to this many coefficients.
inline constexpr std::size_t kDirectRemainder = 64;

// Cost model, ns per coefficient on lc-amd: a forward or inverse transform of length 2^lg, a leaf
// product, a term of the direct remainder, a word of a fold.
inline constexpr std::array<double, 26> kTransformNs = {0,    0,    0,    0,    0,    0,    0.45, 0.48, 0.51,
                                                        0.55, 0.6,  0.65, 0.7,  0.76, 0.81, 0.88, 0.95, 1.02,
                                                        1.09, 1.17, 1.23, 1.3,  1.37, 1.44, 1.51, 1.58};
inline constexpr double kLeafProductNs = 0.92, kDirectTermNs = 0.04, kFoldNs = 0.25;

inline std::size_t transform_length(std::size_t n) { return std::max<std::size_t>(64, std::bit_ceil(n)); }

inline double transform_ns(std::size_t len) { return kTransformNs[std::countr_zero(len)] * double(len); }
inline double cyclic_ns(std::size_t len) { return 2 * transform_ns(len) + kLeafProductNs * double(len); }

// inverse() of n coefficients.
inline double inverse_ns(std::size_t n) {
    double ns = 0;
    for (std::size_t k = kInverseBase; k < n; k *= 2) ns += transform_ns(2 * k) + 2 * cyclic_ns(2 * k);
    return ns;
}

// The quotient's k coefficients in `blocks` blocks of s after pad zeros, transforms of length 2s.
// tails (d < s): residuals from Q_(j-1)'s last d coefficients by cyclic products of length tail;
// else from `windows` stored window transforms W_t.
struct QuotientPlan {
    std::size_t s = 0, blocks = 0, pad = 0, windows = 0, tail = 0;
    bool tails = false;
    double ns = std::numeric_limits<double>::infinity();

    QuotientPlan() = default;

    QuotientPlan(std::size_t k, std::size_t d, std::size_t block)
        : s(block), blocks((k + block - 1) / block), pad(blocks * block - k), tail(transform_length(2 * d)),
          tails(d < block) {
        const std::size_t len = 2 * s;
        ns = inverse_ns(precision(k)) + transform_ns(len) + double(blocks) * cyclic_ns(len);
        if (blocks == 1) return;
        if (tails) {
            ns += transform_ns(tail) + double(blocks - 1) * cyclic_ns(tail);
            return;
        }
        const std::size_t reach = d / s + 1;  // nonzero windows
        windows = std::min(reach, blocks - 1);
        if (windows > kDivisionTerms) {
            ns = std::numeric_limits<double>::infinity();
            return;
        }
        double terms = 0;
        for (std::size_t j = 1; j < blocks; ++j) terms += double(std::min(j, reach));
        ns += double(windows + 2 * (blocks - 1)) * transform_ns(len) + terms * kLeafProductNs * double(len);
    }

    // Coefficients of h: one block needs k only.
    std::size_t precision(std::size_t k) const { return blocks == 1 ? k : s; }
};

// The remainder: directly (k <= kDirectRemainder); from q g mod (x^L - 1) for L >= d (kWrap);
// for d = L + e with 0 < e <= L - k + 1, from q g mod (x^L - 1) and (q g)[L, d) by a middle
// product, both with the transform of q of length L (kSplit); with tails, rev(r) =
// (F - G Q)[k, n) from one more tail product (kTail).
enum class Remainder { kDirect, kWrap, kSplit, kTail };

struct RemainderPlan {
    Remainder method;
    std::size_t length = 0;  // of kWrap's and kSplit's transforms
    double ns = std::numeric_limits<double>::infinity();

    RemainderPlan(Remainder how, std::size_t n, std::size_t k, std::size_t d, const QuotientPlan& quotient) : method(how) {
        const std::size_t wrap = transform_length(d);
        switch (how) {
            case Remainder::kDirect:
                if (k <= kDirectRemainder) ns = double(k) * double(d) * kDirectTermNs;
                break;
            case Remainder::kWrap:
                length = wrap;
                ns = cyclic_ns(length) + transform_ns(length) + double(n + k) * kFoldNs;
                break;
            case Remainder::kSplit:
                length = wrap / 2;
                if (length >= 64 && d > length && k + (d - length) <= length + 1)
                    ns = transform_ns(length) + 2 * cyclic_ns(length) + double(n + d) * kFoldNs;
                break;
            case Remainder::kTail:  // T(G) for the tails is there if blocks > 1
                if (!quotient.tails) break;
                ns = (quotient.blocks == 1 ? transform_ns(quotient.tail) : 0) + cyclic_ns(quotient.tail) + double(d) * kFoldNs;
                break;
        }
    }

    std::size_t buffers() const { return method == Remainder::kWrap ? 2 : method == Remainder::kSplit ? 3 : 0; }
};

// The cheapest plan for n and m >= 1 with n >= m, or a given one (tests).
struct DivisionPlan {
    std::size_t n, m, k, d;
    bool long_division;
    QuotientPlan quotient;
    RemainderPlan remainder;
    int lg = Transform::kMinLog;

    DivisionPlan(std::size_t n_, std::size_t m_)
        : n(n_), m(m_), k(quotient_size(n_, m_)), d(m_ - 1), long_division(k * d <= kLongDivision),
          remainder(Remainder::kWrap, n, k, d, quotient) {
        if (long_division) return;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t s = 32;; s *= 2) {  // up to one block
            const QuotientPlan p(k, d, s);
            for (Remainder how : {Remainder::kDirect, Remainder::kWrap, Remainder::kSplit, Remainder::kTail}) {
                const RemainderPlan r(how, n, k, d, p);
                if (p.ns + r.ns < best) best = p.ns + r.ns, quotient = p, remainder = r;
            }
            if (s >= k || 4 * s > (std::size_t(1) << Transform::kMaxLog)) break;
        }
        set_log();
    }

    // Blocks of s and a given remainder method.
    DivisionPlan(std::size_t n_, std::size_t m_, std::size_t s, Remainder how)
        : n(n_), m(m_), k(quotient_size(n_, m_)), d(m_ - 1), long_division(false), quotient(k, d, s),
          remainder(how, n, k, d, quotient) {
        set_log();
    }

    // Whether tail products run (and the transform of G for them is computed).
    bool tails() const { return quotient.tails && (quotient.blocks > 1 || remainder.method == Remainder::kTail); }

    void set_log() { lg = std::countr_zero(std::max({2 * quotient.s, tails() ? quotient.tail : 0, remainder.length})); }

    // Coefficients of G read: by the inverse, the windows, the tails.
    std::size_t divisor() const {
        const QuotientPlan& p = quotient;
        return tails() ? m : std::min(m, std::max(p.precision(k), (p.windows + 1) * p.s));
    }

    // Scratch words of the quotient: G, T(h), then the inverse's scratch or (later) the blocks
    // and the tails' transforms or the windows.
    std::size_t quotient_words() const {
        const QuotientPlan& p = quotient;
        const std::size_t len = 2 * p.s;
        std::size_t later = Arena::footprint(p.blocks * p.s + p.s);
        if (tails()) later += 2 * Arena::footprint(p.tail);
        if (p.windows) later += (2 * p.windows + 1) * Arena::footprint(len);
        return Arena::footprint(divisor()) + Arena::footprint(len) + std::max(inverse_scratch(p.precision(k)), later);
    }

    // Scratch words: the quotient's, then the remainder's in the same place.
    std::size_t scratch() const { return std::max(quotient_words(), remainder.buffers() * Arena::footprint(remainder.length)); }

    // Arena words of divide().
    std::size_t words() const {
        return long_division ? Arena::footprint(n) : Transform::words(lg) + Arena::footprint(scratch());
    }
};

// Consecutive spans of a scratch span, each placed as Arena::take places them.
class Bump {
public:
    explicit Bump(std::span<std::uint32_t> memory) : memory_(memory) {}

    std::span<std::uint32_t> take(std::size_t n) {
        const std::span<std::uint32_t> s = memory_.subspan(used_, n);
        used_ += Arena::footprint(n);
        return s;
    }

    std::span<std::uint32_t> rest() const { return memory_.subspan(used_); }

private:
    std::span<std::uint32_t> memory_;
    std::size_t used_ = 0;
};

// 64-bit lanes s < 2^32 P (even lanes from even, odd from odd) -> s / 2^32 mod P in [0, 2P).
inline Vec montgomery_reduce(Vec even, Vec odd) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
    odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
    return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
}

inline Vec load_any(const std::uint32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
inline void store_any(std::uint32_t* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }
inline Vec reversed(Vec x) { return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

// out[i] = a[i] - b[i] mod P for i < count, canonical inputs.
inline void subtract(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) store_any(out + i, difference(load_any(a + i), load_any(b + i)));
    for (; i < count; ++i) out[i] = (a[i] + kP - b[i]) % kP;
}

// out[i] = a[top - i] - sub[i] mod P for i < count (no subtraction if sub is null), where
// a[p] = 0 outside [0, a.size()). Canonical inputs.
inline void reverse_difference(std::span<const std::uint32_t> a, std::ptrdiff_t top, std::uint32_t* out, std::size_t count,
                               const std::uint32_t* sub = nullptr) {
    const std::ptrdiff_t size = std::ptrdiff_t(a.size());
    const auto value = [&](std::size_t i) {
        const std::ptrdiff_t p = top - std::ptrdiff_t(i);
        const std::uint32_t x = p >= 0 && p < size ? a[std::size_t(p)] : 0;
        return sub ? (x + kP - sub[i]) % kP : x;
    };
    std::size_t i = top >= size ? std::min(count, std::size_t(top - size + 1)) : 0;  // a[top - i] = 0 below i
    if (sub) {
        for (std::size_t j = 0; j < i; ++j) out[j] = value(j);
    } else {
        std::fill_n(out, i, 0);
    }
    for (; i + 8 <= count && top - std::ptrdiff_t(i) >= 7; i += 8) {
        const Vec x = reversed(load_any(a.data() + (top - std::ptrdiff_t(i) - 7)));
        store_any(out + i, sub ? difference(x, load_any(sub + i)) : x);
    }
    for (; i < count; ++i) out[i] = value(i);
}

// out[j] = sum over t of a[j + t out.size()] mod P.
inline void fold(std::span<const std::uint32_t> a, std::span<std::uint32_t> out) {
    const std::size_t len = out.size();
    std::fill(out.begin(), out.end(), 0);
    for (std::size_t from = 0; from < a.size(); from += len) {
        const std::size_t count = std::min(len, a.size() - from), full = count / 8 * 8;
        const std::uint32_t* x = a.data() + from;
        for (std::size_t j = 0; j < full; j += 8) {
            store(out.data() + j, reduce(add(load(out.data() + j), load_any(x + j)), kP));
        }
        for (std::size_t j = full; j < count; ++j) out[j] = (out[j] + x[j]) % kP;
    }
}

// Long division: q and r from f and g, rest: n words of scratch.
inline void long_division(std::span<const std::uint32_t> f, std::span<const std::uint32_t> g, std::span<std::uint32_t> q,
                          std::span<std::uint32_t> r, std::span<std::uint32_t> rest) {
    using ntt::detail::multiply_mod;
    const std::size_t d = g.size() - 1;
    std::copy(f.begin(), f.end(), rest.begin());
    const std::uint32_t lead = ntt::detail::power(g[d], kP - 2);
    for (std::size_t i = q.size(); i-- > 0;) {
        const std::uint32_t c = multiply_mod(rest[i + d], lead);
        q[i] = c;
        for (std::size_t t = 0; t < d; ++t) rest[i + t] = (rest[i + t] + kP - multiply_mod(c, g[t])) % kP;
    }
    std::copy_n(rest.begin(), r.size(), r.begin());
}

// r[j] = f[j] - sum_i q[i] g[j - i] for j < d = r.size() and k = q.size() <= kDirectRemainder.
// Terms in groups of 4 (4 P^2 < 2^32 P), q times 2^32 for the Montgomery reduction.
inline void remainder_direct(std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                             std::span<const std::uint32_t> q, std::span<std::uint32_t> r) {
    const std::size_t d = r.size(), k = q.size();
    Vec factor[kDirectRemainder];
    for (std::size_t i = 0; i < k; ++i) factor[i] = _mm256_set1_epi64x(std::int64_t((std::uint64_t(q[i]) << 32) % kP));
    const auto scalar = [&](std::size_t j) {
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i <= std::min(j, k - 1); ++i) sum = (sum + std::uint64_t(q[i]) * g[j - i]) % kP;
        r[j] = std::uint32_t((f[j] + kP - sum) % kP);
    };
    std::size_t j = 0;
    for (; j < std::min(d, k); ++j) scalar(j);
    for (; j + 8 <= d; j += 8) {  // g[j - i + 7] <= g[d - 1]
        Vec total = load_any(f.data() + j);  // in [0, 2P)
        for (std::size_t i = 0; i < k; i += 4) {
            Vec even = _mm256_setzero_si256(), odd = _mm256_setzero_si256();
            for (std::size_t t = i; t < std::min(k, i + 4); ++t) {
                const Vec x = load_any(g.data() + (j - t));
                even = _mm256_add_epi64(even, _mm256_mul_epu32(x, factor[t]));
                odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_srli_epi64(x, 32), factor[t]));
            }
            total = reduce(_mm256_sub_epi32(add(total, broadcast(2 * kP)), montgomery_reduce(even, odd)), 2 * kP);
        }
        store_any(r.data() + j, reduce(total, kP));
    }
    for (; j < d; ++j) scalar(j);
}

// r = f - q g mod x^d from c = q g mod (x^L - 1), L = a.size() = b.size() >= d: for j < d,
// c[j] = (q g)[j] + f[j + L] + f[j + 2L] + ...
inline void remainder_wrap(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                           std::span<const std::uint32_t> q, std::span<std::uint32_t> r, std::span<std::uint32_t> a,
                           std::span<std::uint32_t> b) {
    const std::size_t len = a.size();
    if (q.size() <= len) {
        t.forward(q, 0, a);
    } else {
        fold(q, a);
        t.forward(a);
    }
    std::span<const std::uint32_t> gl = g;
    if (g.size() > len) {  // g.size() = len + 1
        std::copy_n(g.begin(), len, b.begin());
        b[0] = (b[0] + g[len]) % kP;
        gl = b;
    }
    t.cyclic_product(gl, 0, b, a);
    fold(f, a);
    subtract(a.data(), b.data(), r.data(), r.size());
}

// The same for d = L + e, L = a.size(), 0 < e and k + e <= L + 1 (k = q.size()): with
// c = q g mod (x^L - 1), for j < L
//   r[j] = (f[j] + f[j + L] + ...) - c[j] - (j < e ? r[L + j] : 0),
// and r[L + i] = f[L + i] - (q g)[L + i], the coefficients [k - 1, k - 1 + e) of q g[L - k + 1, d)
// (exact mod x^L - 1).
inline void remainder_split(const Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                            std::span<const std::uint32_t> q, std::span<std::uint32_t> r, std::span<std::uint32_t> a,
                            std::span<std::uint32_t> b, std::span<std::uint32_t> c) {
    const std::size_t len = a.size(), k = q.size(), e = r.size() - len;
    t.forward(q, 0, a);
    t.cyclic_product(g.subspan(len + 1 - k, k + e - 1), 0, c, a);
    subtract(f.data() + len, c.data() + (k - 1), r.data() + len, e);
    fold(g, b);
    t.cyclic_product(b, a);
    fold(f, c);
    subtract(c.data(), b.data(), r.data(), len);
    subtract(r.data(), r.data() + len, r.data(), e);
}

// out = the upper half of sum_k a_k b_k mod (x^n - 1), K <= kDivisionTerms pairs of transforms.
template <std::size_t K>
void residual_sum(const Transform& t, std::span<const Transform::Pair> pairs, std::span<std::uint32_t> out) {
    InverseProductSumBottom<K> bottom{};
    for (std::size_t i = 0; i < K; ++i)
        bottom.terms[i] = {{t.roots(), t.inverse_roots(), pairs[i].b.data()}, pairs[i].a.data()};
    t.inverse_with(bottom, out, Half::kUpper);
}

inline void residual_sum(const Transform& t, std::span<const Transform::Pair> pairs, std::span<std::uint32_t> out) {
    switch (pairs.size()) {
        case 1:
        case 2:
        case 3: return t.inverse_product_sum(pairs, out, Half::kUpper);
        case 4: return residual_sum<4>(t, pairs, out);
        case 5: return residual_sum<5>(t, pairs, out);
        case 6: return residual_sum<6>(t, pairs, out);
        case 7: return residual_sum<7>(t, pairs, out);
        case 8: return residual_sum<8>(t, pairs, out);
        default: std::abort();
    }
}

// out[j] = a[j] - b[count - 1 - j] mod P for j < count, canonical inputs.
inline void subtract_reversed(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* out, std::size_t count) {
    std::size_t j = 0;
    for (; j + 8 <= count; j += 8) store_any(out + j, difference(load_any(a + j), reversed(load_any(b + count - 8 - j))));
    for (; j < count; ++j) out[j] = (a[j] + kP - b[count - 1 - j]) % kP;
}

// q from Q = x^pad F / G mod x^(blocks s), F = rev(f), block by block (q = rev(Q) without the
// pad); with kTail also r. scratch: plan.quotient_words() words.
inline void divide_blocks(const Transform& t, const DivisionPlan& plan, std::span<const std::uint32_t> f,
                          std::span<const std::uint32_t> g, std::span<std::uint32_t> q, std::span<std::uint32_t> r,
                          std::span<std::uint32_t> scratch) {
    const QuotientPlan& p = plan.quotient;
    const std::size_t k = q.size(), s = p.s, len = 2 * s, d = plan.d;
    const std::ptrdiff_t top = std::ptrdiff_t(f.size() - 1 + p.pad);  // F_j[i] = f[top - js - i]
    Bump bump(scratch);
    const std::span<std::uint32_t> G = bump.take(plan.divisor()), ht = bump.take(len);
    std::reverse_copy(g.end() - std::ptrdiff_t(G.size()), g.end(), G.begin());
    const std::size_t precision = p.precision(k);
    inverse(t, G.first(std::min(G.size(), precision)), ht.first(precision), bump.rest());
    t.forward(ht.first(precision), 0, ht);

    const std::span<std::uint32_t> qs = bump.take(p.blocks * s + s);
    std::span<std::uint32_t> gt, mp, work;
    std::array<std::span<std::uint32_t>, kDivisionTerms> windows{}, ring{};
    if (plan.tails()) {
        gt = bump.take(p.tail), mp = bump.take(p.tail);
        t.forward(G, 0, gt);
    }
    if (p.windows) {
        work = bump.take(len);
        for (std::size_t i = 0; i < p.windows; ++i) {
            windows[i] = bump.take(len), ring[i] = bump.take(len);
            t.forward(G.subspan(i * s, std::min(len, G.size() - i * s)), 0, windows[i]);
        }
    }
    // (G T)[d, 2d) for the last d coefficients T of block j - 1, at the top of mp: x^(tail - 2d) T G
    // has degree < tail.
    const auto tail_product = [&](std::size_t j) {
        t.cyclic_product(qs.subspan(j * s - d, d), p.tail - 2 * d, mp, gt, Half::kUpper);
        return mp.data() + (p.tail - d);
    };
    for (std::size_t j = 0; j < p.blocks; ++j) {
        std::uint32_t* const block = qs.data() + j * s;
        const std::ptrdiff_t from = top - std::ptrdiff_t(j * s);
        if (j == 0) {
            reverse_difference(f, from, block, s);
        } else if (p.tails) {
            reverse_difference(f, from, block, d, tail_product(j));
            reverse_difference(f, from - std::ptrdiff_t(d), block + d, s - d);
        } else {
            const std::size_t terms = std::min(j, d / s + 1);
            std::array<Transform::Pair, kDivisionTerms> pairs;
            for (std::size_t i = 0; i < terms; ++i) pairs[i] = {windows[i], ring[(j - 1 - i) % p.windows]};
            residual_sum(t, std::span<const Transform::Pair>(pairs.data(), terms), work);
            reverse_difference(f, from, block, s, work.data() + s);
        }
        t.cyclic_product(qs.subspan(j * s, s), 0, qs.subspan(j * s, len), ht, Half::kLower);
        if (p.windows && j + 1 < p.blocks) t.forward(qs.subspan(j * s, s), 0, ring[j % p.windows]);
        // q[i] = Q[blocks s - 1 - i]: block j, without its pad, is q[(blocks - 1 - j) s, ...)
        reverse_difference(qs.first((j + 1) * s), std::ptrdiff_t((j + 1) * s - 1), q.data() + (p.blocks - 1 - j) * s,
                           j == 0 ? s - p.pad : s);
    }
    // rev(r) = (F - G Q)[blocks s, blocks s + d) and F[blocks s + i] = f[d - 1 - i]
    if (plan.remainder.method == Remainder::kTail) subtract_reversed(f.data(), tail_product(p.blocks), r.data(), d);
}

inline void divide(Arena& arena, const DivisionPlan& plan, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                   std::span<std::uint32_t> q, std::span<std::uint32_t> r) {
    if (plan.long_division) return long_division(f, g, q, r, arena.take(f.size()));
    const Transform t(arena, plan.lg);
    const std::span<std::uint32_t> scratch = arena.take(plan.scratch());
    divide_blocks(t, plan, f, g, q, r, scratch);
    Bump bump(scratch);
    const std::size_t len = plan.remainder.length;
    switch (plan.remainder.method) {
        case Remainder::kDirect: return remainder_direct(f, g, q, r);
        case Remainder::kWrap: return remainder_wrap(t, f, g, q, r, bump.take(len), bump.take(len));
        case Remainder::kSplit: return remainder_split(t, f, g, q, r, bump.take(len), bump.take(len), bump.take(len));
        case Remainder::kTail: return;
    }
}

}  // namespace detail

// Arena words divide() takes for n coefficients of f and m of g.
inline std::size_t divide_words(std::size_t n, std::size_t m) { return detail::DivisionPlan(n, m).words(); }

// q, r with f = q g + r and deg r < d = m - 1, for f of n = f.size() coefficients and g of
// m = g.size() >= 1, g[m - 1] != 0, all canonical. q: quotient_size(n, m) words (n - d if n > d,
// else 0); r: remainder_size(n, m) words (min(n, d); r = f if n <= d). None may overlap.
// arena: divide_words(n, m) words.
inline void divide(Arena& arena, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                   std::span<std::uint32_t> q, std::span<std::uint32_t> r) {
    const detail::DivisionPlan plan(f.size(), g.size());
    if (plan.k == 0) {
        std::copy(f.begin(), f.end(), r.begin());
        return;
    }
    detail::divide(arena, plan, f, g, q, r);
}

}  // namespace poly
