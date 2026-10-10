// Polynomials of degree < 128 over F_p (odd p < 2^30) with Montgomery coefficients: gcd, division,
// products modulo a monic g by a table of x^(m + j) mod g, powers modulo g.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "problems/polynomial/factorization_of_polynomials/field.hpp"

namespace factor {

inline constexpr int kCap = 128;  // coefficients of a Poly; inputs have degree <= 100

// c[0, n) with c[n - 1] != 0 (n = 0: the zero polynomial); c is zero from n on, so kernels may
// read and write whole vectors past n.
struct Poly {
    int n = 0;
    alignas(32) u32 c[kCap] = {};

    int degree() const { return n - 1; }
    u32 lead() const { return c[n - 1]; }
    void trim() {
        while (n > 0 && c[n - 1] == 0) --n;
    }
};

// A monic g of degree m >= 1 with T_j = x^(m + j) mod g, j < m - 1, at table + j stride.
struct Modulus {
    Poly g;
    int m = 0;
    int stride = 0;
    std::vector<u32> table;
};

class Ring {
public:
    explicit Ring(const Field& field) : F(field) {}

    const Field& F;

    Poly constant(u32 a) const {
        Poly r;
        r.c[0] = a;
        r.n = a != 0;
        return r;
    }
    // x - z.
    Poly linear(u32 z) const {
        Poly r;
        r.c[0] = F.neg(z), r.c[1] = F.one, r.n = 2;
        return r;
    }

    void make_monic(Poly& a) const {
        const u32 s = F.inverse(a.lead());
        for (int i = 0; i < a.n; ++i) a.c[i] = F.mul(a.c[i], s);
    }

    // Monic gcd (zero if both are zero). Fraction-free Euclid: a <- lc(b) a - lc(a) x^s b kills
    // the leading term without inverses; remainders are right up to constant factors.
    Poly gcd(const Poly& a0, const Poly& b0) const {
        alignas(32) u32 buffer[2][2 * kCap + 8];  // kCap zeros, then the coefficients
        int na = a0.n, nb = b0.n;
        const int lead = std::max(na, nb), span = round8(lead) + 8;  // reads reach a - lead, a + span
        u32* a = buffer[0] + kCap;
        u32* b = buffer[1] + kCap;
        for (u32* x : {a, b}) std::fill(x - lead, x + span, 0);
        std::copy(a0.c, a0.c + na, a);
        std::copy(b0.c, b0.c + nb, b);
        if (na < nb) std::swap(a, b), std::swap(na, nb);
        while (nb > 0) {
            while (na >= nb) {
                lincomb(F, a, a, b[nb - 1], b - (na - nb), F.neg(a[na - 1]), na - 1);
                a[--na] = 0;
                while (na > 0 && a[na - 1] == 0) --na;
            }
            std::swap(a, b), std::swap(na, nb);
        }
        Poly g;
        g.n = na;
        std::copy(a, a + na, g.c);
        if (na > 0) make_monic(g);
        return g;
    }

    // h = q d + r with deg r < deg d, d monic. q or r may be null.
    void divide(const Poly& h, const Poly& d, Poly* q, Poly* r) const {
        alignas(32) u32 a[kCap + 8] = {};
        std::copy(h.c, h.c + h.n, a);
        const int dd = d.n - 1;
        if (q) *q = Poly{}, q->n = std::max(0, h.n - dd);
        for (int i = h.n - 1; i >= dd; --i) {
            const u32 c = a[i];
            if (q) q->c[i - dd] = c;
            if (c != 0) lincomb(F, a + i - dd, a + i - dd, F.one, d.c, F.neg(c), dd + 1);  // a[i] -> 0
        }
        if (r) {
            *r = Poly{};
            r->n = std::min(h.n, dd);
            std::copy(a, a + r->n, r->c);
            r->trim();
        }
    }
    Poly quotient(const Poly& h, const Poly& d) const {
        Poly q;
        divide(h, d, &q, nullptr);
        return q;
    }
    Poly remainder(const Poly& h, const Poly& d) const {
        if (h.n < d.n) return h;
        Poly r;
        divide(h, d, nullptr, &r);
        return r;
    }

    // y = x a mod g for a of degree < m = deg g (y may be a).
    void times_x(const u32* a, u32* y, const Poly& g) const {
        const int m = g.n - 1;
        alignas(32) u32 shifted[kCap + 8];
        shifted[0] = 0;
        std::copy(a, a + m, shifted + 1);
        std::fill(shifted + m + 1, shifted + round8(m) + 8, 0);
        lincomb(F, y, shifted, F.one, g.c, F.neg(a[m - 1]), m);  // y[m] = a[m-1] - a[m-1] = 0
    }

    // The table goes into storage (its capacity is reused).
    Modulus modulus(const Poly& g, std::vector<u32>&& storage = {}) const {
        Modulus M;
        M.g = g;
        M.m = g.n - 1;
        M.stride = round8(M.m);
        M.table = std::move(storage);
        M.table.assign(std::size_t(std::max(M.m - 1, 0)) * M.stride, 0);
        if (M.m >= 2) {
            u32* t = M.table.data();
            for (int i = 0; i < M.m; ++i) t[i] = F.neg(g.c[i]);
            for (int j = 1; j < M.m - 1; ++j) times_x(t + (j - 1) * M.stride, t + j * M.stride, g);
        }
        return M;
    }

    // a b mod g; a and b of degree < m.
    Poly multiply_mod(const Poly& a, const Poly& b, const Modulus& M) const {
        Poly r;
        if (a.n == 0 || b.n == 0) return r;
        if (M.m <= 8) return to_poly(multiply_small(load(a.c), load(b.c), M), M.m);
        alignas(32) u32 product[kProductCap];
        multiply(F, a.c, a.n, b.c, b.n, product);
        const int len = a.n + b.n - 1;
        r.n = std::min(len, M.m);
        std::copy(product, product + r.n, r.c);
        if (len > M.m) combine(F, r.c, M.m, r.c, product + M.m, len - M.m, M.table.data(), M.stride);
        r.trim();
        return r;
    }

    // a b mod h without a table; deg a + deg b < kCap.
    Poly multiply_remainder(const Poly& a, const Poly& b, const Poly& h) const {
        Poly c;
        if (a.n == 0 || b.n == 0) return c;
        alignas(32) u32 product[kProductCap];
        multiply(F, a.c, a.n, b.c, b.n, product);
        c.n = a.n + b.n - 1;
        std::copy(product, product + c.n, c.c);
        return remainder(c, h);
    }

    Poly times_x(const Poly& a, const Modulus& M) const {
        Poly r;
        if (a.n == 0) return r;
        times_x(a.c, r.c, M.g);
        r.n = M.m;
        r.trim();
        return r;
    }

    // Polynomials modulo g of degree m <= 8 in one vector (lanes from m on zero): chains of 30
    // squarings at tiny degrees (root finding, splitting) are latency-bound, so no memory trips.
    // a b mod g: product in two vectors (b shifted in registers), then c[0, m) + sum c[m + j] T_j.
    __m256i multiply_small(__m256i a, __m256i b, const Modulus& M) const {
        const int m = M.m;
        const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), zero = _mm256_setzero_si256();
        __m256i lo_even = zero, lo_odd = zero, hi_even = zero, hi_odd = zero;
        for (int i = 0; i < m; ++i) {  // at most 8 products per sum: below 2^63
            const __m256i x = _mm256_permutevar8x32_epi32(a, _mm256_set1_epi32(i));
            const __m256i index = _mm256_sub_epi32(lanes, _mm256_set1_epi32(i));  // lane t: b[t - i]
            const __m256i rotated = _mm256_permutevar8x32_epi32(b, index);
            const __m256i below = _mm256_cmpgt_epi32(_mm256_set1_epi32(i), lanes);  // t < i: high half
            const __m256i low = _mm256_andnot_si256(below, rotated), high = _mm256_and_si256(below, rotated);
            lo_even = _mm256_add_epi64(lo_even, _mm256_mul_epu32(x, low));
            lo_odd = _mm256_add_epi64(lo_odd, _mm256_mul_epu32(x, _mm256_srli_epi64(low, 32)));
            hi_even = _mm256_add_epi64(hi_even, _mm256_mul_epu32(x, high));
            hi_odd = _mm256_add_epi64(hi_odd, _mm256_mul_epu32(x, _mm256_srli_epi64(high, 32)));
        }
        const __m256i lo = Field::pack(F.reduce(F.fold(lo_even)), F.reduce(F.fold(lo_odd)));
        const __m256i hi = Field::pack(F.reduce(F.fold(hi_even)), F.reduce(F.fold(hi_odd)));
        // c[m + j] is lane m + j of lo, or lane m + j - 8 of hi.
        const __m256i in_low = _mm256_cmpgt_epi32(_mm256_set1_epi32(m), lanes);
        const __m256i c = _mm256_and_si256(lo, in_low);
        __m256i even = _mm256_slli_epi64(c, 32), odd = _mm256_and_si256(c, _mm256_set1_epi64x(~0xFFFFFFFFll));
        for (int j = 0; j < m - 1; ++j) {
            const int t = m + j;
            const __m256i x = _mm256_permutevar8x32_epi32(t < 8 ? lo : hi, _mm256_set1_epi32(t & 7));
            const __m256i row = load(M.table.data() + j * M.stride);
            even = _mm256_add_epi64(even, _mm256_mul_epu32(x, row));
            odd = _mm256_add_epi64(odd, _mm256_mul_epu32(x, _mm256_srli_epi64(row, 32)));
        }
        return Field::pack(F.reduce(F.fold(even)), F.reduce(F.fold(odd)));
    }

    // (x + a) u mod g = shifted u + a u - u[m - 1] g.
    __m256i times_linear_small(__m256i u, u32 a, const Modulus& M) const {
        const int m = M.m;
        const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i shifted = _mm256_and_si256(_mm256_permutevar8x32_epi32(u, _mm256_sub_epi32(lanes, _mm256_set1_epi32(1))),
                                                 _mm256_cmpgt_epi32(lanes, _mm256_setzero_si256()));
        const __m256i top = _mm256_permutevar8x32_epi32(u, _mm256_set1_epi32(m - 1));
        const __m256i g = load(M.g.c), scale = _mm256_set1_epi32(int(a));
        const __m256i minus_top = _mm256_sub_epi32(_mm256_set1_epi32(int(F.p)), top);  // p - top, top < p
        const __m256i one = _mm256_set1_epi32(int(F.one));
        auto part = [&](__m256i s, __m256i v, __m256i h) {  // s one + v a + h (p - top): below 3 p^2
            return _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(s, one), _mm256_mul_epu32(v, scale)),
                                    _mm256_mul_epu32(h, minus_top));
        };
        const __m256i even = part(shifted, u, g);
        const __m256i odd = part(_mm256_srli_epi64(shifted, 32), _mm256_srli_epi64(u, 32), _mm256_srli_epi64(g, 32));
        const __m256i r = Field::pack(F.reduce(F.fold(even)), F.reduce(F.fold(odd)));
        return _mm256_and_si256(r, _mm256_cmpgt_epi32(_mm256_set1_epi32(m), lanes));  // lane m: top - top g_m
    }

    Poly to_poly(__m256i x, int m) const {
        Poly r;
        store(r.c, x);
        r.n = m;
        r.trim();
        return r;
    }

    // x^e mod g, e >= 1, deg g >= 2.
    Poly power_of_x(u64 e, const Modulus& M) const {
        Poly r = linear(0);
        for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
            r = multiply_mod(r, r, M);
            if (e >> bit & 1) r = times_x(r, M);
        }
        return r;
    }

    // (x + a)^e mod g, e >= 1, deg g >= 2.
    Poly power_of_linear(u32 a, u64 e, const Modulus& M) const {
        const Poly base = linear(F.neg(a));
        if (M.m <= 8) {
            __m256i r = load(base.c);
            for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
                r = multiply_small(r, r, M);
                if (e >> bit & 1) r = times_linear_small(r, a, M);
            }
            return to_poly(r, M.m);
        }
        Poly r = base;
        for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
            r = multiply_mod(r, r, M);
            if (e >> bit & 1) {
                Poly s = times_x(r, M);
                lincomb(F, s.c, s.c, F.one, r.c, a, M.m);
                s.n = M.m;
                s.trim();
                r = s;
            }
        }
        return r;
    }

    // u^e mod g, e >= 1, deg u < deg g.
    Poly power(const Poly& u, u64 e, const Modulus& M) const {
        if (M.m <= 8) {
            const __m256i base = load(u.c);
            __m256i r = base;
            for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
                r = multiply_small(r, r, M);
                if (e >> bit & 1) r = multiply_small(r, base, M);
            }
            return to_poly(r, M.m);
        }
        Poly r = u;
        for (int bit = 62 - __builtin_clzll(e); bit >= 0; --bit) {
            r = multiply_mod(r, r, M);
            if (e >> bit & 1) r = multiply_mod(r, u, M);
        }
        return r;
    }

    Poly derivative(const Poly& f) const {
        Poly d;
        for (int i = 1; i < f.n; ++i) d.c[i - 1] = F.mul(f.c[i], F.to(u32(i % F.p)));
        d.n = std::max(f.n - 1, 0);
        d.trim();
        return d;
    }

    // g with g(x^p) = f, for f' = 0 (then f = g^p, as a^p = a in F_p).
    Poly pth_root(const Poly& f) const {
        Poly g;
        g.n = (f.n - 1) / int(F.p) + 1;
        for (int i = 0; i < g.n; ++i) g.c[i] = f.c[i * F.p];
        return g;
    }

    // A square root of a square a (Montgomery) by Cipolla: for t with w = t^2 - a a non-square,
    // (t + y)^((p + 1) / 2) in F_p[y] / (y^2 - w). About 2 log p products in F_p^2, whatever the
    // power of 2 in p - 1 (Tonelli-Shanks takes up to log^2 p for p = 998244353).
    u32 sqrt(u32 a) const {
        const u32 minus_one = F.neg(F.one);
        u32 t = 0, w = F.neg(a);
        while (w != 0 && F.power(w, (F.p - 1) / 2) != minus_one) t = F.add(t, F.one), w = F.sub(F.mul(t, t), a);
        if (w == 0) return t;
        u32 x0 = F.one, x1 = 0, b0 = t, b1 = F.one;  // x = x0 + x1 y, b = b0 + b1 y
        for (u64 e = (F.p + 1) / 2; e; e >>= 1) {
            if (e & 1) {
                const u32 y0 = F.add(F.mul(x0, b0), F.mul(F.mul(x1, b1), w));
                x1 = F.add(F.mul(x0, b1), F.mul(x1, b0)), x0 = y0;
            }
            const u32 c0 = F.add(F.mul(b0, b0), F.mul(F.mul(b1, b1), w));
            b1 = F.mul(F.add(b0, b0), b1), b0 = c0;
        }
        return x0;
    }
};

}  // namespace factor
