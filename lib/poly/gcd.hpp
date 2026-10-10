// Polynomial gcd machinery modulo 998244353: half-gcd jumps and the inverse modulo a polynomial.
// x86-64 with AVX2. Design: lib/poly/notes.md (Gcd).
//
//   poly::Arena arena(poly::inverse_mod_words(n, m));
//   const std::ptrdiff_t t = poly::inverse_mod(arena, f, g, h);
//       // f: n >= 1 coefficients, g: m >= 1 with g[m - 1] != 0, h: m - 1 words. h = 1 / f mod g,
//       // deg h < deg g, its t coefficients trimmed (t = 0: h = 0); t = -1 if gcd(f, g) != 1.
//
// A jump of k on (a, b), deg a = n > deg b, is the matrix R = Q_h ... Q_1 of the Euclidean
// quotients q_i of (a, b) with deg r_h >= n - k > deg r_(h+1): R (a, b) = (r_h, r_(h+1)). It
// depends only on a and b divided by x^(n - 2k) (Thull and Yap). Here remainders are scaled
// (fraction-free steps), so R (a, b) = (c r_h, c' r_(h+1)) for nonzero constants c, c'.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/calculus.hpp"
#include "lib/poly/division.hpp"
#include "lib/poly/product_tree.hpp"
#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

inline std::uint32_t mod_mul(std::uint32_t x, std::uint32_t y) { return std::uint32_t(std::uint64_t(x) * y % kP); }
inline std::uint32_t mod_sub(std::uint32_t x, std::uint32_t y) { return x >= y ? x - y : x + kP - y; }
inline std::uint32_t mod_negate(std::uint32_t x) { return x ? kP - x : 0; }
inline std::uint32_t mod_inverse(std::uint32_t x) { return ntt::detail::power(x, kP - 2); }

// out[j] = (x u[j] + y v[j - 1] + z v[j]) / 2^32 mod P for j in [from, to), rounded out to whole
// vectors: from down and to up to multiples of 8 (callers keep 8 words before u, v and out, and 8
// after). x, y, z < P; canonical u, v give canonical out: the 64-bit sums of 3 products stay below
// 2^32 P, one Montgomery reduction each.
inline void combine3(const std::uint32_t* u, const std::uint32_t* v, std::uint32_t* out, std::ptrdiff_t from,
                     std::ptrdiff_t to, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    const Vec xs = broadcast(x), ys = broadcast(y), zs = broadcast(z);
    for (std::ptrdiff_t j = from & ~std::ptrdiff_t(7); j < to; j += 8) {
        const Vec a = load_unaligned(u + j), b = load_unaligned(v + j - 1), c = load_unaligned(v + j);
        const Vec even = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(a, xs), _mm256_mul_epu32(b, ys)),
                                          _mm256_mul_epu32(c, zs));
        const Vec odd = _mm256_add_epi64(
            _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), xs), _mm256_mul_epu32(_mm256_srli_epi64(b, 32), ys)),
            _mm256_mul_epu32(_mm256_srli_epi64(c, 32), zs));
        store_unaligned(out + j, reduce(montgomery_reduce(even, odd), kP));
    }
}

// The windows of the leaves of transforms x0, x1 in group g (leaves 4g .. 4g + 3).
[[gnu::always_inline]] inline void fill_pair(const std::uint32_t* roots, const std::uint32_t* x0, const std::uint32_t* x1,
                                             std::size_t g, Window (&window)[2][4]) {
    const std::size_t at = 32 * g;
    const Vec a0[4] = {load(x0 + at), load(x0 + at + 8), load(x0 + at + 16), load(x0 + at + 24)};
    const Vec a1[4] = {load(x1 + at), load(x1 + at + 8), load(x1 + at + 16), load(x1 + at + 24)};
    fill_windows(window[0], a0, roots, g);
    fill_windows(window[1], a1, roots, g);
}

// out[k] = x0 y[k][0] + x1 y[k][1] for k < K, transforms of length n (leaf products), canonical.
// The windows of group g + 1 are written before the products of group g read those of group g.
template <std::size_t K>
inline void products(const std::uint32_t* roots, const std::uint32_t* x0, const std::uint32_t* x1,
                     const std::uint32_t* const (&y_in)[K][2], std::uint32_t* const (&out_in)[K], std::size_t n) {
    const std::uint32_t* y[K][2];
    std::uint32_t* out[K];
    for (std::size_t k = 0; k < K; ++k) y[k][0] = y_in[k][0], y[k][1] = y_in[k][1], out[k] = out_in[k];
    const Factor undo_montgomery(kR);
    Window window[2][2][4];
    fill_pair(roots, x0, x1, 0, window[0]);
    for (std::size_t g = 0; g < n / 32; ++g) {
        if (g + 1 < n / 32) fill_pair(roots, x0, x1, g + 1, window[(g + 1) & 1]);
        const Window(&w)[2][4] = window[g & 1];
        const std::size_t at = 32 * g;
#pragma GCC unroll 4
        for (std::size_t t = 0; t < 4; ++t)
            for (std::size_t k = 0; k < K; ++k) {
                const Vec s = low(add(leaf_product(w[0][t], y[k][0] + at + 8 * t), leaf_product(w[1][t], y[k][1] + at + 8 * t)));
                store(out[k] + at + 8 * t, reduce(times(s, undo_montgomery), kP));
            }
    }
}

// A 2 x 2 matrix of polynomials: entry[2 i + j] is row i, column j, with size[2 i + j]
// coefficients (0 for the zero polynomial). progress: the degree of entry[3], n - deg r_h for a
// jump. transform[i]: room for transforms of length up to room; length[i] > 0 when it holds the
// transform of entry i mod (x^length[i] - 1). at0[i]: coefficient 0 of entry i, also for the
// entries a jump leaves out if their row has an entry it computes.
struct Matrix {
    std::uint32_t* entry[4];
    std::size_t size[4];
    std::uint32_t at0[4];
    std::uint32_t* transform[4];
    std::size_t length[4];
    std::size_t room, progress;
};

// Jumps of (a, b) for deg a <= n_max.
class HalfGcd {
public:
    // Jumps of at most this many degrees run Euclid's algorithm directly.
    static constexpr std::ptrdiff_t kDirect = 64;

    static int log_length(std::size_t n_max) {
        return std::max(Transform::kMinLog, int(std::bit_width(std::max<std::size_t>(n_max, 2) - 1)));
    }

    static std::size_t scratch_words(std::size_t n_max) { return 48 * (std::size_t(1) << log_length(n_max)) + 4096; }

    // Arena words for n_max: tables and scratch.
    static std::size_t words(std::size_t n_max) {
        return Transform::words(log_length(n_max)) + Arena::footprint(scratch_words(n_max));
    }

    // direct: jumps of at most this many degrees run Euclid's algorithm (tests force others).
    HalfGcd(Arena& arena, std::size_t n_max, std::ptrdiff_t direct = kDirect)
        : transform_(arena, log_length(n_max)), stack_(arena.take(scratch_words(n_max))), direct_(direct) {}

    // Room for a matrix of progress up to k, with transforms of length up to room.
    Matrix matrix(std::size_t k, std::size_t room) {
        Matrix r;
        for (int i = 0; i < 4; ++i) {
            r.entry[i] = take(k + 1).data();
            r.transform[i] = room ? take(room).data() : nullptr;
            r.length[i] = 0;
        }
        r.room = room;
        return r;
    }

    // out = the jump of k on (a, b): a has n + 1 coefficients, a[n] != 0; b has m + 1, m < n
    // (m = -1: b = 0), b[m] != 0; 0 <= k <= n. out: room for progress k. Only the entries in
    // the mask need (bit i: entry i) are computed; the others are left unspecified.
    void jump(const std::uint32_t* a, std::ptrdiff_t n, const std::uint32_t* b, std::ptrdiff_t m, std::ptrdiff_t k,
              Matrix& out, unsigned need = 0xF) {
        const std::ptrdiff_t target = n - k;
        if (m < target) return identity(out);
        if (n > 2 * k) {  // only a and b divided by x^(n - 2k) matter
            const std::ptrdiff_t s = n - 2 * k;
            return jump(a + s, n - s, b + s, m - s, k, out, need);
        }
        if (k <= direct_) return euclid(a, n, b, m, k, out);
        const std::size_t mark = stack_.mark(), room = std::max<std::size_t>(64, std::bit_ceil(std::size_t(k)));
        const std::ptrdiff_t k1 = first_jump(k), s1 = std::max<std::ptrdiff_t>(0, n - 2 * k1);
        Matrix r1 = matrix(std::size_t(k1), room);
        jump(a + s1, n - s1, b + s1, m - s1, k1, r1);
        std::ptrdiff_t c_deg = 0;
        const std::uint32_t *c = nullptr, *d = nullptr;
        if (r1.progress == 0) {
            r1 = matrix(std::size_t(n - m), room);
            std::uint32_t* const rest = take(std::size_t(n + 1)).data();
            divide_step(a, n, b, m, r1, rest);
            c = b, d = rest, c_deg = m;
        } else {
            c_deg = n - std::ptrdiff_t(r1.progress);
            const std::ptrdiff_t s2 = std::max<std::ptrdiff_t>(0, 2 * target - c_deg);
            std::uint32_t *const cs = take(std::size_t(c_deg + 1)).data(), *const ds = take(std::size_t(c_deg + 1)).data();
            apply(a, n, b, m, r1, c_deg, s2, cs, ds);
            c = cs, d = ds;
        }
        std::ptrdiff_t d_deg = c_deg - 1;
        while (d_deg >= target && d[d_deg] == 0) --d_deg;
        if (d_deg < target) {
            copy(r1, out);
            return stack_.release(mark);
        }
        const std::ptrdiff_t k2 = c_deg - target, s2 = std::max<std::ptrdiff_t>(0, c_deg - 2 * k2);
        Matrix r2 = matrix(std::size_t(k2), room);
        jump(c + s2, c_deg - s2, d + s2, d_deg - s2, k2, r2, (need & 3 ? 3u : 0u) | (need & 12 ? 12u : 0u));
        combine(r2, r1, out, need);
        stack_.release(mark);
    }

private:
    // The first part of a jump of k: half of a power of two, else its largest power of two
    // below k, so that the transform lengths fit (lib/poly/notes.md, Gcd).
    static std::ptrdiff_t first_jump(std::ptrdiff_t k) {
        const std::size_t u = std::size_t(k);
        return std::ptrdiff_t(std::has_single_bit(u) ? u / 2 : std::bit_floor(u));
    }

    std::span<std::uint32_t> take(std::size_t n) { return {stack_.take(n), n}; }

    // n words with 8 zero words before them and 8 after (for combine3).
    std::uint32_t* take_padded(std::size_t n) {
        std::uint32_t* const p = stack_.take(n + 16);
        std::fill_n(p, 8, 0);
        std::fill_n(p + 8 + n, 8, 0);
        return p + 8;
    }

    static void identity(Matrix& out) {
        out.entry[0][0] = out.entry[3][0] = 1;
        out.size[0] = out.size[3] = 1;
        out.size[1] = out.size[2] = 0;
        out.at0[0] = out.at0[3] = 1, out.at0[1] = out.at0[2] = 0;
        for (auto& l : out.length) l = 0;
        out.progress = 0;
    }

    static void copy(const Matrix& from, Matrix& to) {
        for (int i = 0; i < 4; ++i) {
            std::copy_n(from.entry[i], from.size[i], to.entry[i]);
            to.size[i] = from.size[i];
            to.at0[i] = from.at0[i];
            to.length[i] = 0;
        }
        to.progress = from.progress;
    }

    // Euclid's algorithm with fraction-free steps on the coefficients of degree >= the floor
    // n - 2k + (progress so far), which are all the remaining quotients read.
    void euclid(const std::uint32_t* a, std::ptrdiff_t n, const std::uint32_t* b, std::ptrdiff_t m, std::ptrdiff_t k,
                Matrix& out) {
        const std::size_t mark = stack_.mark();
        const std::ptrdiff_t target = n - k;
        std::ptrdiff_t floor = n - 2 * k, lo = std::max<std::ptrdiff_t>(0, floor);
        std::uint32_t* p[3];
        for (auto& x : p) x = take_padded(std::size_t(n + 1));
        std::copy(a + lo, a + n + 1, p[0] + lo);
        std::copy(b + lo, b + m + 1, p[1] + lo);
        // Rows of the matrix: row[0] = (u, v) with u a + v b = c, row[1] for d, row[2] free. Zero
        // past their sizes.
        const std::size_t cap = std::size_t(k) + 8;
        std::uint32_t* row[3][2];
        std::size_t row_size[3][2] = {{1, 0}, {0, 1}, {0, 0}};
        for (auto& r : row)
            for (auto& e : r) {
                e = take_padded(cap);
                std::fill_n(e, cap, 0);
            }
        row[0][0][0] = 1, row[1][1][0] = 1;
        std::uint32_t *c = p[0], *d = p[1], *e = p[2];
        std::ptrdiff_t c_deg = n, d_deg = m;
        while (d_deg >= target) {
            const std::ptrdiff_t delta = c_deg - d_deg;
            const std::uint32_t alpha = d[d_deg];
            floor += delta;
            lo = std::max<std::ptrdiff_t>(0, floor);
            if (delta == 1) {
                // e = alpha^2 c - (alpha beta x + gamma) d kills the coefficients c_deg and d_deg.
                const std::uint32_t beta = c[c_deg];
                const std::uint32_t gamma = mod_sub(mod_mul(alpha, c[c_deg - 1]), mod_mul(beta, d[d_deg - 1]));
                // Factors times 2^32 for combine3's Montgomery reduction.
                const std::uint32_t alpha_r = mod_mul(alpha, kR);
                const std::uint32_t x = mod_mul(alpha, alpha_r), y = kP - mod_mul(beta, alpha_r), z = mod_negate(mod_mul(gamma, kR));
                combine3(c, d, e, lo, d_deg, x, y, z);
                for (int i = 0; i < 2; ++i) {
                    const std::size_t sc = row_size[0][i], sd = row_size[1][i];
                    const std::size_t se = std::max(sc, sd ? sd + 1 : 0);
                    combine3(row[0][i], row[1][i], row[2][i], 0, std::ptrdiff_t(se), x, y, z);
                    if (row_size[2][i] > se) std::fill(row[2][i] + se, row[2][i] + row_size[2][i], 0);
                    row_size[2][i] = se;
                }
            } else {
                pseudo_divide(c, c_deg, d, d_deg, e, lo, alpha, row, row_size);
            }
            std::uint32_t* const old = c;
            c = d, d = e, e = old;
            for (int i = 0; i < 2; ++i) {
                std::swap(row[0][i], row[1][i]);
                std::swap(row[1][i], row[2][i]);
                std::swap(row_size[0][i], row_size[1][i]);
                std::swap(row_size[1][i], row_size[2][i]);
            }
            c_deg = d_deg;
            d_deg = c_deg - 1;
            while (d_deg >= target && d[d_deg] == 0) --d_deg;
        }
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                std::copy_n(row[i][j], row_size[i][j], out.entry[2 * i + j]);
                out.size[2 * i + j] = row_size[i][j];
                out.at0[2 * i + j] = row_size[i][j] ? row[i][j][0] : 0;
                out.length[2 * i + j] = 0;
            }
        out.progress = std::size_t(n - c_deg);
        stack_.release(mark);
    }

    // A quotient of degree delta > 1: e = c, then e = alpha e - e[d_deg + t] x^t d for t = delta
    // .. 0, on the coefficients from lo; the rows alike (row[2] from row[0] and row[1]).
    static void pseudo_divide(const std::uint32_t* c, std::ptrdiff_t c_deg, const std::uint32_t* d, std::ptrdiff_t d_deg,
                              std::uint32_t* e, std::ptrdiff_t lo, std::uint32_t alpha, std::uint32_t* (&row)[3][2],
                              std::size_t (&row_size)[3][2]) {
        std::copy(c + lo, c + c_deg + 1, e + lo);
        for (int i = 0; i < 2; ++i) {
            std::copy_n(row[0][i], row_size[0][i], row[2][i]);
            if (row_size[2][i] > row_size[0][i]) std::fill(row[2][i] + row_size[0][i], row[2][i] + row_size[2][i], 0);
            row_size[2][i] = row_size[0][i];
        }
        for (std::ptrdiff_t t = c_deg - d_deg; t >= 0; --t) {
            const std::uint32_t minus_tau = mod_negate(e[d_deg + t]);
            for (std::ptrdiff_t j = lo; j <= d_deg + t; ++j) {
                const std::uint32_t dj = j - t >= 0 ? d[j - t] : 0;
                e[j] = std::uint32_t((std::uint64_t(alpha) * e[j] + std::uint64_t(minus_tau) * dj) % kP);
            }
            for (int i = 0; i < 2; ++i) {
                std::uint32_t* const re = row[2][i];
                const std::uint32_t* const rd = row[1][i];
                const std::size_t sd = row_size[1][i], shift = std::size_t(t);
                const std::size_t se = std::max(row_size[2][i], sd ? sd + shift : 0);
                for (std::size_t j = 0; j < se; ++j) {
                    const std::uint32_t dj = j >= shift ? rd[j - shift] : 0;  // zero past sd
                    re[j] = std::uint32_t((std::uint64_t(alpha) * re[j] + std::uint64_t(minus_tau) * dj) % kP);
                }
                row_size[2][i] = se;
            }
        }
    }

    // (c, d) = (b, a - q b) for q = a quo b, and r = [[0, 1], [1, -q]]. rest: n + 1 words.
    // Long division for short quotients or divisors, else poly::divide in an arena of its own.
    void divide_step(const std::uint32_t* a, std::ptrdiff_t n, const std::uint32_t* b, std::ptrdiff_t m, Matrix& r,
                     std::uint32_t* rest) {
        const std::size_t delta = std::size_t(n - m);
        std::uint32_t* q = r.entry[3];
        if ((delta + 1) * std::size_t(m + 1) <= kLongDivision) {
            std::copy(a, a + n + 1, rest);
            const std::uint32_t lead = mod_inverse(b[m]);
            for (std::size_t i = delta + 1; i-- > 0;) {
                const std::uint32_t qi = mod_mul(rest[std::size_t(m) + i], lead);
                q[i] = qi;
                for (std::ptrdiff_t t = 0; t <= m; ++t)
                    rest[i + std::size_t(t)] = mod_sub(rest[i + std::size_t(t)], mod_mul(qi, b[t]));
            }
        } else {
            const std::span<const std::uint32_t> f(a, std::size_t(n + 1)), g(b, std::size_t(m + 1));
            Arena arena(divide_words(f.size(), g.size()));
            divide(arena, f, g, std::span<std::uint32_t>(q, delta + 1), std::span<std::uint32_t>(rest, std::size_t(m)));
            std::fill(rest + m, rest + n + 1, 0);
        }
        for (std::size_t i = 0; i <= delta; ++i) q[i] = mod_negate(q[i]);
        r.entry[1][0] = r.entry[2][0] = 1;
        r.size[0] = 0, r.size[1] = r.size[2] = 1, r.size[3] = delta + 1;
        r.at0[0] = 0, r.at0[1] = r.at0[2] = 1, r.at0[3] = q[0];
        for (auto& l : r.length) l = 0;
        r.progress = delta;
    }

    // out = the transform of length out.size() of c mod (x^L - 1), c of size coefficients.
    void transform_of(const std::uint32_t* c, std::size_t size, std::span<std::uint32_t> out) const {
        const std::size_t len = out.size();
        if (size <= len) return transform_.forward(std::span<const std::uint32_t>(c, size), 0, out);
        fold(std::span<const std::uint32_t>(c, size), out);
        transform_.forward(out);
    }

    // r.transform[i] = the transform of length len of entry i (room >= len): kept, doubled from
    // the half held (forward_upper of the entry mod x^(len/2) + 1), or computed.
    void extend(Matrix& r, int i, std::size_t len) const {
        if (r.length[i] == len) return;
        const std::span<std::uint32_t> t(r.transform[i], len);
        const std::size_t half = len / 2, size = r.size[i];
        if (r.length[i] == half && size <= half + 1) {
            const std::span<std::uint32_t> upper = t.subspan(half);
            if (size <= half) {
                transform_.forward_upper(std::span<const std::uint32_t>(r.entry[i], size), 0, upper);
            } else {
                std::copy_n(r.entry[i], half, upper.data());
                upper[0] = mod_sub(upper[0], r.entry[i][half]);
                transform_.forward_upper(upper, 0, upper);
            }
        } else {
            transform_of(r.entry[i], size, t);
        }
        r.length[i] = len;
    }

    // c, d = r1 (a, b) at degrees [s2, c_deg], c_deg = deg c. With L >= n / 2, the products are
    // cyclic of length L; when c_deg >= L, the coefficients [0, c_deg - L] of c and d (which the
    // wrap adds to [L, c_deg]) come from the products of r1 with a, b mod x^(c_deg - L + 1).
    void apply(const std::uint32_t* a, std::ptrdiff_t n, const std::uint32_t* b, std::ptrdiff_t m, Matrix& r1,
               std::ptrdiff_t c_deg, std::ptrdiff_t s2, std::uint32_t* c, std::uint32_t* d) {
        const std::size_t mark = stack_.mark();
        const std::size_t len = std::max<std::size_t>(64, std::bit_ceil(std::size_t(n + 1) / 2));
        for (int i = 0; i < 4; ++i) extend(r1, i, len);
        const std::span<std::uint32_t> ta = take(len), tb = take(len);
        transform_of(a, std::size_t(n + 1), ta);
        transform_of(b, std::size_t(m + 1), tb);
        const std::span<std::uint32_t> pc = take(len), pd = take(len);
        apply_products(r1, ta, tb, pc, pd);
        const std::ptrdiff_t l = std::ptrdiff_t(len), mixed = c_deg - l;  // c[j] + c[j + L] at j <= mixed
        if (mixed < 0) {
            std::copy(pc.begin() + s2, pc.begin() + c_deg + 1, c + s2);
            std::copy(pd.begin() + s2, pd.begin() + c_deg + 1, d + s2);
            return stack_.release(mark);
        }
        const std::size_t low = std::size_t(mixed + 1);
        transform_.forward(std::span<const std::uint32_t>(a, low), 0, ta);
        transform_.forward(std::span<const std::uint32_t>(b, std::min(low, std::size_t(m + 1))), 0, tb);
        const std::span<std::uint32_t> lc = take(len), ld = take(len);
        apply_products(r1, ta, tb, lc, ld);
        // Coefficient L of the low products (if n = 2L) wraps onto 0.
        lc[0] = std::uint32_t((std::uint64_t(r1.at0[0]) * a[0] + std::uint64_t(r1.at0[1]) * b[0]) % kP);
        ld[0] = std::uint32_t((std::uint64_t(r1.at0[2]) * a[0] + std::uint64_t(r1.at0[3]) * b[0]) % kP);
        // [s2, mixed] from the low products, (mixed, L) as computed, [L, c_deg] unwrapped.
        const std::ptrdiff_t first = std::min(s2, mixed + 1);
        std::copy(lc.begin() + first, lc.begin() + mixed + 1, c + first);
        std::copy(ld.begin() + first, ld.begin() + mixed + 1, d + first);
        const std::ptrdiff_t from = std::max(s2, mixed + 1);
        std::copy(pc.begin() + from, pc.begin() + l, c + from);
        std::copy(pd.begin() + from, pd.begin() + l, d + from);
        subtract(pc.data(), lc.data(), c + l, std::size_t(mixed + 1));
        subtract(pd.data(), ld.data(), d + l, std::size_t(mixed + 1));
        stack_.release(mark);
    }

    // c, d = the coefficients of r1 (a, b) mod (x^L - 1) from the transforms ta, tb of length L
    // (r1's transforms of that length).
    void apply_products(const Matrix& r1, std::span<const std::uint32_t> ta, std::span<const std::uint32_t> tb,
                        std::span<std::uint32_t> c, std::span<std::uint32_t> d) const {
        const std::size_t len = c.size();
        const auto t = [&](int i) { return std::span<const std::uint32_t>(r1.transform[i], len); };
        const Transform::Pair pc[2] = {{t(0), ta}, {t(1), tb}}, pd[2] = {{t(2), ta}, {t(3), tb}};
        transform_.inverse_product_sum(pc, c);
        transform_.inverse_product_sum(pd, d);
    }

    // out = r2 r1 (the entries in need), progress p = r2.progress + r1.progress: cyclic products
    // of length L >= p (coefficient L, if p = L, wraps onto 0), whose transforms out keeps if its
    // room allows.
    void combine(Matrix& r2, Matrix& r1, Matrix& out, unsigned need) {
        const std::size_t mark = stack_.mark();
        const std::size_t p = r2.progress + r1.progress, len = std::max<std::size_t>(64, std::bit_ceil(p));
        for (int i = 0; i < 2; ++i)
            if (need & (3u << 2 * i)) extend(r2, 2 * i, len), extend(r2, 2 * i + 1, len);
        for (int j = 0; j < 2; ++j)
            if (need & (5u << j)) extend(r1, j, len), extend(r1, 2 + j, len);
        const std::span<std::uint32_t> buffer = take(len);
        const bool keep = out.room >= len;
        std::uint32_t* t[4];
        for (int i = 0; i < 4; ++i) t[i] = keep ? out.transform[i] : take(len).data();
        for (int i = 0; i < 2; ++i) {
            // Row i: out[2i + j] = r2[2i] r1[j] + r2[2i + 1] r1[2 + j], sharing r2's windows.
            const std::uint32_t *x0 = r2.transform[2 * i], *x1 = r2.transform[2 * i + 1];
            const std::uint32_t* const y[2][2] = {{r1.transform[0], r1.transform[2]}, {r1.transform[1], r1.transform[3]}};
            const unsigned row = need >> 2 * i & 3;
            if (row == 3) {
                std::uint32_t* const to[2] = {t[2 * i], t[2 * i + 1]};
                products<2>(transform_.roots(), x0, x1, y, to, len);
            } else if (row) {
                const int j = row == 1 ? 0 : 1;
                const std::uint32_t* const yj[1][2] = {{y[j][0], y[j][1]}};
                std::uint32_t* const to[1] = {t[2 * i + j]};
                products<1>(transform_.roots(), x0, x1, yj, to, len);
            }
        }
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                const int ij = 2 * i + j;
                out.length[ij] = 0;
                const std::uint32_t x0 = std::uint32_t(
                    (std::uint64_t(r2.at0[2 * i]) * r1.at0[j] + std::uint64_t(r2.at0[2 * i + 1]) * r1.at0[2 + j]) % kP);
                out.at0[ij] = x0;
                if (!(need >> ij & 1)) continue;
                if (keep) out.length[ij] = len;
                transform_.inverse(std::span<const std::uint32_t>(t[ij], len), buffer);
                std::uint32_t* const e = out.entry[ij];
                std::copy_n(buffer.data(), std::min(p + 1, len), e);
                if (p == len) e[len] = mod_sub(e[0], x0), e[0] = x0;
                std::size_t size = p + 1;
                while (size > 0 && e[size - 1] == 0) --size;
                out.size[ij] = size;
            }
        out.progress = p;
        stack_.release(mark);
    }

    Transform transform_;
    Stack stack_;
    std::ptrdiff_t direct_;
};

}  // namespace detail

// Arena words inverse_mod() takes for f of n and g of m coefficients.
inline std::size_t inverse_mod_words(std::size_t n, std::size_t m) {
    const std::size_t d = m - 1;
    return divide_words(n, m) + Arena::footprint(d) + Arena::footprint(quotient_size(n, m)) +
           detail::HalfGcd::words(std::max<std::size_t>(d, 1));
}

// h = 1 / f mod g with deg h < deg g = m - 1 (h.size() >= m - 1); returns the number of
// coefficients of h after trimming, or -1 if gcd(f, g) != 1. f: n >= 1 coefficients; g: m >= 1,
// g[m - 1] != 0; all canonical. arena: inverse_mod_words(n, m) words.
inline std::ptrdiff_t inverse_mod(Arena& arena, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                                  std::span<std::uint32_t> h) {
    using namespace detail;
    const std::ptrdiff_t d = std::ptrdiff_t(g.size()) - 1;
    if (d == 0) return 0;
    const std::span<std::uint32_t> b = arena.take(std::size_t(d));
    const std::size_t rs = remainder_size(f.size(), g.size());
    divide(arena, f, g, arena.take(quotient_size(f.size(), g.size())), b.first(rs));
    std::ptrdiff_t m = std::ptrdiff_t(rs) - 1;
    while (m >= 0 && b[std::size_t(m)] == 0) --m;
    if (m < 0) return -1;
    HalfGcd gcd(arena, std::size_t(d));
    Matrix r = gcd.matrix(std::size_t(d), 0);
    gcd.jump(g.data(), d, b.data(), m, d, r, 0x2);
    if (std::ptrdiff_t(r.progress) != d) return -1;  // deg gcd = d - progress
    // r (g, b) = (c, 0) with c a nonzero constant: h = r[0][1] / c.
    const std::uint32_t c = std::uint32_t((std::uint64_t(r.at0[0]) * g[0] + std::uint64_t(r.at0[1]) * b[0]) % kP);
    const std::uint32_t scale = mod_inverse(c);
    std::size_t t = r.size[1];
    while (t > 0 && r.entry[1][t - 1] == 0) --t;
    for (std::size_t i = 0; i < t; ++i) h[i] = mod_mul(r.entry[1][i], scale);
    return std::ptrdiff_t(t);
}

}  // namespace poly
