// c = a * b over F = F_2[x] / (x^64 + x^4 + x^3 + x + 1), by the additive FFT of Lin, Chung and
// Han (novel polynomial basis) on a Cantor basis beta_0 = 1, beta_{i+1}^2 + beta_{i+1} = beta_i.
//
// With W_i = span(beta_0, ..., beta_{i-1}) and s_i the subspace polynomial of W_i, the novel basis
// is X_j = prod s_i^{j_i} over the bits j_i of j. On a Cantor basis s_i(beta_i) = 1 and
// s_{2^k}(x) = x^{2^{2^k}} + x, so the change from the monomial basis to X takes XORs only
// (Taylor expansions in x^tau + x). The transform evaluates f = sum d_j X_j at omega_k =
// sum_{bits of k} beta_b, k < 2^l; stage i uses the twiddle s_i(omega_c) = omega_{c >> i} for the
// block at c.
#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <bit>

#include "lib/io/bulk64.hpp"
#include "lib/io/io.hpp"

namespace {

using u64 = std::uint64_t;

constexpr int kMaxLog = 20;  // N + M - 1 < 2^20
constexpr int kMinLog = 3;   // the last three stages run in one 8-element kernel

// Field arithmetic. The polynomial's low part: x^64 = x^4 + x^3 + x + 1.

constexpr u64 clmul_low_portable(u64 a, u64 b, u64& high) {
    u64 low = 0;
    high = 0;
    for (int i = 0; i < 64; ++i) {
        if (b >> i & 1) {
            low ^= a << i;
            high ^= i ? a >> (64 - i) : 0;
        }
    }
    return low;
}

// (high, low) mod P: high * x^64 = high * (x^4 + x^3 + x + 1); the bits of that above x^63 are
// g = high >> 60 ^ high >> 61 ^ high >> 63, folded once more. Both folds use q = high ^ g.
constexpr u64 reduce_scalar(u64 low, u64 high) {
    const u64 q = high ^ high >> 60 ^ high >> 61 ^ high >> 63;
    return low ^ q ^ q << 1 ^ q << 3 ^ q << 4;
}

constexpr u64 multiply_scalar(u64 a, u64 b) {
    u64 high = 0;
    const u64 low = clmul_low_portable(a, b, high);
    return reduce_scalar(low, high);
}

// y with y^2 + y = c (c of trace 0). y -> y^2 + y is F_2-linear: eliminate on its images of the
// unit vectors, tracking which unit vectors make up each basis vector.
constexpr u64 solve_artin_schreier(u64 c) {
    std::array<u64, 64> pivot{}, source{};
    for (int j = 0; j < 64; ++j) {
        const u64 unit = u64(1) << j;
        u64 v = multiply_scalar(unit, unit) ^ unit, s = unit;
        for (int b = 63; b >= 0 && v; --b) {
            if (!(v >> b & 1)) continue;
            if (!pivot[b]) {
                pivot[b] = v, source[b] = s;
                v = 0;
            } else {
                v ^= pivot[b], s ^= source[b];
            }
        }
    }
    u64 y = 0;
    for (int b = 63; b >= 0; --b) {
        if (c >> b & 1) c ^= pivot[b], y ^= source[b];
    }
    return c == 0 ? y : throw "no root";
}

constexpr std::array<u64, kMaxLog> kBeta = [] {
    std::array<u64, kMaxLog> beta{1};
    for (int i = 1; i < kMaxLog; ++i) beta[i] = solve_artin_schreier(beta[i - 1]);
    return beta;
}();

// omega_m = kOmegaLow[m & 1023] ^ kOmegaHigh[m >> 10], m < 2^20.
struct OmegaTable {
    std::array<u64, 1024> low, high;
};

constexpr OmegaTable kOmega = [] {
    OmegaTable t{};
    for (int m = 1; m < 1024; ++m) {
        const int b = std::countr_zero(unsigned(m));
        t.low[m] = t.low[m & (m - 1)] ^ kBeta[b];
        t.high[m] = t.high[m & (m - 1)] ^ kBeta[b + 10];
    }
    return t;
}();

inline u64 omega(std::size_t m) { return kOmega.low[m & 1023] ^ kOmega.high[m >> 10]; }

// Four field elements per vector.

using Vec = __m256i;

inline Vec load(const u64* p) { return _mm256_load_si256(reinterpret_cast<const Vec*>(p)); }
inline void store(u64* p, Vec v) { _mm256_store_si256(reinterpret_cast<Vec*>(p), v); }
inline Vec broadcast(u64 x) { return _mm256_set1_epi64x(std::int64_t(x)); }

// (high, low) mod P, as reduce_scalar: low ^ (high * (x^4 + x^3 + x + 1) mod x^64) ^ g(high >> 60),
// where g(n) < 256 folds the bits above x^63; looked up by vpshufb.
inline Vec reduce(Vec low, Vec high) {
    constexpr auto kFold = [] {
        std::array<std::int8_t, 16> t{};
        for (u64 n = 0; n < 16; ++n) t[n] = std::int8_t(reduce_scalar(0, n << 60));
        return t;
    }();
    const Vec fold = _mm256_setr_epi8(kFold[0], kFold[1], kFold[2], kFold[3], kFold[4], kFold[5], kFold[6], kFold[7],
                                      kFold[8], kFold[9], kFold[10], kFold[11], kFold[12], kFold[13], kFold[14],
                                      kFold[15], kFold[0], kFold[1], kFold[2], kFold[3], kFold[4], kFold[5], kFold[6],
                                      kFold[7], kFold[8], kFold[9], kFold[10], kFold[11], kFold[12], kFold[13],
                                      kFold[14], kFold[15]);
    const Vec s = _mm256_xor_si256(high, _mm256_slli_epi64(high, 1));
    const Vec times_r = _mm256_xor_si256(s, _mm256_slli_epi64(s, 3));  // high * (x^4 + x^3 + x + 1) mod x^64
    const Vec top = _mm256_shuffle_epi8(fold, _mm256_srli_epi64(high, 60));
    return _mm256_xor_si256(_mm256_xor_si256(low, times_r), top);
}

// Lane-wise products a * b in F.
inline Vec multiply(Vec a, Vec b) {
#ifdef __VPCLMULQDQ__
    const Vec even = _mm256_clmulepi64_epi128(a, b, 0x00);  // lanes 0 and 2: low, high
    const Vec odd = _mm256_clmulepi64_epi128(a, b, 0x11);   // lanes 1 and 3
    return reduce(_mm256_unpacklo_epi64(even, odd), _mm256_unpackhi_epi64(even, odd));
#elif defined(__PCLMUL__)
    __m128i half[2][2];  // [128-bit half][even, odd]
    for (int h = 0; h < 2; ++h) {
        const __m128i x = h ? _mm256_extracti128_si256(a, 1) : _mm256_castsi256_si128(a);
        const __m128i y = h ? _mm256_extracti128_si256(b, 1) : _mm256_castsi256_si128(b);
        half[h][0] = _mm_clmulepi64_si128(x, y, 0x00);
        half[h][1] = _mm_clmulepi64_si128(x, y, 0x11);
    }
    const Vec even = _mm256_set_m128i(half[1][0], half[0][0]), odd = _mm256_set_m128i(half[1][1], half[0][1]);
    return reduce(_mm256_unpacklo_epi64(even, odd), _mm256_unpackhi_epi64(even, odd));
#else
    alignas(32) u64 x[4], y[4];
    _mm256_store_si256(reinterpret_cast<Vec*>(x), a);
    _mm256_store_si256(reinterpret_cast<Vec*>(y), b);
    for (int k = 0; k < 4; ++k) x[k] = multiply_scalar(x[k], y[k]);
    return _mm256_load_si256(reinterpret_cast<const Vec*>(x));
#endif
}

// Change of basis between monomials and X. A sequence's element e spans the W words at f + e * S
// (W <= S); elements are xored as units. Forward passes take size: elements [size, length) are zero.

template <std::size_t W, std::size_t S>
inline void xor_rows(u64* dst, const u64* src, std::size_t count) {
    if constexpr (W == S) {
        for (std::size_t k = 0; k < count * W; ++k) dst[k] ^= src[k];
    } else {
        for (std::size_t e = 0; e < count; ++e) {
            for (std::size_t k = 0; k < W; ++k) dst[e * S + k] ^= src[e * S + k];
        }
    }
}

// The Taylor expansion below for rows <= 16 long rows, in one pass. With y = x^Tau,
// f = sum_r f_r(x) y^r; substituting y = t + x gives sum_k H_k(x) t^k, H_k = sum_r C(r, k) x^(r-k) f_r
// (a Taylor shift by x, its own inverse), deg H_k < Tau + 15. Split H_k = lo_k + x^Tau hi_k:
// x^Tau = t + x, so g_k = lo_k + x hi_k + hi_(k-1). The inverse shifts g the same way and
// f_r = lo_r + hi_(r-1). The shift moves data at most 15 columns right, so it runs on column
// blocks from right to left, each with the 16 still unchanged columns before it.
namespace wide {

constexpr std::size_t kHalo = 16, kBlock = 128, kWidth = kHalo + kBlock;

// The shift on buf's rows; exact for columns j >= kHalo. A level with distance d needs columns
// >= kHalo - (d - 1) of the previous one, so columns below 4 are never needed after d = 4.
template <std::size_t Rows>
void shift(u64 (*buf)[kWidth]) {
#pragma GCC unroll 4
    for (std::size_t d = Rows / 2; d >= 1; d /= 2) {
#pragma GCC unroll 16
        for (std::size_t r = 0; r < Rows; ++r) {
            if (r & d) continue;  // r in the lower half of its block of 2d rows
            for (std::size_t j = std::max<std::size_t>(d, 4); j < kWidth; j += 4) {
                const Vec moved = _mm256_loadu_si256(reinterpret_cast<const Vec*>(buf[r + d] + j - d));
                store(buf[r] + j, _mm256_xor_si256(load(buf[r] + j), moved));
            }
        }
    }
}

// H for columns [c, c + width) of f's rows (each Tau words), zero past Tau, into buf[r][kHalo, ...).
template <std::size_t Tau, std::size_t Rows>
void block(const u64* f, std::size_t c, std::size_t width, u64 (*buf)[kWidth]) {
    const bool inside = c >= kHalo && width == kBlock;  // c + kBlock <= Tau
    for (std::size_t r = 0; r < Rows; ++r) {
        if (inside) {
            std::memcpy(buf[r], f + r * Tau + c - kHalo, kWidth * sizeof(u64));
            continue;
        }
        for (std::size_t j = 0; j < kWidth; ++j) {
            const std::size_t col = c + j - kHalo;  // wraps below 0
            buf[r][j] = c + j >= kHalo && col < Tau && j < kHalo + width ? f[r * Tau + col] : 0;
        }
    }
    shift<Rows>(buf);
}

template <std::size_t Tau, std::size_t Rows, bool Inverse>
void taylor(u64* f) {
    static_assert(Tau % kBlock == 0);
    alignas(32) static u64 buf[Rows][kWidth];
    alignas(32) static u64 high[Rows][kHalo];
    block<Tau, Rows>(f, Tau, 0, buf);
    for (std::size_t r = 0; r < Rows; ++r) std::memcpy(high[r], buf[r] + kHalo, sizeof(high[r]));
    for (std::size_t c = Tau; c > 0;) {
        c -= kBlock;
        block<Tau, Rows>(f, c, kBlock, buf);
        for (std::size_t r = 0; r < Rows; ++r) std::memcpy(f + r * Tau + c, buf[r] + kHalo, kBlock * sizeof(u64));
    }
    for (std::size_t r = 1; r < Rows; ++r) {
        for (std::size_t e = 0; e < kHalo; ++e) f[r * Tau + e] ^= high[r - 1][e];
    }
    if constexpr (!Inverse) {
        for (std::size_t r = 0; r < Rows; ++r) {
            for (std::size_t e = 0; e + 1 < kHalo; ++e) f[r * Tau + e + 1] ^= high[r][e];
        }
    }
}

template <std::size_t Tau, bool Inverse>
void taylor(u64* f, std::size_t rows) {
    switch (rows) {
    case 2: return taylor<Tau, 2, Inverse>(f);
    case 4: return taylor<Tau, 4, Inverse>(f);
    case 8: return taylor<Tau, 8, Inverse>(f);
    case 16: return taylor<Tau, 16, Inverse>(f);
    }
}

}  // namespace wide

// f[0, Len) = sum_m g_m(x) t^m with t = x^Tau + x, deg g_m < Tau: g_m in f[m * Tau, (m + 1) * Tau).
// With A, B the halves and d = Len / 2 / Tau: x^(Len/2) = t^d + x^d, so
// f = (A + x^d B_low + x^d B_high) + t^d (B + B_high), B_high the top d entries of B.
template <std::size_t Len, std::size_t Tau, std::size_t W, std::size_t S, bool Inverse>
void taylor(u64* f, std::size_t size) {
    if constexpr (W == 1 && S == 1 && Tau >= 4096 && Len / Tau <= 16) {
        const std::size_t rows = Inverse ? Len / Tau : std::bit_ceil((size + Tau - 1) / Tau);
        if (rows > 1) wide::taylor<Tau, Inverse>(f, rows);
    } else if constexpr (Len > Tau) {
        constexpr std::size_t half = Len / 2, d = half / Tau;
        u64* a = f;
        u64* b = f + half * S;
        if constexpr (Inverse) {
            taylor<half, Tau, W, S, true>(a, half);
            taylor<half, Tau, W, S, true>(b, half);
            xor_rows<W, S>(b, b + (half - d) * S, d);
            xor_rows<W, S>(a + d * S, b, half - d);
            xor_rows<W, S>(a + d * S, b + (half - d) * S, d);
        } else {
            if (size <= half) return taylor<half, Tau, W, S, false>(a, size);
            xor_rows<W, S>(a + d * S, b, half - d);
            xor_rows<W, S>(a + d * S, b + (half - d) * S, d);
            xor_rows<W, S>(b, b + (half - d) * S, d);
            taylor<half, Tau, W, S, false>(a, half);
            taylor<half, Tau, W, S, false>(b, size - half);
        }
    }
}

constexpr std::size_t kColumnWords = 8192;  // column passes run on blocks of at most 64 KiB

template <int L, std::size_t W, std::size_t S, bool Inverse>
void change_basis(u64* f, std::size_t size);

// The column step of change_basis<L>: rows [0, used) of 2^(L-K) rows, each Tau elements.
// Contiguous rows are cut into column blocks that fit L1/L2; all column stages run per block.
template <int L, int K, std::size_t W, std::size_t S, bool Inverse>
void change_columns(u64* f, std::size_t used) {
    constexpr std::size_t tau = std::size_t(1) << K;
    if constexpr (W == S) {
        constexpr std::size_t width = tau * W, rows = std::size_t(1) << (L - K);
        constexpr std::size_t chunk = std::min(width, std::max<std::size_t>(kColumnWords / rows, 4));
        for (std::size_t p = 0; p < width; p += chunk) change_basis<L - K, chunk, width, Inverse>(f + p, used);
    } else {
        for (std::size_t p = 0; p < tau; ++p) change_basis<L - K, W, tau * S, Inverse>(f + p * S, used);
    }
}

// x[0, 4) to [x0[k], x1[k], x2[k], x3[k]] for k < 4.
inline void transpose4(Vec& x0, Vec& x1, Vec& x2, Vec& x3) {
    const Vec t0 = _mm256_unpacklo_epi64(x0, x1), t1 = _mm256_unpackhi_epi64(x0, x1);
    const Vec t2 = _mm256_unpacklo_epi64(x2, x3), t3 = _mm256_unpackhi_epi64(x2, x3);
    x0 = _mm256_permute2x128_si256(t0, t2, 0x20);
    x1 = _mm256_permute2x128_si256(t1, t3, 0x20);
    x2 = _mm256_permute2x128_si256(t0, t2, 0x31);
    x3 = _mm256_permute2x128_si256(t1, t3, 0x31);
}

// Four rows of Tau words, f[r * Tau + k], to buf[4 * k + r], or back.
template <std::size_t Tau, bool Back>
void transpose_rows(u64* f, u64* buf) {
    for (std::size_t k = 0; k < Tau; k += 4) {
        if constexpr (Back) {
            Vec x0 = load(buf + 4 * k), x1 = load(buf + 4 * k + 4), x2 = load(buf + 4 * k + 8), x3 = load(buf + 4 * k + 12);
            transpose4(x0, x1, x2, x3);
            store(f + k, x0), store(f + Tau + k, x1), store(f + 2 * Tau + k, x2), store(f + 3 * Tau + k, x3);
        } else {
            Vec x0 = load(f + k), x1 = load(f + Tau + k), x2 = load(f + 2 * Tau + k), x3 = load(f + 3 * Tau + k);
            transpose4(x0, x1, x2, x3);
            store(buf + 4 * k, x0), store(buf + 4 * k + 4, x1), store(buf + 4 * k + 8, x2), store(buf + 4 * k + 12, x3);
        }
    }
}

// The row step of change_basis: change_basis<K> on rows [0, used) of Rows rows of 2^K elements.
// Rows of single words go four at a time, transposed, so that every XOR moves a whole vector.
template <int K, std::size_t Rows, std::size_t W, std::size_t S, bool Inverse>
void change_rows(u64* f, std::size_t used) {
    constexpr std::size_t tau = std::size_t(1) << K;
    if constexpr (W == 1 && S == 1 && K >= 2 && K <= 8 && Rows >= 4) {
        alignas(32) static u64 buf[4 * tau];
        for (std::size_t m = 0; m < used; m += 4) {
            transpose_rows<tau, false>(f + m * tau, buf);
            change_basis<K, 4, 4, Inverse>(buf, tau);
            transpose_rows<tau, true>(f + m * tau, buf);
        }
    } else {
        for (std::size_t m = 0; m < used; ++m) change_basis<K, W, S, Inverse>(f + m * tau * S, tau);
    }
}

// Monomial coefficients of f[0, 2^L) to X coefficients (or back). With K = 2^k the largest power
// of two below L: expand in t = s_K = x^(2^K) + x, convert each row g_m (length 2^K), then each
// column (a polynomial in t, whose basis s_i(t) = s_{K + i}(x) for i < L - K <= K).
template <int L, std::size_t W, std::size_t S, bool Inverse>
void change_basis(u64* f, std::size_t size) {
    if constexpr (L >= 2) {
        constexpr int K = 1 << (std::bit_width(unsigned(L - 1)) - 1);
        constexpr std::size_t tau = std::size_t(1) << K, rows = std::size_t(1) << (L - K);
        if constexpr (Inverse) {
            change_columns<L, K, W, S, true>(f, rows);
            change_rows<K, rows, W, S, true>(f, rows);
            taylor<(std::size_t(1) << L), tau, W, S, true>(f, std::size_t(1) << L);
        } else {
            taylor<(std::size_t(1) << L), tau, W, S, false>(f, size);
            const std::size_t used = (size + tau - 1) / tau;  // deg f < size: g_m = 0 for m >= used
            change_rows<K, rows, W, S, false>(f, used);
            change_columns<L, K, W, S, false>(f, used);
        }
    }
}

// f[0, 2^l), of which [size, 2^l) is zero for the forward change.
template <bool Inverse, int L = kMaxLog>
void change_basis(u64* f, int l, std::size_t size) {
    if constexpr (L >= kMinLog) {
        if (l == L) return change_basis<L, 1, 1, Inverse>(f, size);
        change_basis<Inverse, L - 1>(f, l, size);
    }
}

// The transform. Stage i maps the halves (u, v) of each block of 2^(i+1) at c to
// (u + w v, u + w v + v), w = omega_{c >> i}; the inverse undoes the stages in reverse order.
// Blocks of 2^kBlockLog words (32 KiB) run stage by stage; above, two stages share a pass.

constexpr int kBlockLog = 12;

inline void stage_forward(u64* u, std::size_t half, u64 w) {
    const Vec tw = broadcast(w);
    u64* v = u + half;
    for (std::size_t j = 0; j < half; j += 4) {
        const Vec x = _mm256_xor_si256(load(u + j), multiply(load(v + j), tw));
        store(u + j, x);
        store(v + j, _mm256_xor_si256(load(v + j), x));
    }
}

inline void stage_inverse(u64* u, std::size_t half, u64 w) {
    const Vec tw = broadcast(w);
    u64* v = u + half;
    for (std::size_t j = 0; j < half; j += 4) {
        const Vec y = _mm256_xor_si256(load(u + j), load(v + j));
        store(v + j, y);
        store(u + j, _mm256_xor_si256(load(u + j), multiply(y, tw)));
    }
}

// Stages i and i - 1 of the block p = d + c of 2^(i+1) words, in quarters x0..x3.
struct PairTwiddles {
    Vec outer, low, high;  // stage i; stage i - 1 on (x0, x1) and on (x2, x3)
};

inline PairTwiddles pair_twiddles(std::size_t c, int i) {
    const std::size_t q = std::size_t(1) << (i - 1);
    return {broadcast(omega(c >> i)), broadcast(omega(c >> (i - 1))), broadcast(omega((c + 2 * q) >> (i - 1)))};
}

void stages_forward(u64* p, std::size_t c, int i) {
    const std::size_t q = std::size_t(1) << (i - 1);
    const PairTwiddles t = pair_twiddles(c, i);
    for (std::size_t j = 0; j < q; j += 4) {
        Vec x0 = load(p + j), x1 = load(p + q + j), x2 = load(p + 2 * q + j), x3 = load(p + 3 * q + j);
        x0 = _mm256_xor_si256(x0, multiply(x2, t.outer));
        x1 = _mm256_xor_si256(x1, multiply(x3, t.outer));
        x2 = _mm256_xor_si256(x2, x0);
        x3 = _mm256_xor_si256(x3, x1);
        x0 = _mm256_xor_si256(x0, multiply(x1, t.low));
        x2 = _mm256_xor_si256(x2, multiply(x3, t.high));
        store(p + j, x0);
        store(p + q + j, _mm256_xor_si256(x1, x0));
        store(p + 2 * q + j, x2);
        store(p + 3 * q + j, _mm256_xor_si256(x3, x2));
    }
}

void stages_inverse(u64* p, std::size_t c, int i) {
    const std::size_t q = std::size_t(1) << (i - 1);
    const PairTwiddles t = pair_twiddles(c, i);
    for (std::size_t j = 0; j < q; j += 4) {
        Vec x0 = load(p + j), x1 = load(p + q + j), x2 = load(p + 2 * q + j), x3 = load(p + 3 * q + j);
        x1 = _mm256_xor_si256(x1, x0);
        x3 = _mm256_xor_si256(x3, x2);
        x0 = _mm256_xor_si256(x0, multiply(x1, t.low));
        x2 = _mm256_xor_si256(x2, multiply(x3, t.high));
        x2 = _mm256_xor_si256(x2, x0);
        x3 = _mm256_xor_si256(x3, x1);
        store(p + j, _mm256_xor_si256(x0, multiply(x2, t.outer)));
        store(p + q + j, _mm256_xor_si256(x1, multiply(x3, t.outer)));
        store(p + 2 * q + j, x2);
        store(p + 3 * q + j, x3);
    }
}

// Stages 2, 1, 0 of the 8 words at c (c a multiple of 8): x holds c + [0, 4), y c + [4, 8).
// Stage 1 pairs [c, c+1, c+4, c+5] with [c+2, c+3, c+6, c+7]; stage 0 pairs the even indices
// [c, c+2, c+4, c+6] with the odd ones.
struct Twiddles8 {
    Vec w2, w1, w0;
};

inline Twiddles8 twiddles8(std::size_t c) {
    const Vec b1 = broadcast(kBeta[1]), b2 = broadcast(kBeta[2]), zero = _mm256_setzero_si256();
    const Vec w1 = _mm256_xor_si256(broadcast(omega(c >> 1)), _mm256_blend_epi32(zero, b1, 0xF0));
    const Vec w0 = _mm256_xor_si256(broadcast(omega(c)), _mm256_xor_si256(_mm256_blend_epi32(zero, b1, 0xCC),
                                                                           _mm256_blend_epi32(zero, b2, 0xF0)));
    return {broadcast(omega(c >> 2)), w1, w0};
}

inline void kernel8_forward(u64* p, std::size_t c) {
    const Twiddles8 t = twiddles8(c);
    Vec x = load(p), y = load(p + 4);
    x = _mm256_xor_si256(x, multiply(y, t.w2));
    y = _mm256_xor_si256(y, x);
    Vec u = _mm256_permute2x128_si256(x, y, 0x20), v = _mm256_permute2x128_si256(x, y, 0x31);
    u = _mm256_xor_si256(u, multiply(v, t.w1));
    v = _mm256_xor_si256(v, u);
    Vec e = _mm256_unpacklo_epi64(u, v), o = _mm256_unpackhi_epi64(u, v);
    e = _mm256_xor_si256(e, multiply(o, t.w0));
    o = _mm256_xor_si256(o, e);
    u = _mm256_unpacklo_epi64(e, o), v = _mm256_unpackhi_epi64(e, o);
    store(p, _mm256_permute2x128_si256(u, v, 0x20));
    store(p + 4, _mm256_permute2x128_si256(u, v, 0x31));
}

inline void kernel8_inverse(u64* p, std::size_t c) {
    const Twiddles8 t = twiddles8(c);
    const Vec x0 = load(p), y0 = load(p + 4);
    Vec u = _mm256_permute2x128_si256(x0, y0, 0x20), v = _mm256_permute2x128_si256(x0, y0, 0x31);
    Vec e = _mm256_unpacklo_epi64(u, v), o = _mm256_unpackhi_epi64(u, v);
    o = _mm256_xor_si256(o, e);
    e = _mm256_xor_si256(e, multiply(o, t.w0));
    u = _mm256_unpacklo_epi64(e, o), v = _mm256_unpackhi_epi64(e, o);
    v = _mm256_xor_si256(v, u);
    u = _mm256_xor_si256(u, multiply(v, t.w1));
    Vec x = _mm256_permute2x128_si256(u, v, 0x20), y = _mm256_permute2x128_si256(u, v, 0x31);
    y = _mm256_xor_si256(y, x);
    x = _mm256_xor_si256(x, multiply(y, t.w2));
    store(p, x);
    store(p + 4, y);
}

// Block d[c, c + 2^(i+1)), i < kBlockLog: stages i down to 0 (or back), one at a time.
void forward_block(u64* d, std::size_t c, int i) {
    const std::size_t end = c + (std::size_t(2) << i);
    for (int s = i; s >= kMinLog; --s) {
        const std::size_t half = std::size_t(1) << s;
        for (std::size_t b = c; b < end; b += 2 * half) stage_forward(d + b, half, omega(b >> s));
    }
    for (std::size_t b = c; b < end; b += 8) kernel8_forward(d + b, b);
}

void inverse_block(u64* d, std::size_t c, int i) {
    const std::size_t end = c + (std::size_t(2) << i);
    for (std::size_t b = c; b < end; b += 8) kernel8_inverse(d + b, b);
    for (int s = kMinLog; s <= i; ++s) {
        const std::size_t half = std::size_t(1) << s;
        for (std::size_t b = c; b < end; b += 2 * half) stage_inverse(d + b, half, omega(b >> s));
    }
}

// Block d[c, c + 2^(i+1)), stages i down to 0.
void forward(u64* d, std::size_t c, int i) {
    const std::size_t half = std::size_t(1) << i;
    if (i < kBlockLog) {
        forward_block(d, c, i);
    } else if (i == kBlockLog) {
        stage_forward(d + c, half, omega(c >> i));
        forward(d, c, i - 1);
        forward(d, c + half, i - 1);
    } else {
        stages_forward(d + c, c, i);
        for (std::size_t k = 0; k < 4; ++k) forward(d, c + k * half / 2, i - 2);
    }
}

// a[c, c + 2^(i+1)) holds the transform of a; b the X coefficients of b. Transforms b's block,
// multiplies it into a's and transforms a's back, depth first: each block of 2^kBlockLog words
// does all three while in L1.
void multiply_transformed(u64* a, u64* b, std::size_t c, int i) {
    const std::size_t half = std::size_t(1) << i;
    if (i < kBlockLog) {
        forward_block(b, c, i);
        for (std::size_t k = c; k < c + 2 * half; k += 4) store(a + k, multiply(load(a + k), load(b + k)));
        inverse_block(a, c, i);
    } else if (i == kBlockLog) {
        stage_forward(b + c, half, omega(c >> i));
        multiply_transformed(a, b, c, i - 1);
        multiply_transformed(a, b, c + half, i - 1);
        stage_inverse(a + c, half, omega(c >> i));
    } else {
        stages_forward(b + c, c, i);
        for (std::size_t k = 0; k < 4; ++k) multiply_transformed(a, b, c + k * half / 2, i - 2);
        stages_inverse(a + c, c, i);
    }
}

// words u64 words, 2 MiB aligned and zeroed, in huge pages where the kernel allows.
u64* allocate(std::size_t words) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    const std::size_t bytes = (words * sizeof(u64) + kHuge - 1) / kHuge * kHuge + kHuge;
    void* region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) std::abort();
    const std::uintptr_t start = (reinterpret_cast<std::uintptr_t>(region) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
    ::madvise(reinterpret_cast<void*>(start), bytes - kHuge, MADV_HUGEPAGE);
#endif
    return reinterpret_cast<u64*>(start);
}

// a * b, the product left in a[0, n + m - 1). a and b hold 2^l words each, zero past n and m.
void convolve(u64* a, std::size_t n, u64* b, std::size_t m, int l) {
    const std::size_t half = std::size_t(1) << (l - 1);
    change_basis<false>(a, l, n);
    change_basis<false>(b, l, m);
    // The top stage has twiddle omega_0 = 0: (u, v) -> (u, u + v). With v = 0 it copies u.
    if (l - 1 > kBlockLog && n <= half) {
        std::memcpy(a + half, a, half * sizeof(u64));
        forward(a, 0, l - 2);
        forward(a, half, l - 2);
    } else {
        forward(a, 0, l - 1);
    }
    if (l - 1 > kBlockLog && m <= half) {
        std::memcpy(b + half, b, half * sizeof(u64));
        multiply_transformed(a, b, 0, l - 2);
        multiply_transformed(a, b, half, l - 2);
        stage_inverse(a, half, 0);
    } else {
        multiply_transformed(a, b, 0, l - 1);
    }
    change_basis<true>(a, l, std::size_t(1) << l);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    const int l = std::max(kMinLog, int(std::bit_width(n + m - 2)));
    u64* a = allocate(std::size_t(2) << l);
    u64* b = a + (std::size_t(1) << l);
    io::read_bulk(in, a, n);
    io::read_bulk(in, b, m);
    convolve(a, n, b, m, l);
    io::Writer out;
    out.write_array(a, n + m - 1, ' ');
    out.write('\n');
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here). _exit skips their teardown too.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
