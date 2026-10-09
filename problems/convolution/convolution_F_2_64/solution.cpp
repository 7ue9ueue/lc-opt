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

inline Vec reduce(Vec low, Vec high) {
    const Vec q = _mm256_xor_si256(_mm256_xor_si256(high, _mm256_srli_epi64(high, 60)),
                                   _mm256_xor_si256(_mm256_srli_epi64(high, 61), _mm256_srli_epi64(high, 63)));
    return _mm256_xor_si256(_mm256_xor_si256(low, q),
                            _mm256_xor_si256(_mm256_xor_si256(_mm256_slli_epi64(q, 1), _mm256_slli_epi64(q, 3)),
                                             _mm256_slli_epi64(q, 4)));
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

// Change of basis between monomials and X, on rows of W words that are xored as units.

// dst[0, count) ^= src[0, count), in rows of W words.
template <std::size_t W>
inline void xor_rows(u64* dst, const u64* src, std::size_t count) {
    for (std::size_t k = 0; k < count * W; ++k) dst[k] ^= src[k];
}

// f[0, Len) = sum_m g_m(x) t^m with t = x^Tau + x, deg g_m < Tau: g_m in f[m * Tau, (m + 1) * Tau).
// With A, B the halves and d = Len / 2 / Tau: x^(Len/2) = t^d + x^d, so
// f = (A + x^d B_low + x^d B_high) + t^d (B + B_high), B_high the top d entries of B.
template <std::size_t Len, std::size_t Tau, std::size_t W, bool Inverse>
void taylor(u64* f) {
    if constexpr (Len > Tau) {
        constexpr std::size_t half = Len / 2, d = half / Tau;
        u64* a = f;
        u64* b = f + half * W;
        if constexpr (Inverse) {
            taylor<half, Tau, W, true>(a);
            taylor<half, Tau, W, true>(b);
            xor_rows<W>(b, b + (half - d) * W, d);
            xor_rows<W>(a + d * W, b, half - d);
            xor_rows<W>(a + d * W, b + (half - d) * W, d);
        } else {
            xor_rows<W>(a + d * W, b, half - d);
            xor_rows<W>(a + d * W, b + (half - d) * W, d);
            xor_rows<W>(b, b + (half - d) * W, d);
            taylor<half, Tau, W, false>(a);
            taylor<half, Tau, W, false>(b);
        }
    }
}

// Monomial coefficients of f[0, 2^L) to X coefficients (or back). With K = 2^k the largest power
// of two below L: expand in t = s_K = x^(2^K) + x, convert each row g_m (length 2^K), then each
// column (a polynomial in t, whose basis s_i(t) = s_{K + i}(x) for i < L - K <= K).
template <int L, std::size_t W, bool Inverse>
void change_basis(u64* f) {
    if constexpr (L >= 2) {
        constexpr int K = 1 << (std::bit_width(unsigned(L - 1)) - 1);
        constexpr std::size_t tau = std::size_t(1) << K, rows = std::size_t(1) << (L - K);
        if constexpr (Inverse) {
            change_basis<L - K, W * tau, true>(f);
            for (std::size_t m = 0; m < rows; ++m) change_basis<K, W, true>(f + m * tau * W);
            taylor<(std::size_t(1) << L), tau, W, true>(f);
        } else {
            taylor<(std::size_t(1) << L), tau, W, false>(f);
            for (std::size_t m = 0; m < rows; ++m) change_basis<K, W, false>(f + m * tau * W);
            change_basis<L - K, W * tau, false>(f);
        }
    }
}

template <bool Inverse, int L = kMaxLog>
void change_basis(u64* f, int l) {
    if constexpr (L >= kMinLog) {
        if (l == L) return change_basis<L, 1, Inverse>(f);
        change_basis<Inverse, L - 1>(f, l);
    }
}

// The transform. Stage i maps the halves (u, v) of each block of 2^(i+1) at c to
// (u + w v, u + w v + v), w = omega_{c >> i}; the inverse undoes the stages in reverse order.

constexpr std::size_t kBlockLog = 12;  // blocks of 2^12 words (32 KiB) run stage by stage

inline void stage_forward(u64* u, std::size_t half, u64 w) {
    const Vec tw = _mm256_set1_epi64x(std::int64_t(w));
    u64* v = u + half;
    for (std::size_t j = 0; j < half; j += 4) {
        const Vec x = _mm256_xor_si256(load(u + j), multiply(load(v + j), tw));
        store(u + j, x);
        store(v + j, _mm256_xor_si256(load(v + j), x));
    }
}

inline void stage_inverse(u64* u, std::size_t half, u64 w) {
    const Vec tw = _mm256_set1_epi64x(std::int64_t(w));
    u64* v = u + half;
    for (std::size_t j = 0; j < half; j += 4) {
        const Vec y = _mm256_xor_si256(load(u + j), load(v + j));
        store(v + j, y);
        store(u + j, _mm256_xor_si256(load(u + j), multiply(y, tw)));
    }
}

// Stages 2, 1, 0 of the 8 words at c (c a multiple of 8): x holds c + [0, 4), y c + [4, 8).
// Stage 1 pairs [c, c+1, c+4, c+5] with [c+2, c+3, c+6, c+7]; stage 0 pairs the even indices
// [c, c+2, c+4, c+6] with the odd ones.
struct Twiddles8 {
    Vec w2, w1, w0;
};

inline Twiddles8 twiddles8(std::size_t c) {
    const Vec b1 = _mm256_set1_epi64x(std::int64_t(kBeta[1])), b2 = _mm256_set1_epi64x(std::int64_t(kBeta[2]));
    const Vec w1 = _mm256_xor_si256(_mm256_set1_epi64x(std::int64_t(omega(c >> 1))),
                                    _mm256_blend_epi32(_mm256_setzero_si256(), b1, 0xF0));
    const Vec w0 = _mm256_xor_si256(_mm256_set1_epi64x(std::int64_t(omega(c))),
                                    _mm256_xor_si256(_mm256_blend_epi32(_mm256_setzero_si256(), b1, 0xCC),
                                                     _mm256_blend_epi32(_mm256_setzero_si256(), b2, 0xF0)));
    return {_mm256_set1_epi64x(std::int64_t(omega(c >> 2))), w1, w0};
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

// Block d[c, c + 2^(i+1)), stages i down to 0.
void forward(u64* d, std::size_t c, int i) {
    if (i < int(kBlockLog)) {
        for (int s = i; s >= kMinLog; --s) {
            const std::size_t half = std::size_t(1) << s;
            for (std::size_t b = c; b < c + (half << (i + 1 - s)); b += 2 * half) stage_forward(d + b, half, omega(b >> s));
        }
        for (std::size_t b = c; b < c + (std::size_t(2) << i); b += 8) kernel8_forward(d + b, b);
        return;
    }
    const std::size_t half = std::size_t(1) << i;
    stage_forward(d + c, half, omega(c >> i));
    forward(d, c, i - 1);
    forward(d, c + half, i - 1);
}

void inverse(u64* d, std::size_t c, int i) {
    if (i < int(kBlockLog)) {
        for (std::size_t b = c; b < c + (std::size_t(2) << i); b += 8) kernel8_inverse(d + b, b);
        for (int s = kMinLog; s <= i; ++s) {
            const std::size_t half = std::size_t(1) << s;
            for (std::size_t b = c; b < c + (half << (i + 1 - s)); b += 2 * half) stage_inverse(d + b, half, omega(b >> s));
        }
        return;
    }
    const std::size_t half = std::size_t(1) << i;
    inverse(d, c, i - 1);
    inverse(d, c + half, i - 1);
    stage_inverse(d + c, half, omega(c >> i));
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

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    const int l = std::max(kMinLog, int(std::bit_width(n + m - 2)));
    const std::size_t len = std::size_t(1) << l;
    u64* a = allocate(2 * len);
    u64* b = a + len;
    io::read_bulk(in, a, n);
    io::read_bulk(in, b, m);
    change_basis<false>(a, l);
    change_basis<false>(b, l);
    forward(a, 0, l - 1);
    forward(b, 0, l - 1);
    for (std::size_t k = 0; k < len; k += 4) store(a + k, multiply(load(a + k), load(b + k)));
    inverse(a, 0, l - 1);
    change_basis<true>(a, l);
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
