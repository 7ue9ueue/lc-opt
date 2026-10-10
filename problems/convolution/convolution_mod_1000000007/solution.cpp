// a * b mod 1000000007: the product modulo three NTT primes (lib/multimod: lib/ntt's transform
// with the modulus set at run time), the Chinese remainder theorem straight to residues mod
// 10^9 + 7 by one Montgomery reduction, fixed-width output: 10 bytes per value (fields10.hpp),
// or 11 in the rare blocks with a value >= 10^9 (fields11.hpp).
#include <array>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "fields10.hpp"
#include "fields11.hpp"
#include "lib/multimod/transform.hpp"
#include "lib/run/early.hpp"

namespace {

using multimod::Modulus;
using multimod::Vec;

constexpr std::uint32_t kMod = 1000000007;

// Each coefficient is below 2^19 kMod^2 < 2^79; the primes' product M is about 2^89.6.
constexpr int kPrimes = 3;
constexpr std::array<std::array<std::uint32_t, 2>, kPrimes> kPrimeList = {{
    {998244353, 3}, {985661441, 3}, {976224257, 3}}};  // prime, generator; inputs < kMod < 2 p

constexpr std::uint32_t multiply_mod(std::uint64_t x, std::uint64_t y, std::uint32_t p) {
    return std::uint32_t(x % p * (y % p) % p);
}

constexpr std::uint32_t inverse_mod(std::uint32_t x, std::uint32_t p) {
    std::uint32_t result = 1;
    for (std::uint32_t e = p - 2; e; e >>= 1, x = multiply_mod(x, x, p))
        if (e & 1) result = multiply_mod(result, x, p);
    return result;
}

// CRT with M_k = M / p_k: the transform for prime k returns y_k = c / M_k mod p_k
// (scale[k] = 1 / M_k mod p_k). Then c = sum_k y_k M_k - t M with t = floor(sum_k y_k / p_k):
// the sum's fraction is c / M < 2^-10. Modulo kMod, with R = 2^32:
// s = sum_k y_k (M_k R mod kMod) + t (-M R mod kMod) < (3 2^30 + 2) kMod < R kMod, and
// s / R mod kMod = c mod kMod (Montgomery reduction).
struct Crt {
    std::array<std::uint32_t, kPrimes> scale{};
    std::array<std::uint32_t, kPrimes> place{};  // M_k R mod kMod
    std::uint32_t wrap = 0;                      // -M R mod kMod
    std::uint32_t neg_inverse = 0;               // -1 / kMod mod 2^32
};

constexpr Crt kCrt = [] {
    Crt crt;
    constexpr std::uint32_t r = std::uint32_t((std::uint64_t(1) << 32) % kMod);
    std::uint32_t modulus = 1;  // M mod kMod
    for (int k = 0; k < kPrimes; ++k) {
        const std::uint32_t p = kPrimeList[k][0];
        std::uint32_t other = 1, place = 1;  // M_k mod p, M_k mod kMod
        for (int j = 0; j < kPrimes; ++j) {
            if (j == k) continue;
            other = multiply_mod(other, kPrimeList[j][0], p);
            place = multiply_mod(place, kPrimeList[j][0], kMod);
        }
        crt.scale[k] = inverse_mod(other, p);
        crt.place[k] = multiply_mod(place, r, kMod);
        modulus = multiply_mod(modulus, p, kMod);
    }
    crt.wrap = multiply_mod(kMod - modulus, r, kMod);
    std::uint32_t x = kMod;  // Newton: correct to 3, 6, 12, 24, 48 bits
    for (int i = 0; i < 4; ++i) x *= 2 - kMod * x;
    crt.neg_inverse = 0 - x;
    return crt;
}();

#ifdef FORCE_WIDE
constexpr bool kForceWide = true;  // test hook: every block in 11-byte fields
#else
constexpr bool kForceWide = false;
#endif

using Residues = std::array<const std::uint32_t*, kPrimes>;

[[gnu::always_inline]] inline Vec load(const std::uint32_t* p) { return _mm256_load_si256(reinterpret_cast<const Vec*>(p)); }

// t of coefficients [i, i + 8), in float: the error is a few 2^-24, the margin about 0.5.
[[gnu::always_inline]] inline Vec quotients(const Residues& y, std::size_t i) {
    const auto term = [&](int k) { return _mm256_cvtepi32_ps(load(y[k] + i)); };
    const auto inv = [](int k) { return _mm256_set1_ps(1.0f / float(kPrimeList[k][0])); };
    const __m256 sum = _mm256_fmadd_ps(term(0), inv(0), _mm256_fmadd_ps(term(1), inv(1), _mm256_set1_ps(0.5f)));
    return _mm256_cvttps_epi32(_mm256_fmadd_ps(term(2), inv(2), sum));  // < 3
}

// c mod kMod of coefficients [i, i + 8), given their t.
[[gnu::always_inline]] inline Vec combine(const Residues& y, std::size_t i, Vec t) {
    const Vec y0 = load(y[0] + i), y1 = load(y[1] + i), y2 = load(y[2] + i);
    const auto set = [](std::uint32_t c) { return _mm256_set1_epi64x(c); };
    // s / R mod kMod in the high dword of each qword, < 2 kMod; x: the even dwords.
    const auto reduce = [&](Vec x0, Vec x1, Vec x2, Vec tx) {
        const Vec s = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(x0, set(kCrt.place[0])), _mm256_mul_epu32(x1, set(kCrt.place[1]))),
                                       _mm256_add_epi64(_mm256_mul_epu32(x2, set(kCrt.place[2])), _mm256_mul_epu32(tx, set(kCrt.wrap))));
        const Vec m = _mm256_mul_epu32(s, set(kCrt.neg_inverse));
        return _mm256_add_epi64(s, _mm256_mul_epu32(m, set(kMod)));  // < 2^33 kMod
    };
    const auto odd = [](Vec x) { return _mm256_srli_epi64(x, 32); };
    const Vec even_sum = reduce(y0, y1, y2, t), odd_sum = reduce(odd(y0), odd(y1), odd(y2), odd(t));
    const Vec u = _mm256_blend_epi32(_mm256_srli_epi64(even_sum, 32), odd_sum, 0xAA);
    return _mm256_min_epu32(u, _mm256_sub_epi32(u, _mm256_set1_epi32(int(kMod))));
}

// Coefficients [begin, begin + count) into out, count a multiple of 8; the residues are readable
// 8 words past. The quotients of the next 8 are issued before the products of these. Returns
// whether some coefficient is >= 10^9.
bool reconstruct(const Residues& y, std::size_t begin, std::size_t count, std::uint32_t* out) {
    Vec t = quotients(y, begin), top = _mm256_setzero_si256();
    for (std::size_t j = 0; j < count; j += 8) {
        const Vec next = quotients(y, begin + j + 8), c = combine(y, begin + j, t);
        _mm256_store_si256(reinterpret_cast<Vec*>(out + j), c);
        top = _mm256_max_epu32(top, c);
        t = next;
    }
    const Vec small = _mm256_cmpgt_epi32(_mm256_set1_epi32(1000000000), top);
    return _mm256_movemask_epi8(small) != -1;
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>(), count = n + m - 1;
    const int lg = std::max(6, int(std::bit_width(count - 1)));
    const std::size_t len = std::size_t(1) << lg, words = len + multimod::Transform::kPadding;
    const auto padded = [](std::size_t k) { return (k + 7) & ~std::size_t(7); };

    mem::Arena arena(4 * (padded(n) + padded(m) + len + multimod::Transform::table_words(lg) + (kPrimes + 1) * words +
                          fields11::kBlock) +
                     fields11::kTextBytes + 64 * 16);
    // Zero up to half the length too: the first level reads that far. b holds the last prime's
    // transform of b in place.
    auto* a = arena.take<std::uint32_t>(std::max(padded(n), len / 2));
    auto* b = arena.take<std::uint32_t>(words);
    io::read_bulk(in, a, n);
    io::read_bulk(in, b, m);

    const multimod::Transform transform(lg, arena.take<std::uint32_t>(multimod::Transform::table_words(lg)));
    auto* work = arena.take<std::uint32_t>(words);
    Residues residues;
    for (int k = 0; k < kPrimes; ++k) {
        const Modulus mod(kPrimeList[k][0], kPrimeList[k][1]);
        const bool last = k + 1 == kPrimes;
        auto* r = last ? work : arena.take<std::uint32_t>(words);
        transform.multiply(multimod::Padded{a, n}, multimod::Padded{b, m}, r, last ? b : work, mod, kCrt.scale[k]);
        residues[k] = r;
    }

    io::Writer out;
    static_assert(fields10::kBlock == fields11::kBlock);
    constexpr std::size_t kBlock = fields10::kBlock;
    char* text = arena.take<char>(fields11::kTextBytes);
    auto* block = arena.take<std::uint32_t>(kBlock);
    for (std::size_t i = 0; i < count; i += kBlock) {
        const std::size_t size = std::min(kBlock, count - i);
        const bool wide = reconstruct(residues, i, padded(size), block);
        if (wide || kForceWide) fields11::write(out, block, size, text, i + size == count);
        else fields10::write(out, block, size, text, i + size == count);
    }
}

}  // namespace

RUN_EARLY(solve)
