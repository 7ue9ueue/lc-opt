// a * b mod 2^64: the product modulo five NTT primes (product.hpp, or lib/multimod when a factor
// fills more than half of the transform), the Chinese remainder theorem in 64-bit arithmetic,
// fixed-width output (fields64.hpp).
#include <array>

#include "fields64.hpp"
#include "lib/io/bulk64.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "lib/multimod/transform.hpp"
#include "lib/run/early.hpp"
#include "product.hpp"

namespace {

using multimod::Modulus;
using multimod::Vec;

// Each coefficient is below 2^19 (2^64 - 1)^2 < 2^147; the primes' product is about 2^149.35.
constexpr int kPrimes = 5;
constexpr std::array<std::array<std::uint32_t, 2>, kPrimes> kPrimeList = {{
    {998244353, 3}, {985661441, 3}, {976224257, 3}, {975175681, 17}, {972029953, 10}}};  // prime, generator

constexpr std::uint32_t multiply_mod(std::uint64_t x, std::uint64_t y, std::uint32_t p) {
    return std::uint32_t(x % p * (y % p) % p);
}

constexpr std::uint32_t inverse_mod(std::uint32_t x, std::uint32_t p) {
    std::uint32_t result = 1;
    for (std::uint32_t e = p - 2; e; e >>= 1, x = multiply_mod(x, x, p))
        if (e & 1) result = multiply_mod(result, x, p);
    return result;
}

// CRT with M = p_0 ... p_4 and M_k = M / p_k: the transform for prime k returns
// y_k = c / M_k mod p_k (scale[k] = 1 / M_k mod p_k). Then c = sum_k y_k M_k - t M with
// t = floor(sum_k y_k / p_k): the sum's fraction is c / M < 2^147 / M < 0.2.
struct Crt {
    std::array<std::uint32_t, kPrimes> scale{};
    std::array<std::uint64_t, kPrimes> place{};  // M_k mod 2^64
    std::uint64_t modulus = 1;                   // M mod 2^64
};

constexpr Crt kCrt = [] {
    Crt crt;
    for (int k = 0; k < kPrimes; ++k) {
        const std::uint32_t p = kPrimeList[k][0];
        std::uint32_t other = 1;  // M_k mod p
        crt.place[k] = 1;
        for (int j = 0; j < kPrimes; ++j) {
            if (j == k) continue;
            other = multiply_mod(other, kPrimeList[j][0], p);
            crt.place[k] *= kPrimeList[j][0];
        }
        crt.scale[k] = inverse_mod(other, p);
        crt.modulus *= p;
    }
    return crt;
}();

using Residues = std::array<const std::uint32_t*, kPrimes>;

// t of coefficients [i, i + 8), in float: each term's error is a few 2^-24, against a margin of
// 0.2 on either side of t + 0.4.
[[gnu::always_inline]] inline Vec quotients(const Residues& y, std::size_t i) {
    const auto term = [&](int k) {
        const Vec r = _mm256_load_si256(reinterpret_cast<const Vec*>(y[k] + i));
        return std::pair(_mm256_cvtepi32_ps(r), _mm256_set1_ps(1.0f / float(kPrimeList[k][0])));
    };
    const auto [r0, i0] = term(0);
    const auto [r1, i1] = term(1);
    const auto [r2, i2] = term(2);
    const auto [r3, i3] = term(3);
    const auto [r4, i4] = term(4);
    const __m256 sum01 = _mm256_fmadd_ps(r0, i0, _mm256_mul_ps(r1, i1));
    const __m256 sum23 = _mm256_fmadd_ps(r2, i2, _mm256_mul_ps(r3, i3));
    const __m256 sum4 = _mm256_fmadd_ps(r4, i4, _mm256_set1_ps(0.4f));
    return _mm256_cvttps_epi32(_mm256_add_ps(_mm256_add_ps(sum01, sum23), sum4));  // < 5
}

// c mod 2^64 of coefficients [i, i + 8) into out[0, 8), given their t. With c_k = lo + 2^32 hi,
// y c_k mod 2^64 = y lo + 2^32 (y hi mod 2^32): y lo in 64 bits (even and odd lanes apart), y hi
// in 32-bit lanes.
[[gnu::always_inline]] inline void combine(const Residues& y, std::size_t i, Vec t, std::uint64_t* out) {
    Vec even = _mm256_setzero_si256(), odd = even, high = even;
    const auto term = [&](Vec x, std::uint64_t c) {
        const Vec lo = _mm256_set1_epi64x(std::int64_t(c & 0xFFFFFFFF));
        even = _mm256_add_epi64(even, _mm256_mul_epu32(x, lo));
        odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_srli_epi64(x, 32), lo));
        high = _mm256_add_epi32(high, _mm256_mullo_epi32(x, _mm256_set1_epi32(int(c >> 32))));
    };
#pragma GCC unroll 5  // -O2 kept the loop, reloading and splitting each constant per step
    for (int k = 0; k < kPrimes; ++k) term(_mm256_load_si256(reinterpret_cast<const Vec*>(y[k] + i)), kCrt.place[k]);
    term(t, 0 - kCrt.modulus);
    even = _mm256_add_epi64(even, _mm256_slli_epi64(high, 32));                                   // coefficients 0 2 4 6
    odd = _mm256_add_epi64(odd, _mm256_and_si256(high, _mm256_set1_epi64x(std::int64_t(0xFFFFFFFF00000000))));  // 1 3 5 7
    const Vec low = _mm256_unpacklo_epi64(even, odd), top = _mm256_unpackhi_epi64(even, odd);  // 0 1 4 5, 2 3 6 7
    _mm256_storeu_si256(reinterpret_cast<Vec*>(out), _mm256_permute2x128_si256(low, top, 0x20));
    _mm256_storeu_si256(reinterpret_cast<Vec*>(out + 4), _mm256_permute2x128_si256(low, top, 0x31));
}

// Coefficients [begin, begin + count) into out, count a multiple of 8; the residues are readable
// 8 words past. The quotients of the next 8 are issued before the products of these.
void reconstruct(const Residues& y, std::size_t begin, std::size_t count, std::uint64_t* out) {
    Vec t = quotients(y, begin);
    for (std::size_t j = 0; j < count; j += 8) {
        const Vec next = quotients(y, begin + j + 8);
        combine(y, begin + j, t, out + j);
        t = next;
    }
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>(), count = n + m - 1;
    const int lg = std::max(6, int(std::bit_width(count - 1)));
    const std::size_t len = std::size_t(1) << lg, words = len + multimod::Transform::kPadding;
    const auto padded = [](std::size_t k) { return (k + 7) & ~std::size_t(7); };

    mem::Arena arena(8 * (padded(n) + padded(m) + len) +
                     4 * (multimod::Transform::table_words(lg) + (kPrimes + 1) * words) + 8 * fields64::kBlock +
                     fields64::kTextBytes + 64 * 16);
    // Zero up to half the length too: the first level reads that far. a's storage also holds the
    // last prime's work words (at least 2^lg + kPadding).
    auto* a = arena.take<std::uint64_t>(std::max(padded(n), len / 2) + multimod::Transform::kPadding / 2);
    auto* b = arena.take<std::uint64_t>(std::max(padded(m), len / 2));
    io::read_bulk(in, a, n);
    io::read_bulk(in, b, m);

    auto* tables = arena.take<std::uint32_t>(multimod::Transform::table_words(lg));
    const multimod::Transform transform(lg, tables);
    const wide::Product product(lg, tables);
    const bool fits = wide::Product::fits(lg, n, m);
    // The last prime's residues go to the work words, and its work to a: no fresh pages for it.
    auto* work = arena.take<std::uint32_t>(words);
    Residues residues;
    for (int k = 0; k < kPrimes; ++k) {
        const Modulus mod(kPrimeList[k][0], kPrimeList[k][1]);
        const bool last = k + 1 == kPrimes;
        auto* r = last ? work : arena.take<std::uint32_t>(words);
        auto* w = last ? reinterpret_cast<std::uint32_t*>(a) : work;
        if (fits)
            product.multiply(multimod::Wide{a, n}, multimod::Wide{b, m}, r, w, mod, kCrt.scale[k]);
        else
            transform.multiply(multimod::Wide{a, n}, multimod::Wide{b, m}, r, w, mod, kCrt.scale[k]);
        residues[k] = r;
    }

    io::Writer out;
    char* text = arena.take<char>(fields64::kTextBytes);
    fields64::prepare(text);
    auto* block = arena.take<std::uint64_t>(fields64::kBlock);
    for (std::size_t i = 0; i < count; i += fields64::kBlock) {
        const std::size_t size = std::min(fields64::kBlock, count - i);
        reconstruct(residues, i, (size + 7) & ~std::size_t(7), block);
        out.write(std::string_view(text, fields64::format(text, block, size, i + size == count)));
    }
}

}  // namespace

RUN_EARLY(solve)
