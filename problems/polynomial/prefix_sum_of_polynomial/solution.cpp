// g with g(n) = sum_(i < n) f(i) for f = sum a_t x^t of N <= 2^19 coefficients, mod 998244353.
// With y / (e^y - 1) = sum_j B_j y^j / j! (Bernoulli numbers), g_0 = 0 and
//   g_(k+1) = (1 / (k+1)!) sum_(t >= k) a_t t! B_(t-k) / (t-k)!.
// y / (e^y - 1) = E(y^2 / 4) - y / 2 with E(w) = sqrt(w) coth(sqrt(w)) = C(w) / S(w),
// C_j = 1 / (2j)!, S_j = 1 / (2j + 1)!. With abar_t = a_t t! / 2^t this is
//   g_(k+1) = 2^k / (k+1)! sum_j abar_(k+2j) E_j - a_(k+1) / 2:
// for even and for odd k a product with E of half the length.
// E to K >= N / 2 terms: sigma = 1 / S mod w^(K/2) by Newton (lib/poly/inverse.hpp), then one
// division step (A. Karp, P. Markstein): q0 = C sigma mod w^(K/2), r = (C - S q0) / w^(K/2),
// E = q0 + w^(K/2) (r sigma mod w^(K/2)). Products by lib/poly's transforms; factorials and the
// weights from product chains in 32 lanes (lib/poly/factorials.hpp). Output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp).
#include <algorithm>
#include <bit>
#include <cstddef>
#include <span>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/poly/factorials.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

using ntt::detail::broadcast;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::multiply_mod;
using ntt::detail::power;
using ntt::detail::reduce;
using ntt::detail::Vec;
using poly::Half;
using poly::Lanes;
using poly::detail::Chain;
using poly::detail::difference;
using poly::detail::load;
using poly::detail::load_unaligned;
using poly::detail::montgomery;
using poly::detail::scan;
using poly::detail::scan_chunk;
using poly::detail::store_unaligned;

constexpr std::uint32_t kHalf = (kP + 1) / 2;  // 1 / 2 mod P

std::uint32_t mont(std::uint32_t x) { return multiply_mod(x, kR); }  // x 2^32 mod P

// Lanes 0..3 of x at p, lanes 4..7 at q.
[[gnu::always_inline]] inline void store_halves(std::uint32_t* p, std::uint32_t* q, Vec x) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_castsi256_si128(x));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q), _mm256_extracti128_si256(x, 1));
}

// 4 words at p in lanes 0..3, 4 words at q in lanes 4..7.
[[gnu::always_inline]] inline Vec load_halves(const std::uint32_t* p, const std::uint32_t* q) {
    return _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(q), reinterpret_cast<const __m128i*>(p));
}

// x / 2 mod P for canonical x.
[[gnu::always_inline]] inline Vec halve(Vec x) {
    const Vec odd = _mm256_srai_epi32(_mm256_slli_epi32(x, 31), 31);
    return _mm256_srli_epi32(_mm256_add_epi32(x, _mm256_and_si256(odd, broadcast(kP))), 1);
}

// The prefix sum of f for n >= 1 coefficients. Scans over the n positions t (or k) run in 32 lanes
// of C positions, C = scan_chunk(n), so up to 1023 positions past n. Products of length 2K, K a
// power of two >= max(64, ceil(n / 2)): the even and the odd abar reversed, pe[K - 1 - i] =
// abar_(2i) and po[K - 1 - i] = abar_(2i+1), times E in [0, K): the sums for k = 2m and
// k = 2m + 1 sit at pe[K - 1 - m] and po[K - 1 - m]. One mapping in huge pages, never freed.
class PrefixSum {
public:
    explicit PrefixSum(std::size_t n)
        : chunk_(scan_chunk(n)), k_(std::max<std::size_t>(64, std::bit_ceil((n + 1) / 2))), arena_(words(chunk_, k_)),
          transform_(arena_, std::countr_zero(2 * k_)) {
        const std::size_t k = k_, h = k / 2;
        a_ = arena_.take(32 * chunk_ + 16).data();
        c_ = arena_.take(half_scan(k)).data();
        s_ = arena_.take(half_scan(k)).data();
        sigma_ = arena_.take(h);
        scratch_ = arena_.take(poly::inverse_scratch(h));
        st_ = arena_.take(k);
        e_ = arena_.take(k);
        qt_ = arena_.take(k);
        w_ = arena_.take(k);
        et_ = arena_.take(2 * k);
        arena_.take(kPadding);
        pe_ = arena_.take(2 * k);
        arena_.take(kPadding);
        po_ = arena_.take(2 * k);
        text_ = reinterpret_cast<char*>(arena_.take(kTextWords).data());
    }

    std::uint32_t* coefficients() { return a_; }  // room for f: n values < P, zero after them
    char* text() { return text_; }                // fields::kTextBytes bytes, 16-byte aligned

    // g_0 .. g_n, canonical, in place of f.
    const std::uint32_t* run() {
        lane_factorials();
        inverse_factorials();
        bernoulli();
        weights();
        const poly::Transform& t = transform_;
        t.forward(e_, 0, et_);
        t.cyclic_product(pe_.first(k_), 0, pe_, et_, Half::kLower);
        t.cyclic_product(po_.first(k_), 0, po_, et_, Half::kLower);
        output();
        return a_;
    }

private:
    // Words before pe and po: the scans reach 512 words below them (16 C <= K + 512).
    static constexpr std::size_t kPadding = 1024;
    static constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);

    // Words of c and s: the scan over 2K positions writes 16 C' + 4 of each.
    static std::size_t half_scan(std::size_t k) { return 16 * scan_chunk(2 * k) + 8; }

    static std::size_t words(std::size_t chunk, std::size_t k) {
        using poly::Arena;
        return poly::Transform::words(std::countr_zero(2 * k)) + Arena::footprint(32 * chunk + 16) +
               2 * Arena::footprint(half_scan(k)) + Arena::footprint(k / 2) + poly::inverse_scratch(k / 2) +
               4 * Arena::footprint(k) + 3 * Arena::footprint(2 * k) + 2 * Arena::footprint(kPadding) +
               Arena::footprint(kTextWords);
    }

    // factorial_[s] = (s C)!, inverse_factorial_[s] = 1 / (s C)! for s <= 32.
    void lane_factorials() {
        Lanes top;
        for (int s = 0; s < 32; ++s) top[s] = std::uint32_t((s + 1) * chunk_);
        const Lanes f = poly::factorials(top);
        Lanes inverse = f;
        poly::invert(inverse);
        factorial_[0] = inverse_factorial_[0] = 1;
        std::copy(f.begin(), f.end(), factorial_.begin() + 1);
        std::copy(inverse.begin(), inverse.end(), inverse_factorial_.begin() + 1);
    }

    // c[j] = 1 / (2j)!, s[j] = 1 / (2j + 1)! for j < 16 C', C' = scan_chunk(2K): a reversed chain
    // of 1 / i! from 1 / top! at each lane's top, multipliers i.
    void inverse_factorials() {
        const std::size_t chunk = scan_chunk(2 * k_);
        Lanes top, base;
        for (int s = 0; s < 32; ++s) top[s] = std::uint32_t((s + 1) * chunk - 1), base[s] = mont(top[s]);
        Lanes start = poly::factorials(top);
        poly::invert(start);
        Chain<true> chain(start, base, kP - kR);
        std::uint32_t *const even = c_, *const odd = s_;
        const Vec order = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        scan(chunk, [=](std::size_t j, int s, Vec x) {
            const std::size_t i = ((s + 1) * chunk - 8 - j) / 2;
            store_halves(even + i, odd + i, _mm256_permutevar8x32_epi32(reduce(x, kP), order));
        }, chain);
    }

    // e = E mod w^K = C / S by one division step from sigma = 1 / S mod w^h, h = K / 2.
    void bernoulli() {
        const poly::Transform& t = transform_;
        const std::size_t k = k_, h = k / 2;
        const std::span<const std::uint32_t> c(c_, k), s(s_, k);
        poly::inverse(t, s.first(h), sigma_, scratch_);
        t.forward(sigma_, 0, st_);
        t.cyclic_product(c.first(h), 0, e_, st_, Half::kLower);  // e[0, h) = q0
        t.forward(e_.first(h), 0, qt_);
        t.cyclic_product(s, 0, w_, qt_, Half::kUpper);  // w[h, K) = (S q0)[h, K)
        for (std::size_t i = h; i < k; i += 8)
            store_unaligned(w_.data() + i, difference(load(w_.data() + i), load_unaligned(c_ + i)));  // -r
        t.cyclic_product(w_.subspan(h), h, w_, st_, Half::kUpper, kP - 1);  // w[h, K) = r sigma mod w^h
        std::copy(w_.begin() + std::ptrdiff_t(h), w_.end(), e_.begin() + std::ptrdiff_t(h));
    }

    // abar_t = a_t t! / 2^t into pe and po: a chain of t! / 2^t 2^32 from each lane's start s C,
    // multipliers (t + 1) / 2. Positions past n read zeros.
    void weights() {
        Lanes start, base;
        for (int s = 0; s < 32; ++s) {
            const std::uint32_t first = std::uint32_t(s * chunk_);
            start[s] = mont(multiply_mod(factorial_[s], power(kHalf, first)));
            base[s] = mont(multiply_mod(first + 1, kHalf));
        }
        Chain<> chain(start, base, mont(kHalf));
        const std::uint32_t* const a = a_;
        std::uint32_t *const pe = pe_.data() + k_ - 4, *const po = po_.data() + k_ - 4;
        const std::size_t chunk = chunk_;
        const Vec order = _mm256_setr_epi32(6, 4, 2, 0, 7, 5, 3, 1);
        scan(chunk, [=](std::size_t j, int s, Vec x) {
            const std::size_t t = s * chunk + j;
            const std::ptrdiff_t i = -std::ptrdiff_t(t / 2);
            const Vec abar = reduce(montgomery(load(a + t), x), kP);
            store_halves(pe + i, po + i, _mm256_permutevar8x32_epi32(abar, order));
        }, chain);
    }

    // g_(k+1) = z_k sum_k - a_(k+1) / 2 in place of a_(k+1), z_k = 2^k / (k+1)!: a reversed chain
    // of z_k 2^32 from each lane's top, multipliers (k + 1) / 2. Then g_0 = 0.
    void output() {
        Lanes start, base;
        for (int s = 0; s < 32; ++s) {
            const std::uint32_t top = std::uint32_t((s + 1) * chunk_ - 1);
            start[s] = mont(multiply_mod(power(2, top), inverse_factorial_[s + 1]));
            base[s] = mont(multiply_mod(top + 1, kHalf));
        }
        Chain<true> chain(start, base, kP - mont(kHalf));
        std::uint32_t* const a = a_;
        const std::uint32_t *const pe = pe_.data() + k_ - 4, *const po = po_.data() + k_ - 4;
        const std::size_t chunk = chunk_;
        const Vec order = _mm256_setr_epi32(3, 7, 2, 6, 1, 5, 0, 4);
        scan(chunk, [=](std::size_t j, int s, Vec z) {
            const std::size_t k = (s + 1) * chunk - 8 - j;
            const std::ptrdiff_t i = -std::ptrdiff_t(k / 2);
            const Vec sum = _mm256_permutevar8x32_epi32(load_halves(pe + i, po + i), order);
            const Vec next = load_unaligned(a + k + 1);
            store_unaligned(a + k + 1, difference(reduce(montgomery(sum, z), kP), halve(next)));
        }, chain);
        a_[0] = 0;
    }

    std::size_t chunk_, k_;
    poly::Arena arena_;
    poly::Transform transform_;
    std::uint32_t *a_, *c_, *s_;
    std::span<std::uint32_t> sigma_, scratch_, st_, e_, qt_, w_, et_, pe_, po_;
    char* text_;
    std::array<std::uint32_t, 33> factorial_, inverse_factorial_;
};

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    PrefixSum prefix(n);
    io::read_bulk(in, prefix.coefficients(), n);
    io::Writer out;
    fields::write(out, prefix.run(), n + 1, prefix.text());
}

}  // namespace

RUN_EARLY(solve)
