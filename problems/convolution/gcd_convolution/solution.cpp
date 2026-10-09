// c_k = sum over gcd(i, j) = k of a_i b_j mod 998244353, 1 <= k <= N <= 10^6.
// Sums over multiples (zeta) of a and b, pointwise product, then the inverse (Moebius), one pass
// per prime p (pass order is free). Zeta: a_i += a_ip for i descending; Moebius: c_i -= c_ip for i
// ascending. a and b are interleaved as pairs, so a scattered read fetches both.
#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "lib/io/io.hpp"
#include "../fixed_width.hpp"

namespace {

constexpr std::uint32_t kP = 998244353;
constexpr std::uint32_t kNegInverse = [] {  // -1/P mod 2^32
    std::uint32_t x = kP;                     // P^-1 mod 2^3; each step doubles the bits
    for (int i = 0; i < 4; ++i) x *= 2 - kP * x;
    return 0 - x;
}();
constexpr std::uint32_t kR2 = std::uint32_t((std::uint64_t(1) << 32) % kP * ((std::uint64_t(1) << 32) % kP) % kP);

using Vec = __m256i;
using Half = __m128i;

Vec broadcast(std::uint32_t x) { return _mm256_set1_epi32(int(x)); }
Vec load(const void* p) { return _mm256_loadu_si256(static_cast<const Vec*>(p)); }
void store(void* p, Vec x) { _mm256_storeu_si256(static_cast<Vec*>(p), x); }

// Per dword, for x, y < P.
Vec add(Vec x, Vec y) {
    const Vec s = _mm256_add_epi32(x, y);
    return _mm256_min_epu32(s, _mm256_sub_epi32(s, broadcast(kP)));
}
Vec sub(Vec x, Vec y) {
    const Vec d = _mm256_sub_epi32(x, y);
    return _mm256_min_epu32(d, _mm256_add_epi32(d, broadcast(kP)));
}
Half add(Half x, Half y) {
    const Half s = _mm_add_epi32(x, y);
    return _mm_min_epu32(s, _mm_sub_epi32(s, _mm_set1_epi32(int(kP))));
}

// Montgomery: for 64-bit lanes t < 2^62, (t + m P) / 2^32 = t / 2^32 mod P in the high dword,
// below t / 2^32 + P.
Vec redc(Vec t) {
    const Vec m = _mm256_mul_epu32(t, broadcast(kNegInverse));
    return _mm256_add_epi64(t, _mm256_mul_epu32(m, broadcast(kP)));
}

// Pairs [a | b R] (R = 2^32) -> a b mod P in the high dword of each 64-bit lane, below 1.25 P.
Vec product4(Vec pairs) { return redc(_mm256_mul_epu32(pairs, _mm256_srli_epi64(pairs, 32))); }

// The high dwords of x and y: [x0 x1 x2 x3 y0 y1 y2 y3], reduced from below 2P.
Vec high_dwords(Vec x, Vec y) {
    const Vec v = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), 0xDD));
    const Vec ordered = _mm256_permute4x64_epi64(v, 0xD8);
    return _mm256_min_epu32(ordered, _mm256_sub_epi32(ordered, broadcast(kP)));
}

// Pairs 0, 2, 4, 6 of the 8 at p.
Vec even_pairs(const std::uint64_t* p) {
    return _mm256_permute4x64_epi64(_mm256_unpacklo_epi64(load(p), load(p + 4)), 0xD8);
}

Vec gather_pairs(const std::uint64_t* p, std::size_t stride) {
    return _mm256_set_epi64x(std::int64_t(p[3 * stride]), std::int64_t(p[2 * stride]), std::int64_t(p[stride]),
                             std::int64_t(p[0]));
}

void add_pair(std::uint64_t* dst, const std::uint64_t* src) {
    const Half x = _mm_loadl_epi64(reinterpret_cast<const Half*>(dst));
    const Half y = _mm_loadl_epi64(reinterpret_cast<const Half*>(src));
    _mm_storel_epi64(reinterpret_cast<Half*>(dst), add(x, y));
}

std::uint32_t add(std::uint32_t x, std::uint32_t y) {
    const std::uint32_t s = x + y;
    return std::min(s, s - kP);
}
std::uint32_t sub(std::uint32_t x, std::uint32_t y) {
    const std::uint32_t d = x - y;
    return std::min(d, d + kP);
}

// Zeroed memory in 2 MiB pages where the kernel allows. Never freed.
template <class T>
T* allocate(std::size_t count) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    const std::size_t bytes = (count * sizeof(T) + kHuge - 1) / kHuge * kHuge;
    void* p = ::mmap(nullptr, bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) std::abort();
    const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(p) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
    ::madvise(reinterpret_cast<void*>(aligned), bytes, MADV_HUGEPAGE);
#endif
    return reinterpret_cast<T*>(aligned);
}

// Primes up to n <= 2^21, ascending; returns their count. Segmented sieve over the odd numbers
// 2k + 1, one byte each (nonzero: composite). Multiples of 3..13 come from a periodic pattern.
std::uint32_t sieve(std::uint32_t n, std::uint32_t* primes) {
    constexpr std::uint32_t kSegment = 1 << 15, kPeriod = 3 * 5 * 7 * 11 * 13;
    alignas(32) static std::uint8_t pattern[kPeriod + kSegment], segment[kSegment + 32];
    if (n < 2) return 0;
    std::uint32_t count = 0;
    primes[count++] = 2;
    for (const std::uint32_t p : {3, 5, 7, 11, 13}) {
        if (p > n) return count;
        primes[count++] = p;
        for (std::uint32_t k = p / 2; k < kPeriod + kSegment; k += p) pattern[k] = 1;
    }
    // Sieving primes 17 <= p <= sqrt(n), with the index of the next odd multiple to mark.
    std::uint32_t sieving[256], next[256], sieving_count = 0;
    for (std::uint32_t p = 17; p * p <= n; p += 2) {
        bool prime = true;
        for (std::uint32_t d = 3; d * d <= p && prime; d += 2) prime = p % d != 0;
        if (prime) sieving[sieving_count] = p, next[sieving_count++] = p * p / 2;
    }
    const std::uint32_t odds = (n + 1) / 2;  // odd numbers 2k + 1 <= n
    for (std::uint32_t low = 0; low < odds; low += kSegment) {
        const std::uint32_t high = std::min(odds, low + kSegment);
        std::memcpy(segment, pattern + low % kPeriod, kSegment);
        for (std::uint32_t j = 0; j < sieving_count; ++j) {
            std::uint32_t k = next[j];
            for (; k < high; k += sieving[j]) segment[k - low] = 1;
            next[j] = k;
        }
        if (low == 0) segment[0] = 1, segment[1] = segment[2] = segment[3] = segment[5] = segment[6] = 1;  // 1, 3..13
        std::memset(segment + (high - low), 1, 32);
        for (std::uint32_t k = 0; k < high - low; k += 32) {
            const Vec bytes = _mm256_load_si256(reinterpret_cast<const Vec*>(segment + k));
            for (auto mask = std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(bytes, Vec{}))); mask;
                 mask &= mask - 1)
                primes[count++] = 2 * (low + k + std::uint32_t(std::countr_zero(mask))) + 1;
        }
    }
    return count;
}

using AliasWord [[gnu::may_alias]] = std::uint32_t;

// pairs[i] = (a_i, b_i R mod P) for 1 <= i <= n, then the zeta pass of p = 3. a may be the first
// half of pairs: i descends, so pairs[i] only overwrites a_2i and a_2i+1, which are read already.
void interleave_zeta3(const AliasWord* a, const std::uint32_t* b, std::uint64_t* pairs, std::uint32_t n) {
    const Vec r2 = broadcast(kR2);
    // b_i R as Montgomery products of b_i and R^2: dwords [b R] in the high halves.
    auto scaled_pairs = [&](std::uint32_t i, Vec& low, Vec& high) {
        const Vec va = load(a + i), vb = load(b + i);
        const Vec even = redc(_mm256_mul_epu32(vb, r2));
        const Vec odd = redc(_mm256_mul_epu32(_mm256_srli_epi64(vb, 32), r2));
        // even/odd: [_ bR] per qword (below 1.25 P). Reduce, then place below a.
        const Vec bs = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
        const Vec br = _mm256_min_epu32(bs, _mm256_sub_epi32(bs, broadcast(kP)));
        const Vec lo = _mm256_unpacklo_epi32(va, br), hi = _mm256_unpackhi_epi32(va, br);
        low = _mm256_permute2x128_si256(lo, hi, 0x20);
        high = _mm256_permute2x128_si256(lo, hi, 0x31);
    };
    const std::uint32_t third = n / 3;
    std::uint32_t i = n + 1;
    // Targets i > n / 3 only take their own values. Eight at a time, from the top.
    while (i >= third + 9) {
        i -= 8;
        Vec low, high;
        scaled_pairs(i, low, high);
        store(pairs + i, low);
        store(pairs + i + 4, high);
    }
    auto one = [&](std::uint32_t k) {
        const std::uint32_t br = std::uint32_t((std::uint64_t(b[k]) << 32) % kP);
        const std::uint64_t pair = a[k] | std::uint64_t(br) << 32;
        std::memcpy(pairs + k, &pair, sizeof pair);
        if (k <= third) add_pair(pairs + k, pairs + 3 * k);
    };
    while (i > third + 1) one(--i);
    while (i >= 8 + 4) {  // sources 3i > i + 7
        i -= 8;
        Vec low, high;
        scaled_pairs(i, low, high);
        store(pairs + i, add(low, gather_pairs(pairs + 3 * i, 3)));
        store(pairs + i + 4, add(high, gather_pairs(pairs + 3 * (i + 4), 3)));
    }
    while (i > 1) one(--i);
}

// Zeta pass of p >= 5 on the pairs: four targets at a time from the top, then one at a time.
void zeta_pass_vector(std::uint64_t* pairs, std::uint32_t n, std::uint32_t p) {
    std::uint32_t i = n / p + 1;
    while (i >= 5) {
        i -= 4;
        store(pairs + i, add(load(pairs + i), gather_pairs(pairs + std::size_t(p) * i, p)));
    }
    while (i > 1) {
        --i;
        add_pair(pairs + i, pairs + std::size_t(p) * i);
    }
}

void zeta_pass(std::uint64_t* pairs, std::uint32_t n, std::uint32_t p) {
    for (std::uint32_t i = n / p; i >= 1; --i) add_pair(pairs + i, pairs + std::size_t(i) * p);
}

// Zeta of every prime p with p^2 > n at once: no source ip is a target of another such prime.
void zeta_large(std::uint64_t* pairs, std::uint32_t n, const std::uint32_t* primes, std::uint32_t first,
                std::uint32_t count) {
    std::uint32_t end = count;
    for (std::uint32_t i = 1;; ++i) {
        const std::uint32_t limit = n / i;
        while (end > first && primes[end - 1] > limit) --end;
        if (end == first) break;
        // 64-bit lane sums of [a, bR]: fewer than 2^17 values below 2^30 each.
        Half s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
        std::uint32_t k = first;
        for (; k + 2 <= end; k += 2) {
            s0 = _mm_add_epi64(s0, _mm_cvtepu32_epi64(_mm_loadl_epi64(reinterpret_cast<const Half*>(pairs + std::size_t(i) * primes[k]))));
            s1 = _mm_add_epi64(s1, _mm_cvtepu32_epi64(_mm_loadl_epi64(reinterpret_cast<const Half*>(pairs + std::size_t(i) * primes[k + 1]))));
        }
        if (k < end)
            s0 = _mm_add_epi64(s0, _mm_cvtepu32_epi64(_mm_loadl_epi64(reinterpret_cast<const Half*>(pairs + std::size_t(i) * primes[k]))));
        const Half s = _mm_add_epi64(s0, s1);
        const std::uint64_t sa = std::uint64_t(_mm_cvtsi128_si64(s)) % kP;
        const std::uint64_t sb = std::uint64_t(_mm_extract_epi64(s, 1)) % kP;
        const std::uint32_t ai = add(std::uint32_t(pairs[i]), std::uint32_t(sa));
        const std::uint32_t bi = add(std::uint32_t(pairs[i] >> 32), std::uint32_t(sb));
        pairs[i] = ai | std::uint64_t(bi) << 32;
    }
}

// Zeta pass of p = 2 fused with the product and the Moebius pass of 2:
// c_i = A_i B_i - A_2i B_2i (the second term for i <= n / 2).
void zeta2_product_moebius2(std::uint64_t* pairs, std::uint32_t* c, std::uint32_t n) {
    const std::uint32_t half = n / 2;
    std::uint32_t i = n + 1;
    while (i >= half + 9) {
        i -= 8;
        store(c + i, high_dwords(product4(load(pairs + i)), product4(load(pairs + i + 4))));
    }
    auto one = [&](std::uint32_t k) {
        std::uint32_t ai = std::uint32_t(pairs[k]), bi = std::uint32_t(pairs[k] >> 32);
        auto product = [](std::uint32_t x, std::uint32_t yr) {
            const std::uint64_t t = std::uint64_t(x) * yr;
            const std::uint32_t m = std::uint32_t(t) * kNegInverse;
            const std::uint32_t r = std::uint32_t((t + std::uint64_t(m) * kP) >> 32);
            return std::min(r, r - kP);
        };
        if (k <= half) {
            const std::uint64_t s = pairs[2 * k];
            ai = add(ai, std::uint32_t(s));
            bi = add(bi, std::uint32_t(s >> 32));
            pairs[k] = ai | std::uint64_t(bi) << 32;
            c[k] = sub(product(ai, bi), product(std::uint32_t(s), std::uint32_t(s >> 32)));
        } else {
            c[k] = product(ai, bi);
        }
    };
    while (i > half + 1) one(--i);
    while (i >= 8 + 8) {  // sources 2i > i + 7
        i -= 8;
        const Vec s0 = even_pairs(pairs + 2 * i), s1 = even_pairs(pairs + 2 * i + 8);
        const Vec t0 = add(load(pairs + i), s0), t1 = add(load(pairs + i + 4), s1);
        store(pairs + i, t0);
        store(pairs + i + 4, t1);
        store(c + i, sub(high_dwords(product4(t0), product4(t1)), high_dwords(product4(s0), product4(s1))));
    }
    while (i > 1) one(--i);
}

// Moebius pass of p on c: eight targets at a time from i = 4 up (sources ip > i + 7 are unchanged).
void moebius_pass_vector(std::uint32_t* c, std::uint32_t n, std::uint32_t p) {
    const std::uint32_t last = n / p;
    std::uint32_t i = 1;
    for (; i < 4 && i <= last; ++i) c[i] = sub(c[i], c[std::size_t(i) * p]);
    const Vec index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), broadcast(p));
    for (; i + 7 <= last; i += 8) {
        const Vec src = _mm256_i32gather_epi32(reinterpret_cast<const int*>(c + std::size_t(i) * p), index, 4);
        store(c + i, sub(load(c + i), src));
    }
    for (; i <= last; ++i) c[i] = sub(c[i], c[std::size_t(i) * p]);
}

void moebius_pass(std::uint32_t* c, std::uint32_t n, std::uint32_t p) {
    const std::uint32_t last = n / p;
    for (std::uint32_t i = 1; i <= last; ++i) c[i] = sub(c[i], c[std::size_t(i) * p]);
}

// Moebius of every prime p with p^2 > n at once.
void moebius_large(std::uint32_t* c, std::uint32_t n, const std::uint32_t* primes, std::uint32_t first,
                   std::uint32_t count) {
    std::uint32_t end = count;
    for (std::uint32_t i = 1;; ++i) {
        const std::uint32_t limit = n / i;
        while (end > first && primes[end - 1] > limit) --end;
        if (end == first) break;
        std::uint64_t s0 = 0, s1 = 0;
        std::uint32_t k = first;
        for (; k + 2 <= end; k += 2) {
            s0 += c[std::size_t(i) * primes[k]];
            s1 += c[std::size_t(i) * primes[k + 1]];
        }
        if (k < end) s0 += c[std::size_t(i) * primes[k]];
        c[i] = sub(c[i], std::uint32_t((s0 + s1) % kP));
    }
}

}  // namespace

int main() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    // One region: the pairs, with a parsed into their first half; then b (later c); then the primes.
    constexpr std::size_t kPad = 64;
    const std::size_t words = n + kPad;
    auto* region = allocate<std::uint32_t>(3 * words + n / 2 + kPad);
    auto* pairs = reinterpret_cast<std::uint64_t*>(region);
    std::uint32_t* const b = region + 2 * words;
    std::uint32_t* const primes = b + words;
    in.read(region + 1, n);
    in.read(b + 1, n);

    const std::uint32_t count = sieve(n, primes);
    // Zeta passes 2 and 3 are fused with other work; primes from index first_large have p^2 > n.
    std::uint32_t first_large = std::min<std::uint32_t>(2, count);
    while (first_large < count && std::uint64_t(primes[first_large]) * primes[first_large] <= n) ++first_large;

    interleave_zeta3(region, b, pairs, n);
    for (std::uint32_t k = 2; k < first_large; ++k) {  // primes[0..1] = 2, 3
        const std::uint32_t p = primes[k];
        if (p <= 7) zeta_pass_vector(pairs, n, p);
        else zeta_pass(pairs, n, p);
    }
    zeta_large(pairs, n, primes, first_large, count);

    std::uint32_t* const c = b;
    zeta2_product_moebius2(pairs, c, n);
    for (std::uint32_t k = 1; k < first_large; ++k) {
        const std::uint32_t p = primes[k];
        if (p <= 13) moebius_pass_vector(c, n, p);
        else moebius_pass(c, n, p);
    }
    moebius_large(c, n, primes, first_large, count);

    io::Writer out;
    fixed_width::write(out, c + 1, n);
}
