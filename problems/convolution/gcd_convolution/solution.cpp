// c_k = sum over gcd(i, j) = k of a_i b_j mod 998244353, 1 <= k <= N <= 10^6.
// Sums over multiples (zeta) of a and b, pointwise product, then the inverse (Moebius). Both are
// products of commuting per-prime passes. Primes 2..13: one pass each (zeta: x_i += x_ip, i
// descending; Moebius: x_i -= x_ip, i ascending). Primes >= 17 together: one sweep over the
// multipliers m coprime to 30030 ("rough"), in two stages (prime factors below and above 100),
// segment by segment so the sources stay in L2 (in L1 for m < 256).
// a and b are interleaved as pairs, so one load fetches both.
#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/io/sequential.hpp"
#include "lib/mem/huge.hpp"
#include "lib/run/early.hpp"
#include "../convolution_mod/fields.hpp"

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
using AliasWord [[gnu::may_alias]] = std::uint32_t;

Vec broadcast(std::uint32_t x) { return _mm256_set1_epi32(int(x)); }
Vec load(const void* p) { return _mm256_loadu_si256(static_cast<const Vec*>(p)); }
void store(void* p, Vec x) { _mm256_storeu_si256(static_cast<Vec*>(p), x); }
Half load_pair(const std::uint64_t* p) { return _mm_loadl_epi64(reinterpret_cast<const Half*>(p)); }
void store_pair(std::uint64_t* p, Half x) { _mm_storel_epi64(reinterpret_cast<Half*>(p), x); }

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
std::uint32_t add(std::uint32_t x, std::uint32_t y) {
    const std::uint32_t s = x + y;
    return std::min(s, s - kP);
}
std::uint32_t sub(std::uint32_t x, std::uint32_t y) {
    const std::uint32_t d = x - y;
    return std::min(d, d + kP);
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

// Pairs 0, s, 2s, 3s at p.
Vec gather_pairs(const std::uint64_t* p, std::size_t s) {
    return _mm256_set_epi64x(std::int64_t(p[3 * s]), std::int64_t(p[2 * s]), std::int64_t(p[s]), std::int64_t(p[0]));
}

// Dwords 0, s, ..., 7s at p. Scalar loads: on Zen 3 vpgatherdd costs 1.38 cycles per element,
// a load 0.9 (AGENTS.md).
Vec gather_dwords(const std::uint32_t* p, std::size_t s) {
    return _mm256_setr_epi32(int(p[0]), int(p[s]), int(p[2 * s]), int(p[3 * s]), int(p[4 * s]), int(p[5 * s]),
                             int(p[6 * s]), int(p[7 * s]));
}

void add_pair(std::uint64_t* dst, const std::uint64_t* src) { store_pair(dst, add(load_pair(dst), load_pair(src))); }

// Adds 64-bit lane sums [a, b] to the pair at dst.
void add_sums(std::uint64_t* dst, Half sums) {
    const auto a = std::uint32_t(std::uint64_t(_mm_cvtsi128_si64(sums)) % kP);
    const auto b = std::uint32_t(std::uint64_t(_mm_extract_epi64(sums, 1)) % kP);
    store_pair(dst, add(load_pair(dst), _mm_setr_epi32(int(a), int(b), 0, 0)));
}

// The pair at p as two 64-bit lanes.
Half widen_pair(const std::uint64_t* p) { return _mm_cvtepu32_epi64(load_pair(p)); }

// The rough numbers (coprime to 30030 = 2 3 5 7 11 13) as a wheel: the one with index t is
// (t / kSpokes) kWheel + spoke[t % kSpokes]. Index 0 is 1, index 1 is 17.
constexpr std::uint32_t kWheel = 30030, kSpokes = 5760;

struct Wheel {
    std::uint16_t spoke[kSpokes];
    std::uint16_t rank[kWheel];  // spokes below r

    constexpr Wheel() : spoke(), rank() {
        bool shared[kWheel] = {};
        for (const std::uint32_t p : {2, 3, 5, 7, 11, 13})
            for (std::uint32_t r = 0; r < kWheel; r += p) shared[r] = true;
        std::uint16_t count = 0;
        for (std::uint32_t r = 0; r < kWheel; ++r) {
            rank[r] = count;
            if (!shared[r]) spoke[count++] = std::uint16_t(r);
        }
    }

    // Index of the first rough number >= x.
    constexpr std::uint32_t index(std::uint32_t x) const { return x / kWheel * kSpokes + rank[x % kWheel]; }
    constexpr std::uint32_t value(std::uint32_t t) const { return t / kSpokes * kWheel + spoke[t % kSpokes]; }
};

constexpr Wheel kRough;

// floor(x / d) = (x r) >> 42 with r = reciprocal(d), for x, d < 2^20.
constexpr std::uint64_t reciprocal(std::uint32_t d) { return (std::uint64_t(1) << 42) / d + 1; }
inline std::uint32_t divide(std::uint32_t x, std::uint64_t r) { return std::uint32_t(x * r >> 42); }

// The rough sweeps run in two stages (as ../lcm_convolution), which cuts contributions from 2.14 N
// to 1.77 N. Stage 1: m > 1 with all prime factors in [17, 100); stage 2: m > 1 with all prime
// factors above 100 (for m <= 10^6, primes and products of two primes). Rough m with prime factors
// on both sides come from composing the two.
constexpr std::uint32_t kStagePrimes[] = {17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
constexpr std::uint32_t kStage2Min = 101;  // smallest stage-2 multiplier

// Multipliers m <= kSplit[stage] go by m over a run of targets; larger ones by target
// i <= n / (kSplit[stage] + 1) into 64-bit sums (stage 1 has few m above 2048, so it goes by m
// further up). Tiny m < kTinyBound take their sources by L1-sized pieces of each segment.
constexpr std::uint32_t kSplit[3] = {0, 16384, 2048}, kTinyBound = 256;  // [0]: unused
// Stage-2 large m of targets i <= kPieceTargets (about half of the large contributions) also take
// their sources by those pieces, before the tiny m, so that the piece is in L1 for both.
constexpr std::uint32_t kPieceTargets = 8;
constexpr std::uint32_t kMaxN = 1000000;
constexpr std::uint32_t kMaxTargets = kMaxN / (kSplit[2] + 1) + 1;
constexpr std::uint32_t kEnd = 1 << 20;  // above kMaxN; kEnd i < 2^32 for i < 4096
constexpr std::uint32_t kMaxSweepTarget = (1 << 16) / 17;  // largest i in a sum over m, segment 0
static_assert(std::uint64_t(kEnd) * kMaxSweepTarget < (std::uint64_t(1) << 32));

// The stage of each rough m <= kMaxN, by wheel index: 1 (only primes below 100), 2 (none below
// 100), 0 otherwise. Used only at compile time.
struct Stages {
    static constexpr std::uint32_t kTotal = kRough.index(kMaxN + 1);
    std::uint8_t of[kTotal];
    std::uint32_t small[3], large[3];  // counts per stage, m <= kSplit[stage] and above

    constexpr Stages() : of(), small(), large() {
        for (std::uint32_t t = 1; t < kTotal; ++t) of[t] = 2;
        for (const std::uint32_t p : kStagePrimes)
            for (std::uint32_t t = 0; kRough.value(t) <= kMaxN / p; ++t) of[kRough.index(p * kRough.value(t))] = 0;
        mark_smooth(1, 0);
        for (std::uint32_t t = 1; t < kTotal; ++t) ++(kRough.value(t) <= kSplit[of[t]] ? small : large)[of[t]];
    }

    // of[m w] = 1 for each m w <= kMaxN with w > 1 made of kStagePrimes[k..].
    constexpr void mark_smooth(std::uint32_t m, std::size_t k) {
        for (; k < std::size(kStagePrimes) && kStagePrimes[k] <= kMaxN / m; ++k)
            for (std::uint32_t w = m * kStagePrimes[k];; w *= kStagePrimes[k]) {
                of[kRough.index(w)] = 1;
                mark_smooth(w, k + 1);
                if (w > kMaxN / kStagePrimes[k]) break;
            }
    }
};

constexpr Stages kStages;

// One stage's multipliers up to kMaxN: small ones (m <= kSplit[kStage]) with their reciprocals, the first
// kTiny below kTinyBound; large ones framed by sentinels: large[0, 2) = 0, large[2, kLarge + 2)
// ascending, large[kLarge + 2, kLarge + 4) = kEnd.
template <std::uint32_t kStage>
struct Multipliers {
    static constexpr std::uint32_t kSmall = kStages.small[kStage], kLarge = kStages.large[kStage];
    std::uint32_t small[kSmall];
    std::uint64_t reciprocal[kSmall];
    std::uint32_t large[kLarge + 4];
    std::uint32_t tiny;

    constexpr Multipliers() : small(), reciprocal(), large(), tiny() {
        std::uint32_t s = 0, l = 2;
        for (std::uint32_t t = 1; t < Stages::kTotal; ++t) {
            const std::uint32_t m = kRough.value(t);
            if (kStages.of[t] != kStage) continue;
            if (m <= kSplit[kStage]) {
                tiny += m < kTinyBound;
                small[s] = m;
                reciprocal[s++] = ::reciprocal(m);
            } else {
                large[l++] = m;
            }
        }
        large[l] = large[l + 1] = kEnd;
    }

    // Index of the first large m with i m > bound.
    std::uint32_t large_after(std::uint32_t i, std::uint32_t bound) const {
        return std::uint32_t(std::upper_bound(large + 2, large + kLarge + 2, bound / i) - large);
    }
};

template <std::uint32_t kStage>
constexpr Multipliers<kStage> kMultipliers;

// Sum of term(i m) over the large m of a stage from index k on while i m <= last, in two chains;
// k moves past them.
template <std::uint32_t kStage, class T, class Term>
[[gnu::always_inline]] inline T sum_large(std::uint32_t i, std::uint32_t& k, std::uint32_t last, T sum, Term term) {
    const std::uint32_t* const large = kMultipliers<kStage>.large;
    T other{};
    for (; large[k + 1] * i <= last; k += 2) {
        sum += term(large[k] * i);
        other += term(large[k + 1] * i);
    }
    if (large[k] * i <= last) sum += term(large[k++] * i);
    return sum + other;
}

// Sum of term(i m) over all m of a stage with i m <= last (i <= kMaxSweepTarget).
template <std::uint32_t kStage, class T, class Term>
[[gnu::always_inline]] inline T sum_multiples(std::uint32_t i, std::uint32_t last, Term term) {
    const auto& s = kMultipliers<kStage>;
    constexpr std::uint32_t kSmall = Multipliers<kStage>::kSmall;
    T sum{}, other{};
    std::uint32_t k = 0;
    for (; k + 1 < kSmall && s.small[k + 1] * i <= last; k += 2) {
        sum += term(s.small[k] * i);
        other += term(s.small[k + 1] * i);
    }
    if (k < kSmall && s.small[k] * i <= last) sum += term(s.small[k++] * i);
    if (k < kSmall) return sum + other;
    k = 2;
    return sum_large<kStage>(i, k, last, sum + other, term);
}

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
void zeta_pass(std::uint64_t* pairs, std::uint32_t n, std::uint32_t p) {
    std::uint32_t i = n / p + 1;
    while (i >= 5) {
        i -= 4;
        store(pairs + i, add(load(pairs + i), gather_pairs(pairs + p * i, p)));
    }
    while (i > 1) {
        --i;
        add_pair(pairs + i, pairs + p * i);
    }
}

// One stage of the zeta sweep, sources ascending by segment, and the state carried between
// segments. Targets go to dst: the pairs (stage 1) or the stage-2 sums t2.
template <std::uint32_t kStage>
struct ZetaSweep {
    using M = Multipliers<kStage>;
    static constexpr const M& s = kMultipliers<kStage>;
    std::uint32_t next_target[M::kSmall];  // per small m
    std::uint32_t next_index[kMaxTargets];  // per large target: into s.large
    Half sums[kMaxTargets];                 // per large target: below 2^18 terms of 2^30 per lane

    // Sources from first on.
    void init(std::uint32_t first, std::uint32_t n) {
        for (std::uint32_t k = 0; k < M::kSmall; ++k) next_target[k] = (first - 1) / s.small[k] + 1;
        for (std::uint32_t i = 1; i * (kSplit[kStage] + 1) <= n; ++i) {
            next_index[i] = s.large_after(i, first - 1);
            sums[i] = Half{};
        }
    }

    // Small m [0, k_end) with sources up to last_source: dst[i] += pairs[i m].
    void by_multiplier(std::uint64_t* dst, const std::uint64_t* pairs, std::uint32_t k_end, std::uint32_t last_source) {
        for (std::uint32_t k = 0; k < k_end; ++k) {
            const std::uint32_t m = s.small[k], last = divide(last_source, s.reciprocal[k]);
            std::uint32_t i = next_target[k];
            for (; i + 3 <= last; i += 4) store(dst + i, add(load(dst + i), gather_pairs(pairs + i * m, m)));
            for (; i <= last; ++i) add_pair(dst + i, pairs + i * m);
            next_target[k] = i;
        }
    }

    // Large m with sources up to last_source, by target, for targets up to last_target.
    void by_target(const std::uint64_t* pairs, std::uint32_t last_source, std::uint32_t last_target) {
        auto widened = [&](std::uint32_t j) { return widen_pair(pairs + j); };
        last_target = std::min(last_target, last_source / (kSplit[kStage] + 1));
        for (std::uint32_t i = 1; i <= last_target; ++i)
            sums[i] = sum_large<kStage>(i, next_index[i], last_source, sums[i], widened);
    }

    void finish(std::uint64_t* dst, std::uint32_t n) const {
        for (std::uint32_t i = 1; i * (kSplit[kStage] + 1) <= n; ++i) add_sums(dst + i, sums[i]);
    }
};

// pairs[i] += sum of pairs[i m] over rough m > 1, i m <= n: the zeta passes of all primes >= 17,
// as stage 1 after stage 2. Stage 2 changes only i <= n / 101, all in segment 0; its sums go to t2
// (n / 101 + 1 pairs). Segment 0 goes first, by target ascending, so its sources are unchanged
// when read; then source segments ascend (their targets lie in earlier segments). Last, stage 1
// on t2, which is then added to the pairs.
void zeta_rough(std::uint64_t* pairs, std::uint32_t n, std::uint64_t* t2) {
    constexpr std::uint32_t kSegment = 1 << 15, kPiece = 1 << 12;  // pairs: 256 KiB, 32 KiB
    const std::uint32_t last2 = n / kStage2Min;
    std::memset(t2, 0, (last2 + 1) * sizeof(std::uint64_t));
    auto widened = [&](std::uint32_t j) { return widen_pair(pairs + j); };
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    for (std::uint32_t i = 1; kStagePrimes[0] * i <= last0; ++i) {
        add_sums(pairs + i, sum_multiples<1, Half>(i, last0, widened));
        if (kStage2Min * i <= last0) add_sums(t2 + i, sum_multiples<2, Half>(i, last0, widened));
    }
    if (n >= kSegment) {
        static ZetaSweep<1> one;
        static ZetaSweep<2> two;
        one.init(kSegment, n);
        two.init(kSegment, n);
        for (std::uint32_t start = kSegment; start <= n; start += kSegment) {
            const std::uint32_t last_source = std::min(n, start + kSegment - 1);
            for (std::uint32_t piece = start + kPiece; piece <= last_source; piece += kPiece) {
                two.by_target(pairs, piece - 1, kPieceTargets);
                one.by_multiplier(pairs, pairs, one.s.tiny, piece - 1);
                two.by_multiplier(t2, pairs, two.s.tiny, piece - 1);
            }
            one.by_multiplier(pairs, pairs, ZetaSweep<1>::M::kSmall, last_source);
            two.by_multiplier(t2, pairs, ZetaSweep<2>::M::kSmall, last_source);
            one.by_target(pairs, last_source, kMaxTargets);
            two.by_target(pairs, last_source, kMaxTargets);
        }
        one.finish(pairs, n);
        two.finish(t2, n);
    }
    auto from_t2 = [&](std::uint32_t j) { return widen_pair(t2 + j); };
    for (std::uint32_t i = 1; kStagePrimes[0] * i <= last2; ++i) add_sums(t2 + i, sum_multiples<1, Half>(i, last2, from_t2));
    for (std::uint32_t i = 1; i <= last2; ++i) add_pair(pairs + i, t2 + i);
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
void moebius_pass(std::uint32_t* c, std::uint32_t n, std::uint32_t p) {
    const std::uint32_t last = n / p;
    std::uint32_t i = 1;
    for (; i < 4 && i <= last; ++i) c[i] = sub(c[i], c[i * p]);
    for (; i + 7 <= last; i += 8) store(c + i, sub(load(c + i), gather_dwords(c + i * p, p)));
    for (; i <= last; ++i) c[i] = sub(c[i], c[i * p]);
}

// One stage of the Moebius sweep, sources descending by segment, and the state carried between
// segments. Targets go to dst: c (stage 1) or the stage-2 sums b2.
template <std::uint32_t kStage>
struct MoebiusSweep {
    using M = Multipliers<kStage>;
    static constexpr const M& s = kMultipliers<kStage>;
    std::uint32_t last_target[M::kSmall];  // per small m
    std::uint32_t end_index[kMaxTargets];   // per large target: into s.large
    std::uint64_t sums[kMaxTargets];        // per large target: below 2^18 terms of 2^30

    void init(std::uint32_t n) {
        for (std::uint32_t k = 0; k < M::kSmall; ++k) last_target[k] = n / s.small[k];
        for (std::uint32_t i = 1; i * (kSplit[kStage] + 1) <= n; ++i) {
            end_index[i] = s.large_after(i, n);
            sums[i] = 0;
        }
    }

    // Small m [0, k_end) with sources from first_source on: dst[i] -= c[i m].
    void by_multiplier(std::uint32_t* dst, const std::uint32_t* c, std::uint32_t k_end, std::uint32_t first_source) {
        for (std::uint32_t k = 0; k < k_end; ++k) {
            const std::uint32_t m = s.small[k], last = last_target[k];
            std::uint32_t i = divide(first_source - 1, s.reciprocal[k]) + 1;
            last_target[k] = i - 1;
            for (; i + 7 <= last; i += 8) store(dst + i, sub(load(dst + i), gather_dwords(c + i * m, m)));
            for (; i <= last; ++i) dst[i] = sub(dst[i], c[i * m]);
        }
    }

    // Large m with sources from start on (up to last_source), by target, descending, for targets
    // up to last_target.
    void by_target(const std::uint32_t* c, std::uint32_t start, std::uint32_t last_source, std::uint32_t last_target) {
        const std::uint32_t* const large = s.large;
        last_target = std::min(last_target, last_source / (kSplit[kStage] + 1));
        for (std::uint32_t i = 1; i <= last_target; ++i) {
            std::uint32_t k = end_index[i];
            std::uint64_t sum = sums[i], other = 0;
            for (; large[k - 2] * i >= start; k -= 2) {
                sum += c[large[k - 1] * i];
                other += c[large[k - 2] * i];
            }
            if (large[k - 1] * i >= start) sum += c[large[--k] * i];
            sums[i] = sum + other;
            end_index[i] = k;
        }
    }

    void finish(std::uint32_t* dst, std::uint32_t n) const {
        for (std::uint32_t i = 1; i * (kSplit[kStage] + 1) <= n; ++i) dst[i] = sub(dst[i], std::uint32_t(sums[i] % kP));
    }
};

// c_i -= sum of final c_im over rough m > 1, im <= n: the Moebius passes of all primes >= 17
// (inverting zeta_rough: u = stage-1 inverse of c, then c_i = u_i - sum of final c_im over stage-2
// m, which changes only i <= n / 101, in segment 0). Source segments descend, so a segment is
// final when read; stage-2 terms from them go to b2 (n / 101 + 1 words, negated). Segment 0 goes
// last by target, descending: stage 1, then stage 2.
void moebius_rough(std::uint32_t* c, std::uint32_t n, std::uint32_t* b2) {
    constexpr std::uint32_t kSegment = 1 << 16, kPiece = 1 << 13;  // 256 KiB, 32 KiB
    const std::uint32_t last2 = n / kStage2Min;
    std::memset(b2, 0, (last2 + 1) * sizeof(std::uint32_t));
    if (n >= kSegment) {
        static MoebiusSweep<1> one;
        static MoebiusSweep<2> two;
        one.init(n);
        two.init(n);
        for (std::uint32_t start = n / kSegment * kSegment; start >= kSegment; start -= kSegment) {
            const std::uint32_t last_source = std::min(n, start + kSegment - 1);
            for (std::uint32_t piece = start + (last_source - start) / kPiece * kPiece; piece > start; piece -= kPiece) {
                two.by_target(c, piece, last_source, kPieceTargets);
                one.by_multiplier(c, c, one.s.tiny, piece);
                two.by_multiplier(b2, c, two.s.tiny, piece);
            }
            one.by_multiplier(c, c, MoebiusSweep<1>::M::kSmall, start);
            two.by_multiplier(b2, c, MoebiusSweep<2>::M::kSmall, start);
            one.by_target(c, start, last_source, kMaxTargets);
            two.by_target(c, start, last_source, kMaxTargets);
        }
        one.finish(c, n);
        two.finish(b2, n);
    }
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    auto value = [&](std::uint32_t j) { return std::uint64_t(c[j]); };
    for (std::uint32_t i = last0 / kStagePrimes[0]; i >= 1; --i)
        c[i] = sub(c[i], std::uint32_t(sum_multiples<1, std::uint64_t>(i, last0, value) % kP));
    for (std::uint32_t i = last2; i >= 1; --i) {
        std::uint32_t x = add(c[i], b2[i]);
        if (kStage2Min * i <= last0) x = sub(x, std::uint32_t(sum_multiples<2, std::uint64_t>(i, last0, value) % kP));
        c[i] = x;
    }
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    io::advise_sequential(in);
    // One region: the pairs, with a parsed into their first half; then b, later c; then the stage-2
    // sums of the rough sweeps (t2: pairs, b2: words). The pairs' first huge page later holds the
    // output text.
    constexpr std::size_t kPad = 64;
    const std::size_t words = (n + kPad + 1) / 2 * 2, sums = n / kStage2Min + 1;
    static_assert(fields::kTextBytes <= std::size_t(1) << 21);
    auto* region = mem::huge<std::uint32_t>(std::max(3 * (words + sums), fields::kTextBytes / sizeof(std::uint32_t)));
    auto* pairs = reinterpret_cast<std::uint64_t*>(region);
    std::uint32_t* const b = region + 2 * words;
    auto* const t2 = reinterpret_cast<std::uint64_t*>(region + 3 * words);
    std::uint32_t* const b2 = region + 3 * words + 2 * sums;
    io::read_bulk(in, region + 1, n);
    io::read_bulk(in, b + 1, n);

    interleave_zeta3(region, b, pairs, n);
    for (const std::uint32_t p : {5, 7, 11, 13}) zeta_pass(pairs, n, p);
    zeta_rough(pairs, n, t2);
    std::uint32_t* const c = b;
    zeta2_product_moebius2(pairs, c, n);
    for (const std::uint32_t p : {3, 5, 7, 11, 13}) moebius_pass(c, n, p);
    moebius_rough(c, n, b2);

    io::Writer out;
    fields::write(out, c + 1, n, reinterpret_cast<char*>(region));
}

}  // namespace

RUN_EARLY(solve)
