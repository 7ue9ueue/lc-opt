// c_k = sum over lcm(i, j) = k of a_i b_j mod 998244353, 1 <= k <= N <= 10^6.
// Sums over divisors (zeta) of a and b, pointwise product, then the inverse (Moebius). Both are
// products of commuting per-prime passes (zeta: x_ip += x_i, i ascending; Moebius: x_ip -= x_i,
// i descending). Zeta: 3, 5 and 7 with the interleave, 2 in the product sweep, all others in one
// sweep, target segment by target segment, in L2, in four stages of multipliers m: made of 11
// and 13, of 17..47, of 53..293, and of primes above 300. Moebius: 2 in the product sweep, 3 and
// 5 by passes, 7, 11 and 13 in one pass, then the sweep with the last three stages.
// a and b are interleaved as pairs, so one load fetches both. Design and measurements: notes.md.
#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "lib/io/io.hpp"
#include "lib/io/sequential.hpp"
#include "lib/mem/huge.hpp"
#include "lib/run/early.hpp"
#include "../convolution_mod/fields.hpp"
#include "chunk_read.hpp"

namespace {

constexpr std::uint32_t kP = 998244353;
constexpr std::uint32_t kNegInverse = [] {  // -1/P mod 2^32
    std::uint32_t x = kP;                     // P^-1 mod 2^3; each step doubles the bits
    for (int i = 0; i < 4; ++i) x *= 2 - kP * x;
    return 0 - x;
}();
static_assert(kP * (0 - kNegInverse) == 1);
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

// a b mod P for a, b R < P.
std::uint32_t product(std::uint32_t a, std::uint32_t br) {
    const std::uint64_t t = std::uint64_t(a) * br;
    const std::uint32_t m = std::uint32_t(t) * kNegInverse;
    const auto r = std::uint32_t((t + std::uint64_t(m) * kP) >> 32);
    return std::min(r, r - kP);
}

// The high dwords of x and y: [x0 x1 x2 x3 y0 y1 y2 y3], reduced from below 2P.
Vec high_dwords(Vec x, Vec y) {
    const Vec v = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), 0xDD));
    const Vec ordered = _mm256_permute4x64_epi64(v, 0xD8);
    return _mm256_min_epu32(ordered, _mm256_sub_epi32(ordered, broadcast(kP)));
}

// The 64-bit lanes of x picked by Lanes, with the lanes outside Keep (a dword blend mask) zero.
template <int Lanes, int Keep>
Vec spread(Vec x) {
    return _mm256_blend_epi32(_mm256_setzero_si256(), _mm256_permute4x64_epi64(x, Lanes), Keep);
}

void add_pair(std::uint64_t* dst, const std::uint64_t* src) { store_pair(dst, add(load_pair(dst), load_pair(src))); }

// x broadcast, in a register: GCC otherwise reloads such constants inside the sweep loops.
Half held(std::uint32_t x) {
    Half v = _mm_set1_epi32(int(x));
    asm("" : "+x"(v));
    return v;
}

// One update x op= y: on pairs (zeta) or dwords (Moebius), values below P.
struct PairAdd {
    using T = std::uint64_t;
    Half neg_p = held(0 - kP);
    static Half load(const T* p) { return load_pair(p); }
    void apply(T* t, Half v) const {
        const Half s = _mm_add_epi32(load_pair(t), v);
        store_pair(t, _mm_min_epu32(s, _mm_add_epi32(s, neg_p)));
    }
};
struct PairSub {
    using T = std::uint64_t;
    Half p = held(kP);
    static Half load(const T* q) { return load_pair(q); }
    void apply(T* t, Half v) const {
        const Half d = _mm_sub_epi32(load_pair(t), v);
        store_pair(t, _mm_min_epu32(d, _mm_add_epi32(d, p)));
    }
};
struct WordSub {
    using T = std::uint32_t;
    Half p = held(kP);
    static Half load(const T* q) { return _mm_loadu_si32(q); }
    void apply(T* t, Half v) const {
        const Half d = _mm_sub_epi32(load(t), v);
        _mm_storeu_si32(t, _mm_min_epu32(d, _mm_add_epi32(d, p)));
    }
};
struct WordAdd {
    using T = std::uint32_t;
    Half neg_p = held(0 - kP);
    static Half load(const T* q) { return _mm_loadu_si32(q); }
    void apply(T* t, Half v) const {
        const Half s = _mm_add_epi32(load(t), v);
        _mm_storeu_si32(t, _mm_min_epu32(s, _mm_add_epi32(s, neg_p)));
    }
};

// target[k stride] op= source[k] for k in [0, count), ascending. Each step loads its sources
// before it updates targets, so in place the sources must not be the step's own targets.
template <class Op>
[[gnu::always_inline]] inline void strided(const Op& op, typename Op::T* target, std::size_t stride, const typename Op::T* source,
                                           std::size_t count) {
    const typename Op::T* const end = source + count;
    for (; source + 4 <= end; source += 4, target += 4 * stride) {
        const Half v0 = op.load(source), v1 = op.load(source + 1), v2 = op.load(source + 2), v3 = op.load(source + 3);
        op.apply(target, v0);
        op.apply(target + stride, v1);
        op.apply(target + 2 * stride, v2);
        op.apply(target + 3 * stride, v3);
    }
    for (; source < end; ++source, target += stride) op.apply(target, op.load(source));
}

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

// Primes in [lo, hi).
template <std::uint32_t kLo, std::uint32_t kHi>
struct PrimesIn {
    std::uint32_t p[64];
    std::size_t count;

    constexpr PrimesIn() : p(), count() {
        for (std::uint32_t x = std::max(kLo, 2u); x < kHi; ++x) {
            bool prime = true;
            for (std::uint32_t d = 2; d * d <= x; ++d) prime &= x % d != 0;
            if (prime) p[count++] = x;
        }
    }
    constexpr const std::uint32_t* begin() const { return p; }
    constexpr const std::uint32_t* end() const { return p + count; }
};

// The sweep's stages; composing them yields every multiplier once. Stage 0 (zeta only): m > 1
// made of 11 and 13. Stage 1: m > 1 with all prime factors in [17, 50); stage 2: in [50, 300);
// stage 3: above 300. Rough m with prime factors in several ranges come from composing stages:
// 1.67 N contributions instead of 2.14 N (1.77 N with two stages, cut at 100).
constexpr std::uint32_t kSmoothPrimes[] = {11, 13};
constexpr std::uint32_t kCut1 = 50, kCut2 = 300;
constexpr PrimesIn<17, kCut1> kStage1Primes;
constexpr PrimesIn<kCut1, kCut2> kStage2Primes;
constexpr std::uint32_t kStageMin[] = {11, 17, kStage2Primes.p[0], PrimesIn<kCut2, kCut2 + 100>().p[0]};  // smallest m

// In a segment of targets, multipliers m <= kSplit go by m over a run of sources i; larger ones
// by source i <= n / (kSplit + 1), over a run of m.
constexpr std::uint32_t kSplit = 4096;
constexpr std::uint32_t kMaxN = 1000000;
constexpr std::uint32_t kMaxSources = kMaxN / (kSplit + 1) + 1;
constexpr std::uint32_t kEnd = 1 << 20;  // above kMaxN; kEnd i < 2^32 for i < 2^12

// The stage of each rough m <= kMaxN, by wheel index: 1, 2 or 3, or 0 for m with prime factors
// in several stages. Used only at compile time.
struct Stages {
    static constexpr std::uint32_t kTotal = kRough.index(kMaxN + 1);
    std::uint8_t of[kTotal];
    std::uint32_t small[4], large[4];  // counts per stage, m <= kSplit and m > kSplit

    constexpr Stages() : of(), small(), large() {
        for (std::uint32_t t = 1; t < kTotal; ++t) of[t] = 3;
        for (const std::uint32_t p : PrimesIn<17, kCut2>())
            for (std::uint32_t t = 0; kRough.value(t) <= kMaxN / p; ++t) of[kRough.index(p * kRough.value(t))] = 0;
        mark_smooth(kStage1Primes, 1, 1, 0);
        mark_smooth(kStage2Primes, 2, 1, 0);
        for (std::uint32_t t = 1; t < kTotal; ++t) ++(kRough.value(t) <= kSplit ? small : large)[of[t]];
    }

    // of[m w] = stage for each m w <= kMaxN with w > 1 made of primes[k..].
    template <class Primes>
    constexpr void mark_smooth(const Primes& primes, std::uint8_t stage, std::uint32_t m, std::size_t k) {
        for (; k < primes.count && primes.p[k] <= kMaxN / m; ++k)
            for (std::uint32_t w = m * primes.p[k];; w *= primes.p[k]) {
                of[kRough.index(w)] = stage;
                mark_smooth(primes, stage, w, k + 1);
                if (w > kMaxN / primes.p[k]) break;
            }
    }
};

constexpr Stages kStages;

// Stage 0's multipliers, ascending: 20 up to kMaxN.
struct Smooth {
    std::uint32_t value[64];
    std::uint32_t count, small;  // all, and m <= kSplit

    constexpr Smooth() : value(), count(), small() {
        add(1, 0);
        std::sort(value, value + count);
        while (value[small] <= kSplit) ++small;
    }

    // Each m w <= kMaxN with w > 1 made of kSmoothPrimes[k..].
    constexpr void add(std::uint32_t m, std::size_t k) {
        for (; k < std::size(kSmoothPrimes) && kSmoothPrimes[k] <= kMaxN / m; ++k)
            for (std::uint32_t w = m * kSmoothPrimes[k];; w *= kSmoothPrimes[k]) {
                value[count++] = w;
                add(w, k + 1);
                if (w > kMaxN / kSmoothPrimes[k]) break;
            }
    }
};

constexpr Smooth kSmooth;

// One stage's multipliers up to kMaxN. Large ones are framed by sentinels for the sweeps:
// large[0] = 0, large[1, kLarge] ascending, large[kLarge + 1] = kEnd.
template <std::uint32_t kStage>
struct Multipliers {
    static constexpr std::uint32_t kSmall = kStage == 0 ? kSmooth.small : kStages.small[kStage];
    static constexpr std::uint32_t kLarge = kStage == 0 ? kSmooth.count - kSmooth.small : kStages.large[kStage];
    std::uint32_t small[kSmall];
    std::uint64_t reciprocal[kSmall];
    std::uint32_t large[kLarge + 2];

    constexpr Multipliers() : small(), reciprocal(), large() {
        std::uint32_t s = 0, l = 1;
        auto put = [&](std::uint32_t m) {
            if (m <= kSplit) {
                small[s] = m;
                reciprocal[s++] = ::reciprocal(m);
            } else {
                large[l++] = m;
            }
        };
        if constexpr (kStage == 0) {
            for (std::uint32_t k = 0; k < kSmooth.count; ++k) put(kSmooth.value[k]);
        } else {
            for (std::uint32_t t = 1; t < Stages::kTotal; ++t)
                if (kStages.of[t] == kStage) put(kRough.value(t));
        }
        large[l] = kEnd;
    }
};

template <std::uint32_t kStage>
constexpr Multipliers<kStage> kMultipliers;
constexpr const Multipliers<0>& kStage0 = kMultipliers<0>;
constexpr const Multipliers<1>& kStage1 = kMultipliers<1>;
constexpr const Multipliers<2>& kStage2 = kMultipliers<2>;
constexpr const Multipliers<3>& kStage3 = kMultipliers<3>;
static_assert(kStage0.small[0] == kStageMin[0] && kStage1.small[0] == kStageMin[1] && kStage2.small[0] == kStageMin[2] &&
              kStage3.small[0] == kStageMin[3]);

// Tiny multipliers m <= kTinyBound go over a segment in L1-sized pieces (32 KiB), the tiny ones of
// all stages together, so one fetch of a target line from L2 serves all of them.
constexpr std::uint32_t kTinyBound = 256, kZetaPiece = 4096, kMoebiusPiece = 8192;

// A stage in a sweep: its multipliers and the bounds carried from segment to segment.
template <std::uint32_t kStage>
struct Sweep {
    static constexpr std::uint32_t kTiny = std::uint32_t(
        std::upper_bound(kMultipliers<kStage>.small, kMultipliers<kStage>.small + Multipliers<kStage>::kSmall, kTinyBound) -
        kMultipliers<kStage>.small);
    const Multipliers<kStage>& m;
    std::uint32_t run[Multipliers<kStage>::kSmall];  // per small m: the next source bound (zeta: not tiny m)
    std::uint32_t edge[kMaxSources];                 // per source i: the next index into m.large

    // Index of the first large m with i m > bound.
    std::uint32_t large_end(std::uint32_t i, std::uint32_t bound) const {
        const std::uint32_t* const end = m.large + Multipliers<kStage>::kLarge + 1;
        return std::uint32_t(std::upper_bound(m.large + 1, end, bound / i) - m.large);
    }
};

static_assert(Sweep<3>::kTiny == 0);  // the sweeps' tiny loops take stages 0..2

// pairs[k] = (a_k, b_k R mod P) + pairs[k / 3] (the term k / 3 for k = 0 mod 3) for first <= k <
// end; b[k - base] = b_k. pairs[k / 3] must be final. a lies in the second half of pairs, at word
// offset a - pairs: pair k overwrites only a values below k, which are read already.
void interleave_zeta3(const AliasWord* a, const std::uint32_t* b, std::uint32_t base, std::uint64_t* pairs, std::uint32_t first,
                      std::uint32_t end) {
    const Vec r2 = broadcast(kR2);
    // b_k R as Montgomery products of b_k and R^2: dwords [b R] in the high halves.
    auto scaled_pairs = [&](std::uint32_t k, Vec& low, Vec& high) {
        const Vec va = load(a + k), vb = load(b + (k - base));
        const Vec even = redc(_mm256_mul_epu32(vb, r2));
        const Vec odd = redc(_mm256_mul_epu32(_mm256_srli_epi64(vb, 32), r2));
        // even/odd: [_ bR] per qword (below 1.25 P). Reduce, then place above a.
        const Vec bs = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
        const Vec br = _mm256_min_epu32(bs, _mm256_sub_epi32(bs, broadcast(kP)));
        const Vec lo = _mm256_unpacklo_epi32(va, br), hi = _mm256_unpackhi_epi32(va, br);
        low = _mm256_permute2x128_si256(lo, hi, 0x20);
        high = _mm256_permute2x128_si256(lo, hi, 0x31);
    };
    auto one = [&](std::uint32_t k) {
        const std::uint32_t br = std::uint32_t((std::uint64_t(b[k - base]) << 32) % kP);
        const std::uint64_t pair = a[k] | std::uint64_t(br) << 32;
        std::memcpy(pairs + k, &pair, sizeof pair);
        if (k % 3 == 0) add_pair(pairs + k, pairs + k / 3);
    };
    std::uint32_t k = first;
    for (; (k < 24 || k % 24) && k < end; ++k) one(k);
    // 24 targets from 8 sources below them: multiples of 3 at offsets 0, 3 | 6 | 9 of each 12.
    for (; k + 24 <= end; k += 24) {
        Vec v[6];
        scaled_pairs(k, v[0], v[1]);
        scaled_pairs(k + 8, v[2], v[3]);
        scaled_pairs(k + 16, v[4], v[5]);
        for (int h = 0; h < 2; ++h) {
            const Vec s = load(pairs + k / 3 + 4 * h);
            store(pairs + k + 12 * h, add(v[3 * h], spread<0x40, 0xC3>(s)));
            store(pairs + k + 12 * h + 4, add(v[3 * h + 1], spread<0x20, 0x30>(s)));
            store(pairs + k + 12 * h + 8, add(v[3 * h + 2], spread<0x0C, 0x0C>(s)));
        }
    }
    for (; k < end; ++k) one(k);
}

// The squarefree d > 1 made of the primes given, but skip; odd marks those with an odd number of
// prime factors (Moebius function -1). For a set S of primes, zeta_S x = y is y_k = x_k - sum of
// mu(d) y_k/d (final sources), and Moebius_S y = y_k + sum of mu(d) y_k/d (old sources).
template <std::size_t kPrimes>
struct Squarefree {
    static constexpr std::size_t kMax = (std::size_t(1) << kPrimes) - 1;
    std::uint32_t d[kMax];
    std::uint64_t reciprocal[kMax];
    bool odd[kMax];
    std::size_t count;

    constexpr Squarefree(const std::uint32_t (&primes)[kPrimes], std::uint32_t skip) : d(), reciprocal(), odd(), count() {
        for (std::size_t mask = 1; mask <= kMax; ++mask) {
            std::uint32_t product = 1;
            bool parity = false;
            for (std::size_t j = 0; j < kPrimes; ++j)
                if (mask >> j & 1) product *= primes[j], parity = !parity;
            if (product == skip) continue;
            d[count] = product;
            reciprocal[count] = ::reciprocal(product);
            odd[count++] = parity;
        }
    }
};

// x[k] op= x[k / d] for the terms d and the targets k in [first, end): Odd for odd d, else Even.
// The sources must lie below first.
template <class Odd, class Even, std::size_t kPrimes>
void apply_terms(typename Odd::T* x, const Squarefree<kPrimes>& terms, std::uint32_t first, std::uint32_t end) {
    const Odd odd;
    const Even even;
    for (std::size_t t = 0; t < terms.count; ++t) {
        const std::uint32_t d = terms.d[t], lo = divide(first - 1, terms.reciprocal[t]) + 1, hi = divide(end - 1, terms.reciprocal[t]);
        if (lo > hi) continue;
        if (terms.odd[t]) strided(odd, x + std::size_t(lo) * d, d, x + lo, hi + 1 - lo);
        else strided(even, x + std::size_t(lo) * d, d, x + lo, hi + 1 - lo);
    }
}

// Zeta of 3, 5 and 7 with final sources: 3 by vectors in interleave_zeta3, the other terms after
// it, per range of targets. 0.83 N updates against 0.68 N in passes, but the targets are in L1.
constexpr std::uint32_t kInterleavePrimes[] = {3, 5, 7};
constexpr Squarefree kInterleaveTerms(kInterleavePrimes, 3);

// pairs[k] for first <= k < end as interleave_zeta3, with the zeta passes of 3, 5 and 7, in ranges
// of targets [s, e) with e <= 3 s: their sources lie below them, final. Called for consecutive
// ranges from first = 1.
void interleave_zeta(const AliasWord* a, const std::uint32_t* b, std::uint32_t base, std::uint64_t* pairs, std::uint32_t first,
                     std::uint32_t end) {
    constexpr std::uint32_t kRange = 2016;  // pairs: 15.75 KiB, a multiple of 24
    for (std::uint32_t s = first, e; s < end; s = e) {
        e = std::min(end, s < kRange ? std::min(3 * s, kRange) : (s / kRange + 1) * kRange);
        interleave_zeta3(a, b, base, pairs, s, e);
        apply_terms<PairAdd, PairSub>(pairs, kInterleaveTerms, s, e);
    }
}

// Zeta pass of p >= 5: pairs[i p] += pairs[i], i ascending.
void zeta_pass(std::uint64_t* pairs, std::uint32_t n, std::uint32_t p) { strided(PairAdd{}, pairs + p, p, pairs + 1, n / p); }

// visit(m) for the multipliers m of a stage with i m <= last, ascending.
template <std::uint32_t kStage, class Visit>
[[gnu::always_inline]] inline void for_multipliers(const Multipliers<kStage>& s, std::uint32_t i, std::uint32_t last, Visit visit) {
    std::uint32_t k = 0;
    for (; k < Multipliers<kStage>::kSmall && s.small[k] * i <= last; ++k) visit(s.small[k]);
    if (k < Multipliers<kStage>::kSmall) return;
    for (const std::uint32_t* m = s.large + 1; *m * i <= last; ++m) visit(*m);
}

// The tiny m of one stage on the targets in [bottom, top]: pairs[i m] += source[i].
template <std::uint32_t kStage>
void zeta_tiny(std::uint64_t* pairs, const std::uint64_t* source, const Sweep<kStage>& s, std::uint32_t bottom, std::uint32_t top) {
    const PairAdd op;
    for (std::uint32_t k = 0; k < Sweep<kStage>::kTiny; ++k) {
        const std::uint32_t m = s.m.small[k], first = divide(bottom - 1, s.m.reciprocal[k]) + 1;
        strided(op, pairs + std::size_t(first) * m, m, source + first, divide(top, s.m.reciprocal[k]) + 1 - first);
    }
}

// The other m of one stage on the targets in [start, last], segments descending.
template <std::uint32_t kStage>
void zeta_segment(std::uint64_t* pairs, const std::uint64_t* source, Sweep<kStage>& s, std::uint32_t start, std::uint32_t last) {
    const PairAdd op;
    for (std::uint32_t k = Sweep<kStage>::kTiny; k < Multipliers<kStage>::kSmall; ++k) {
        const std::uint32_t m = s.m.small[k], first = divide(start - 1, s.m.reciprocal[k]) + 1;
        strided(op, pairs + std::size_t(first) * m, m, source + first, s.run[k] + 1 - first);
        s.run[k] = first - 1;
    }
    for (std::size_t i = 1; i * (kSplit + 1) <= last; ++i) {
        const Half v = op.load(source + i);
        const std::uint32_t* m = s.m.large + s.edge[i] - 1;
        for (; *m * i >= start; --m) op.apply(pairs + *m * i, v);
        s.edge[i] = std::uint32_t(m - s.m.large) + 1;
    }
}

// x[i m] += source[i] over the stage's m, i m <= last, i descending (in place: x[i] still old).
template <std::uint32_t kStage>
void zeta_by_source(std::uint64_t* x, const std::uint64_t* source, const Multipliers<kStage>& s, std::uint32_t last) {
    const PairAdd op;
    for (std::uint32_t i = last / s.small[0]; i >= 1; --i) {
        const Half v = op.load(source + i);
        for_multipliers(s, i, last, [&](std::uint32_t m) { op.apply(x + std::size_t(m) * i, v); });
    }
}

// Pairs for the prefix copies of the sweeps: stage s > 0 reads its sources, up to n / kStageMin[s],
// from a copy.
std::size_t prefix_pairs(std::uint32_t n) { return n / kStageMin[1] + n / kStageMin[2] + n / kStageMin[3] + 24; }

// The zeta passes of all primes above 7, stage by stage. Stage s > 0 reads its sources from
// prefix s, a copy with stages 0..s-1 applied. Target segments descend, so stage 0 sources (below
// the segment) are unchanged when read. Segment 0: stage 0 in place, then stages 1..3.
void zeta_sweep(std::uint64_t* pairs, std::uint32_t n, std::uint64_t* prefix) {
    const std::uint32_t last1 = n / kStageMin[1], last2 = n / kStageMin[2], last3 = n / kStageMin[3];
    std::uint64_t* const prefix1 = prefix;
    std::uint64_t* const prefix2 = prefix1 + (last1 + 8) / 8 * 8;
    std::uint64_t* const prefix3 = prefix2 + (last2 + 8) / 8 * 8;
    std::memcpy(prefix1, pairs, (last1 + 1) * sizeof(std::uint64_t));
    for (const std::uint32_t p : kSmoothPrimes) zeta_pass(prefix1, last1, p);
    std::memcpy(prefix2, prefix1, (last2 + 1) * sizeof(std::uint64_t));
    zeta_by_source(prefix2, prefix2, kStage1, last2);
    std::memcpy(prefix3, prefix2, (last3 + 1) * sizeof(std::uint64_t));
    zeta_by_source(prefix3, prefix3, kStage2, last3);
    constexpr std::uint32_t kSegment = 1 << 15;  // 256 KiB of pairs
    if (n >= kSegment) {
        static Sweep<0> zero{kStage0};
        static Sweep<1> one{kStage1};
        static Sweep<2> two{kStage2};
        static Sweep<3> three{kStage3};
        auto init = [&](auto& s) {
            for (std::uint32_t k = 0; k < std::size(s.run); ++k) s.run[k] = n / s.m.small[k];
            for (std::uint32_t i = 1; i * (kSplit + 1) <= n; ++i) s.edge[i] = s.large_end(i, n);
        };
        init(zero);
        init(one);
        init(two);
        init(three);
        for (std::uint32_t start = n / kSegment * kSegment; start >= kSegment; start -= kSegment) {
            const std::uint32_t last = std::min(n, start + kSegment - 1);
            for (std::uint32_t bottom = start; bottom <= last; bottom += kZetaPiece) {  // ascending: faster
                const std::uint32_t top = std::min(last, bottom + kZetaPiece - 1);
                zeta_tiny(pairs, pairs, zero, bottom, top);
                zeta_tiny(pairs, prefix1, one, bottom, top);
                zeta_tiny(pairs, prefix2, two, bottom, top);
            }
            zeta_segment(pairs, pairs, zero, start, last);
            zeta_segment(pairs, prefix1, one, start, last);
            zeta_segment(pairs, prefix2, two, start, last);
            zeta_segment(pairs, prefix3, three, start, last);
        }
    }
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    zeta_by_source(pairs, pairs, kStage0, last0);
    zeta_by_source(pairs, prefix1, kStage1, last0);
    zeta_by_source(pairs, prefix2, kStage2, last0);
    zeta_by_source(pairs, prefix3, kStage3, last0);
}

// Zeta pass of p = 2 fused with the product and the Moebius pass of 2, ascending:
// A_k += A_k/2, then c_k = A_k B_k - A_k/2 B_k/2 (the second term for even k). c is the pairs'
// memory as words: c_k overwrites pair k / 2, which is read for the last time at target k.
void zeta2_product_moebius2(std::uint64_t* pairs, AliasWord* c, std::uint32_t n) {
    auto one = [&](std::uint32_t k) {
        std::uint32_t ai = std::uint32_t(pairs[k]), bi = std::uint32_t(pairs[k] >> 32);
        if (k % 2) {
            c[k] = product(ai, bi);
            return;
        }
        const std::uint64_t s = pairs[k / 2];
        ai = add(ai, std::uint32_t(s));
        bi = add(bi, std::uint32_t(s >> 32));
        pairs[k] = ai | std::uint64_t(bi) << 32;
        c[k] = sub(product(ai, bi), product(std::uint32_t(s), std::uint32_t(s >> 32)));
    };
    std::uint32_t k = 1;
    for (; k < 16 && k <= n; ++k) one(k);  // below 16 the sources k / 2 + 8 would be targets
    // 16 targets from 8 sources k / 2 below them; the products of the sources in one vector.
    const Vec lanes_lo = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3), lanes_hi = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
    for (; k + 15 <= n; k += 16) {
        const Vec s0 = load(pairs + k / 2), s1 = load(pairs + k / 2 + 4);
        const Vec t0 = add(load(pairs + k), spread<0x10, 0x33>(s0));
        const Vec t1 = add(load(pairs + k + 4), spread<0x32, 0x33>(s0));
        const Vec t2 = add(load(pairs + k + 8), spread<0x10, 0x33>(s1));
        const Vec t3 = add(load(pairs + k + 12), spread<0x32, 0x33>(s1));
        store(pairs + k, t0);
        store(pairs + k + 4, t1);
        store(pairs + k + 8, t2);
        store(pairs + k + 12, t3);
        const Vec h = high_dwords(product4(s0), product4(s1));
        const Vec h0 = _mm256_blend_epi32(_mm256_setzero_si256(), _mm256_permutevar8x32_epi32(h, lanes_lo), 0x55);
        const Vec h1 = _mm256_blend_epi32(_mm256_setzero_si256(), _mm256_permutevar8x32_epi32(h, lanes_hi), 0x55);
        store(c + k, sub(high_dwords(product4(t0), product4(t1)), h0));
        store(c + k + 8, sub(high_dwords(product4(t2), product4(t3)), h1));
    }
    for (; k <= n; ++k) one(k);
}

// Moebius pass of p on c: c[i p] -= c[i], i descending.
void moebius_pass(std::uint32_t* c, std::uint32_t n, std::uint32_t p) {
    const WordSub op;
    std::size_t count = n / p;
    const std::uint32_t* source = c + count;
    std::uint32_t* target = c + count * p;
    for (; count >= 2; count -= 2, source -= 2, target -= 2 * p) {
        const Half v0 = op.load(source), v1 = op.load(source - 1);
        op.apply(target, v0);
        op.apply(target - p, v1);
    }
    if (count) op.apply(target, op.load(source));
}

// Moebius of 7, 11 and 13 in one pass with old sources: 0.34 N updates against 0.31 N in three
// passes, but one walk over c instead of three.
constexpr std::uint32_t kJointPrimes[] = {7, 11, 13};
constexpr Squarefree kJointTerms(kJointPrimes, 0);

// c_k += sum of mu(d) c_k/d over kJointTerms, in ranges of targets [s, e) descending with
// e - 1 < 7 s: their sources lie below them, still old.
void moebius_joint(std::uint32_t* c, std::uint32_t n) {
    constexpr std::uint32_t kRange = 4096;  // dwords: 16 KiB
    for (std::uint32_t e = n + 1, s; e > 1; e = s) {
        s = std::max(e > kRange ? e - kRange : 1u, (e - 1) / kJointPrimes[0] + 1);
        apply_terms<WordSub, WordAdd>(c, kJointTerms, s, e);
    }
}

// Small multipliers k in [k0, k1) on the targets from their carried bound up to last.
template <std::uint32_t kStage>
void moebius_small(std::uint32_t* c, const std::uint32_t* source, Sweep<kStage>& s, std::uint32_t k0, std::uint32_t k1, std::uint32_t last) {
    const WordSub op;
    for (std::uint32_t k = k0; k < k1; ++k) {
        const std::uint32_t m = s.m.small[k], first = s.run[k], last_source = divide(last, s.m.reciprocal[k]);
        strided(op, c + std::size_t(first) * m, m, source + first, last_source + 1 - first);
        s.run[k] = last_source + 1;
    }
}

// The m above the tiny ones of one stage on the targets in [start, last], segments ascending:
// c[i m] -= source[i].
template <std::uint32_t kStage>
void moebius_segment(std::uint32_t* c, const std::uint32_t* source, Sweep<kStage>& s, std::uint32_t start, std::uint32_t last) {
    moebius_small(c, source, s, Sweep<kStage>::kTiny, Multipliers<kStage>::kSmall, last);
    const WordSub op;
    for (std::size_t i = 1; i * (kSplit + 1) <= last; ++i) {
        const Half v = op.load(source + i);
        const std::uint32_t* m = s.m.large + s.edge[i];
        for (; *m * i <= last; ++m) op.apply(c + *m * i, v);
        s.edge[i] = std::uint32_t(m - s.m.large);
    }
}

// x[i m] -= source[i] over the stage's m, i m <= last, i ascending (in place: x[i] final).
template <std::uint32_t kStage>
void moebius_by_source(std::uint32_t* x, const std::uint32_t* source, const Multipliers<kStage>& s, std::uint32_t last) {
    const WordSub op;
    for (std::uint32_t i = 1; s.small[0] * i <= last; ++i) {
        const Half v = op.load(source + i);
        for_multipliers(s, i, last, [&](std::uint32_t m) { op.apply(x + std::size_t(m) * i, v); });
    }
}

// The Moebius passes of all primes above 13 (inverting stages 1..3 of zeta_sweep): stage 3, then 2,
// then 1, each as c[i m] -= final c[i]. Stage 3's final values up to n / kStageMin[3] go to
// prefix 3 first, those of stages 3 and 2 up to n / kStageMin[2] to prefix 2. Segment 0: stages
// 3 and 2, then stage 1 in place; then target segments ascend, so stage 1 sources (below the
// segment) are final when read.
void moebius_sweep(std::uint32_t* c, std::uint32_t n, std::uint32_t* prefix) {
    const std::uint32_t last2 = n / kStageMin[2], last3 = n / kStageMin[3];
    std::uint32_t* const prefix2 = prefix;
    std::uint32_t* const prefix3 = prefix2 + (last2 + 16) / 16 * 16;
    std::memcpy(prefix3, c, (last3 + 1) * sizeof(std::uint32_t));
    moebius_by_source(prefix3, prefix3, kStage3, last3);
    std::memcpy(prefix2, c, (last2 + 1) * sizeof(std::uint32_t));
    moebius_by_source(prefix2, prefix3, kStage3, last2);
    moebius_by_source(prefix2, prefix2, kStage2, last2);
    constexpr std::uint32_t kSegment = 1 << 15;  // 128 KiB: with its sources, within L2
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    moebius_by_source(c, prefix3, kStage3, last0);
    moebius_by_source(c, prefix2, kStage2, last0);
    moebius_by_source(c, c, kStage1, last0);
    if (n < kSegment) return;
    static Sweep<1> one{kStage1};
    static Sweep<2> two{kStage2};
    static Sweep<3> three{kStage3};
    auto init = [&](auto& s) {
        for (std::uint32_t k = 0; k < std::size(s.run); ++k) s.run[k] = last0 / s.m.small[k] + 1;
        for (std::uint32_t i = 1; i * (kSplit + 1) <= n; ++i) s.edge[i] = s.large_end(i, last0);
    };
    init(one);
    init(two);
    init(three);
    for (std::uint32_t start = kSegment; start <= n; start += kSegment) {
        const std::uint32_t last = std::min(n, start + kSegment - 1);
        for (std::uint32_t bottom = start; bottom <= last; bottom += kMoebiusPiece) {
            const std::uint32_t top = std::min(last, bottom + kMoebiusPiece - 1);
            moebius_small(c, c, one, 0, Sweep<1>::kTiny, top);
            moebius_small(c, prefix2, two, 0, Sweep<2>::kTiny, top);
        }
        moebius_segment(c, c, one, start, last);
        moebius_segment(c, prefix2, two, start, last);
        moebius_segment(c, prefix3, three, start, last);
    }
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    io::advise_sequential(in);
    // One region: the pairs (2 words each), with a parsed into their second half, later c; then
    // scratch: chunks of b, later the sweeps' prefix copies, later the output text; then the
    // parser's workspace. 5 huge pages.
    constexpr std::size_t kPad = 64;
    const std::size_t words = n + kPad;
    const std::size_t scratch_offset = (2 * words * sizeof(std::uint32_t) + 63) / 64 * 64;
    const std::size_t scratch_bytes = std::max(
        {chunks::Parser::kMaxChunkTokens * sizeof(std::uint32_t), prefix_pairs(n) * sizeof(std::uint64_t), fields::kTextBytes});
    const std::size_t workspace_offset = (scratch_offset + scratch_bytes + 63) / 64 * 64;
    char* const base = mem::huge<char>(workspace_offset + chunks::Parser::kWorkspaceBytes);
    auto* const region = reinterpret_cast<std::uint32_t*>(base);
    auto* const pairs = reinterpret_cast<std::uint64_t*>(region);
    std::uint32_t* const a = region + words;
    char* const scratch = base + scratch_offset;
    chunks::Parser parser(base + workspace_offset);
    parser.read(in, a + 1, n, [](std::uint32_t* values, std::size_t count) { return values + count; });
    // b chunk by chunk into the scratch, each interleaved while in L2.
    std::uint32_t next = 1;
    parser.read(in, reinterpret_cast<std::uint32_t*>(scratch), n, [&](std::uint32_t* b, std::size_t count) {
        interleave_zeta(a, b, next, pairs, next, next + std::uint32_t(count));
        next += std::uint32_t(count);
        return b;
    });

    zeta_sweep(pairs, n, reinterpret_cast<std::uint64_t*>(scratch));
    std::uint32_t* const c = region;
    zeta2_product_moebius2(pairs, c, n);
    moebius_pass(c, n, 3);
    moebius_pass(c, n, 5);
    moebius_joint(c, n);
    moebius_sweep(c, n, reinterpret_cast<std::uint32_t*>(scratch));

    io::Writer out;
    fields::write(out, c + 1, n, scratch);
}

}  // namespace

RUN_EARLY(solve)
