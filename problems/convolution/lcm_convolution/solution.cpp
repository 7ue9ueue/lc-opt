// c_k = sum over lcm(i, j) = k of a_i b_j mod 998244353, 1 <= k <= N <= 10^6.
// Sums over divisors (zeta) of a and b, pointwise product, then the inverse (Moebius). Both are
// products of commuting per-prime passes (zeta: x_ip += x_i, i ascending; Moebius: x_ip -= x_i,
// i descending). Zeta: p = 3 in the interleave, p = 2 in the product sweep, all others in one
// sweep, target segment by target segment, in L2, in three stages of multipliers m: made of
// 5..13, of 17..97, and of primes above 100. Moebius: one pass per prime up to 13, then the
// sweep with the last two stages.
// a and b are interleaved as pairs, so one load fetches both. Design and measurements: notes.md.
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
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

// One update of the sweeps: pairs (zeta, x += y) or dwords (Moebius, x -= y).
struct PairAdd {
    using T = std::uint64_t;
    Half neg_p = held(0 - kP);
    static Half load(const T* p) { return load_pair(p); }
    void apply(T* t, Half v) const {
        const Half s = _mm_add_epi32(load_pair(t), v);
        store_pair(t, _mm_min_epu32(s, _mm_add_epi32(s, neg_p)));
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

// The sweep's stages; composing them yields every multiplier once. Stage 0 (zeta only): m > 1
// with all prime factors in {5, 7, 11, 13}. Stage 1: m > 1 with all prime factors in [17, 100);
// stage 2: m > 1 with all prime factors above 100. Rough m with prime factors on both sides of
// 100 come from composing stages 1 and 2: 1.77 N contributions instead of 2.14 N.
constexpr std::uint32_t kSmoothPrimes[] = {5, 7, 11, 13};
constexpr std::uint32_t kStagePrimes[] = {17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
constexpr std::uint32_t kStage1Min = 17, kStage2Min = 101;  // smallest multipliers

// In a segment of targets, multipliers m <= kSplit go by m over a run of sources i; larger ones
// by source i <= n / (kSplit + 1), over a run of m.
constexpr std::uint32_t kSplit = 4096;
constexpr std::uint32_t kMaxN = 1000000;
constexpr std::uint32_t kMaxSources = kMaxN / (kSplit + 1) + 1;
constexpr std::uint32_t kEnd = 1 << 20;  // above kMaxN; kEnd i < 2^32 for i < 2^12

// The stage of each rough m <= kMaxN, by wheel index: 1 (only primes below 100), 2 (none below
// 100), 0 otherwise. Used only at compile time.
struct Stages {
    static constexpr std::uint32_t kTotal = kRough.index(kMaxN + 1);
    std::uint8_t of[kTotal];
    std::uint32_t small[3], large[3];  // counts per stage, m <= kSplit and m > kSplit

    constexpr Stages() : of(), small(), large() {
        for (std::uint32_t t = 1; t < kTotal; ++t) of[t] = 2;
        for (const std::uint32_t p : kStagePrimes)
            for (std::uint32_t t = 0; kRough.value(t) <= kMaxN / p; ++t) of[kRough.index(p * kRough.value(t))] = 0;
        mark_smooth(1, 0);
        for (std::uint32_t t = 1; t < kTotal; ++t) ++(kRough.value(t) <= kSplit ? small : large)[of[t]];
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

// Stage 0's multipliers, ascending: 222 up to kMaxN.
struct Smooth {
    std::uint32_t value[256];
    std::uint32_t count, small;  // all, and m <= kSplit

    constexpr Smooth() : value(), count(), small() {
        for (std::uint32_t a = 1; a <= kMaxN; a *= kSmoothPrimes[0])
            for (std::uint32_t b = a; b <= kMaxN; b *= kSmoothPrimes[1])
                for (std::uint32_t c = b; c <= kMaxN; c *= kSmoothPrimes[2])
                    for (std::uint32_t d = c; d <= kMaxN; d *= kSmoothPrimes[3])
                        if (d > 1) value[count++] = d;
        std::sort(value, value + count);
        while (value[small] <= kSplit) ++small;
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
static_assert(kStage1.small[0] == kStage1Min && kStage2.small[0] == kStage2Min);

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

// pairs[k] = (a_k, b_k R mod P) for first <= k < end, with the zeta pass of p = 3 (pairs[k] +=
// pairs[k / 3] for k = 0 mod 3, ascending); b[k - first] = b_k. Called for consecutive ranges
// from first = 1, each ending at a multiple of 24 but the last. a lies in the second half of
// pairs, at word offset a - pairs: pair k overwrites only a values below k, which are read already.
void interleave_zeta3(const AliasWord* a, const std::uint32_t* b, std::uint64_t* pairs, std::uint32_t first, std::uint32_t end) {
    const Vec r2 = broadcast(kR2);
    // b_k R as Montgomery products of b_k and R^2: dwords [b R] in the high halves.
    auto scaled_pairs = [&](std::uint32_t k, Vec& low, Vec& high) {
        const Vec va = load(a + k), vb = load(b + (k - first));
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
        const std::uint32_t br = std::uint32_t((std::uint64_t(b[k - first]) << 32) % kP);
        const std::uint64_t pair = a[k] | std::uint64_t(br) << 32;
        std::memcpy(pairs + k, &pair, sizeof pair);
        if (k % 3 == 0) add_pair(pairs + k, pairs + k / 3);
    };
    std::uint32_t k = first;
    for (; k < 24 && k < end; ++k) one(k);
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

// The zeta passes of all primes but 2 and 3: stage 0, then 1, then 2. Stage 1 reads its sources
// (up to n / 17) from prefix1, a copy with stage 0 applied; stage 2 (up to n / 101) from prefix2,
// with stages 0 and 1. Target segments descend, so stage 0 sources (below the segment) are
// unchanged when read. Segment 0: stage 0 in place, then stages 1 and 2. prefix: room for
// n / 17 + n / 101 + 16 pairs.
void zeta_sweep(std::uint64_t* pairs, std::uint32_t n, std::uint64_t* prefix) {
    const std::uint32_t last1 = n / kStage1Min, last2 = n / kStage2Min;
    std::uint64_t* const prefix1 = prefix;
    std::uint64_t* const prefix2 = prefix + (last1 + 8) / 8 * 8;
    std::memcpy(prefix1, pairs, (last1 + 1) * sizeof(std::uint64_t));
    for (const std::uint32_t p : kSmoothPrimes) zeta_pass(prefix1, last1, p);
    std::memcpy(prefix2, prefix1, (last2 + 1) * sizeof(std::uint64_t));
    zeta_by_source(prefix2, prefix2, kStage1, last2);
    constexpr std::uint32_t kSegment = 1 << 15;  // 256 KiB of pairs
    if (n >= kSegment) {
        static Sweep<0> zero{kStage0};
        static Sweep<1> one{kStage1};
        static Sweep<2> two{kStage2};
        auto init = [&](auto& s) {
            for (std::uint32_t k = 0; k < std::size(s.run); ++k) s.run[k] = n / s.m.small[k];
            for (std::uint32_t i = 1; i * (kSplit + 1) <= n; ++i) s.edge[i] = s.large_end(i, n);
        };
        init(zero);
        init(one);
        init(two);
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
        }
    }
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    zeta_by_source(pairs, pairs, kStage0, last0);
    zeta_by_source(pairs, prefix1, kStage1, last0);
    zeta_by_source(pairs, prefix2, kStage2, last0);
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

// The Moebius passes of all primes >= 17 (inverting stages 1 and 2 of zeta_sweep): stage 2, then
// stage 1, each as c[i m] -= final c[i]. Stage 2's final values up to n / kStage2Min go to
// `prefix` first. Segment 0: stage 2, then stage 1 in place; then target segments ascend, so
// stage 1 sources (below the segment) are final when read.
void moebius_sweep(std::uint32_t* c, std::uint32_t n, std::uint32_t* prefix) {
    const std::uint32_t prefix_last = n / kStage2Min;
    std::memcpy(prefix, c, (prefix_last + 1) * sizeof(std::uint32_t));
    moebius_by_source(prefix, prefix, kStage2, prefix_last);
    constexpr std::uint32_t kSegment = 1 << 15;  // 128 KiB: with its sources, within L2
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    moebius_by_source(c, prefix, kStage2, last0);
    moebius_by_source(c, c, kStage1, last0);
    if (n < kSegment) return;
    static Sweep<1> one{kStage1};
    static Sweep<2> two{kStage2};
    auto init = [&](auto& s) {
        for (std::uint32_t k = 0; k < std::size(s.run); ++k) s.run[k] = last0 / s.m.small[k] + 1;
        for (std::uint32_t i = 1; i * (kSplit + 1) <= n; ++i) s.edge[i] = s.large_end(i, last0);
    };
    init(one);
    init(two);
    for (std::uint32_t start = kSegment; start <= n; start += kSegment) {
        const std::uint32_t last = std::min(n, start + kSegment - 1);
        for (std::uint32_t bottom = start; bottom <= last; bottom += kMoebiusPiece) {
            const std::uint32_t top = std::min(last, bottom + kMoebiusPiece - 1);
            moebius_small(c, c, one, 0, Sweep<1>::kTiny, top);
            moebius_small(c, prefix, two, 0, Sweep<2>::kTiny, top);
        }
        moebius_segment(c, c, one, start, last);
        moebius_segment(c, prefix, two, start, last);
    }
}

// Marks a mapped input as read once, so the kernel skips marking each page accessed when the
// Reader unmaps it. The range starts at the page of the next token.
void advise_sequential(const io::Reader& in) {
    struct stat st;
    if (::fstat(0, &st) != 0 || !S_ISREG(st.st_mode) || std::size_t(st.st_size) <= io::detail::kMapAbove) return;
    const auto start = reinterpret_cast<std::uintptr_t>(in.scan().cur) & ~std::uintptr_t(4095);
    ::madvise(reinterpret_cast<void*>(start), std::size_t(st.st_size), MADV_SEQUENTIAL);
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    advise_sequential(in);
    // One region: the pairs (2 words each), with a parsed into their second half, later c; then
    // scratch: chunks of b, later the sweeps' prefix copies, later the output text. b goes by
    // chunks so that its pages need not be faulted in: 5 huge pages instead of 7.
    constexpr std::uint32_t kChunk = 262080;  // b values, a multiple of 24
    constexpr std::size_t kPad = 64;
    const std::size_t words = n + kPad;
    const std::size_t scratch_offset = (2 * words * sizeof(std::uint32_t) + 63) / 64 * 64;
    const std::size_t prefix_bytes = (n / kStage1Min + n / kStage2Min + 16) * sizeof(std::uint64_t);
    const std::size_t scratch_bytes = std::max({kChunk * sizeof(std::uint32_t), prefix_bytes, fields::kTextBytes});
    char* const base = mem::huge<char>(scratch_offset + scratch_bytes);
    auto* const region = reinterpret_cast<std::uint32_t*>(base);
    auto* const pairs = reinterpret_cast<std::uint64_t*>(region);
    std::uint32_t* const a = region + words;
    char* const scratch = base + scratch_offset;
    io::read_bulk(in, a + 1, n);
    auto* const b = reinterpret_cast<std::uint32_t*>(scratch);
    for (std::uint32_t first = 1, end; first <= n; first = end) {
        end = std::min(n + 1, (first / kChunk + 1) * kChunk);
        io::read_bulk(in, b, end - first);
        interleave_zeta3(a, b, pairs, first, end);
    }

    zeta_sweep(pairs, n, reinterpret_cast<std::uint64_t*>(scratch));
    std::uint32_t* const c = region;
    zeta2_product_moebius2(pairs, c, n);
    for (const std::uint32_t p : {3, 5, 7, 11, 13}) moebius_pass(c, n, p);
    moebius_sweep(c, n, reinterpret_cast<std::uint32_t*>(scratch));

    io::Writer out;
    fields::write(out, c + 1, n, scratch);
}

}  // namespace

RUN_EARLY(solve)
