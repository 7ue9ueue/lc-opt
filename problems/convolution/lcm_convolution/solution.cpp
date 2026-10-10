// c_k = sum over lcm(i, j) = k of a_i b_j mod 998244353, 1 <= k <= N <= 10^6.
// Sums over divisors (zeta) of a and b, pointwise product, then the inverse (Moebius). Both are
// products of commuting per-prime passes. Primes 2..13: one pass each (zeta: x_ip += x_i, i
// ascending; Moebius: x_ip -= x_i, i descending). Primes >= 17 together: one sweep over the
// multipliers m coprime to 30030 ("rough"), target segment by target segment, in L2, in two
// stages (prime factors below and above 100).
// a and b are interleaved as pairs, so one load fetches both. Design and measurements: notes.md.
#include <immintrin.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
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

// The rough sweep runs in two stages, which cuts contributions from 2.14 N to 1.77 N.
// Stage 1: m > 1 with all prime factors in [17, 100); stage 2: m > 1 with all prime factors
// above 100. Rough m with prime factors on both sides come from composing the two.
constexpr std::uint32_t kStagePrimes[] = {17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
constexpr std::uint32_t kStage2Min = 101;  // smallest stage-2 multiplier

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

// One stage's multipliers up to kMaxN. Large ones are framed by sentinels for the sweeps:
// large[0] = 0, large[1, kLarge] ascending, large[kLarge + 1] = kEnd.
template <std::uint32_t kStage>
struct Multipliers {
    static constexpr std::uint32_t kSmall = kStages.small[kStage], kLarge = kStages.large[kStage];
    std::uint32_t small[kSmall];
    std::uint64_t reciprocal[kSmall];
    std::uint32_t large[kLarge + 2];

    constexpr Multipliers() : small(), reciprocal(), large() {
        std::uint32_t s = 0, l = 1;
        for (std::uint32_t t = 1; t < Stages::kTotal; ++t) {
            const std::uint32_t m = kRough.value(t);
            if (kStages.of[t] != kStage) continue;
            if (m <= kSplit) {
                small[s] = m;
                reciprocal[s++] = ::reciprocal(m);
            } else {
                large[l++] = m;
            }
        }
        large[l] = kEnd;
    }
};

template <std::uint32_t kStage>
constexpr Multipliers<kStage> kMultipliers;
constexpr const Multipliers<1>& kStage1 = kMultipliers<1>;
constexpr const Multipliers<2>& kStage2 = kMultipliers<2>;

// Tiny multipliers m <= kTinyBound go over a segment in L1-sized pieces (32 KiB), so one fetch of
// a target line from L2 serves all of them.
constexpr std::uint32_t kTinyBound = 512, kZetaPiece = 4096, kMoebiusPiece = 8192;

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

// pairs[k] = (a_k, b_k R mod P) for 1 <= k <= n, with the zeta pass of p = 3 (pairs[k] +=
// pairs[k / 3] for k = 0 mod 3, ascending). a lies in the second half of pairs, at word offset
// a - pairs: pair k overwrites only a values below k, which are read already.
void interleave_zeta3(const AliasWord* a, const std::uint32_t* b, std::uint64_t* pairs, std::uint32_t n) {
    const Vec r2 = broadcast(kR2);
    // b_k R as Montgomery products of b_k and R^2: dwords [b R] in the high halves.
    auto scaled_pairs = [&](std::uint32_t k, Vec& low, Vec& high) {
        const Vec va = load(a + k), vb = load(b + k);
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
        const std::uint32_t br = std::uint32_t((std::uint64_t(b[k]) << 32) % kP);
        const std::uint64_t pair = a[k] | std::uint64_t(br) << 32;
        std::memcpy(pairs + k, &pair, sizeof pair);
        if (k % 3 == 0) add_pair(pairs + k, pairs + k / 3);
    };
    std::uint32_t k = 1;
    for (; k < 24 && k <= n; ++k) one(k);
    // 24 targets from 8 sources below them: multiples of 3 at offsets 0, 3 | 6 | 9 of each 12.
    for (; k + 23 <= n; k += 24) {
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
    for (; k <= n; ++k) one(k);
}

// Zeta pass of p >= 5 on the pairs: pairs[i p] += pairs[i], i ascending.
void zeta_pass(std::uint64_t* pairs, std::uint32_t n, std::uint32_t p) {
    for (std::uint32_t i = 1, last = n / p; i <= last; ++i) add_pair(pairs + i * p, pairs + i);
}

// visit(m) for the multipliers m of a stage with i m <= last, ascending.
template <std::uint32_t kStage, class Visit>
[[gnu::always_inline]] inline void for_multipliers(const Multipliers<kStage>& s, std::uint32_t i, std::uint32_t last, Visit visit) {
    std::uint32_t k = 0;
    for (; k < Multipliers<kStage>::kSmall && s.small[k] * i <= last; ++k) visit(s.small[k]);
    if (k < Multipliers<kStage>::kSmall) return;
    for (const std::uint32_t* m = s.large + 1; *m * i <= last; ++m) visit(*m);
}

// One stage on the targets in [start, last], segments descending: pairs[i m] += source[i].
// Tiny m take the segment in pieces, ascending (descending pieces measured slower).
template <std::uint32_t kStage>
void zeta_segment(std::uint64_t* pairs, const std::uint64_t* source, Sweep<kStage>& s, std::uint32_t start, std::uint32_t last) {
    for (std::uint32_t bottom = start; bottom <= last; bottom += kZetaPiece) {
        const std::uint32_t top = std::min(last, bottom + kZetaPiece - 1);
        for (std::uint32_t k = 0; k < Sweep<kStage>::kTiny; ++k) {
            const std::uint32_t m = s.m.small[k], end = divide(top, s.m.reciprocal[k]);
            for (std::uint32_t i = divide(bottom - 1, s.m.reciprocal[k]) + 1; i <= end; ++i) add_pair(pairs + i * m, source + i);
        }
    }
    for (std::uint32_t k = Sweep<kStage>::kTiny; k < Multipliers<kStage>::kSmall; ++k) {
        const std::uint32_t m = s.m.small[k], first = divide(start - 1, s.m.reciprocal[k]) + 1;
        for (std::uint32_t i = first; i <= s.run[k]; ++i) add_pair(pairs + i * m, source + i);
        s.run[k] = first - 1;
    }
    for (std::uint32_t i = 1; i * (kSplit + 1) <= last; ++i) {
        const Half v = load_pair(source + i);
        std::uint32_t j = s.edge[i];
        for (; s.m.large[j - 1] * i >= start; --j) {
            std::uint64_t* const target = pairs + s.m.large[j - 1] * i;
            store_pair(target, add(load_pair(target), v));
        }
        s.edge[i] = j;
    }
}

// x[i m] += source[i] over the stage's m, i m <= last, i descending (in place: x[i] still old).
template <std::uint32_t kStage>
void zeta_by_source(std::uint64_t* x, const std::uint64_t* source, const Multipliers<kStage>& s, std::uint32_t last) {
    for (std::uint32_t i = last / s.small[0]; i >= 1; --i) {
        const Half v = load_pair(source + i);
        for_multipliers(s, i, last, [&](std::uint32_t m) { store_pair(x + m * i, add(load_pair(x + m * i), v)); });
    }
}

// The zeta passes of all primes >= 17: stage 1, then stage 2. Stage 2 reads its sources (up to
// n / kStage2Min) from `prefix`, a copy with stage 1 applied. Target segments descend, so stage 1
// sources (below the segment) are unchanged when read. Segment 0: stage 1 in place, then stage 2.
void zeta_rough(std::uint64_t* pairs, std::uint32_t n, std::uint64_t* prefix) {
    const std::uint32_t prefix_last = n / kStage2Min;
    std::memcpy(prefix, pairs, (prefix_last + 1) * sizeof(std::uint64_t));
    zeta_by_source(prefix, prefix, kStage1, prefix_last);
    constexpr std::uint32_t kSegment = 1 << 15;  // 256 KiB of pairs
    if (n >= kSegment) {
        static Sweep<1> one{kStage1};
        static Sweep<2> two{kStage2};
        auto init = [&](auto& s) {
            for (std::uint32_t k = 0; k < std::size(s.run); ++k) s.run[k] = n / s.m.small[k];
            for (std::uint32_t i = 1; i * (kSplit + 1) <= n; ++i) s.edge[i] = s.large_end(i, n);
        };
        init(one);
        init(two);
        for (std::uint32_t start = n / kSegment * kSegment; start >= kSegment; start -= kSegment) {
            const std::uint32_t last = std::min(n, start + kSegment - 1);
            zeta_segment(pairs, pairs, one, start, last);
            zeta_segment(pairs, prefix, two, start, last);
        }
    }
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    zeta_by_source(pairs, pairs, kStage1, last0);
    zeta_by_source(pairs, prefix, kStage2, last0);
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
    for (; k < 8 && k <= n; ++k) one(k);
    const Vec even_lanes = _mm256_setr_epi32(0, 4, 1, 4, 2, 4, 3, 4);  // dword 4 is zero
    for (; k + 7 <= n; k += 8) {  // 8 targets, 4 sources k / 2 below them
        const Vec s = load(pairs + k / 2);
        const Vec t0 = add(load(pairs + k), spread<0x10, 0x33>(s));
        const Vec t1 = add(load(pairs + k + 4), spread<0x32, 0x33>(s));
        store(pairs + k, t0);
        store(pairs + k + 4, t1);
        const Vec halves = _mm256_permutevar8x32_epi32(high_dwords(product4(s), _mm256_setzero_si256()), even_lanes);
        store(c + k, sub(high_dwords(product4(t0), product4(t1)), halves));
    }
    for (; k <= n; ++k) one(k);
}

// Moebius pass of p on c: c[i p] -= c[i], i descending.
void moebius_pass(std::uint32_t* c, std::uint32_t n, std::uint32_t p) {
    for (std::uint32_t i = n / p; i >= 1; --i) c[i * p] = sub(c[i * p], c[i]);
}

// Small multipliers k in [k0, k1) on the targets from their carried bound up to last.
template <std::uint32_t kStage>
void moebius_small(std::uint32_t* c, const std::uint32_t* source, Sweep<kStage>& s, std::uint32_t k0, std::uint32_t k1, std::uint32_t last) {
    for (std::uint32_t k = k0; k < k1; ++k) {
        const std::uint32_t m = s.m.small[k], last_source = divide(last, s.m.reciprocal[k]);
        for (std::uint32_t i = s.run[k]; i <= last_source; ++i) c[i * m] = sub(c[i * m], source[i]);
        s.run[k] = last_source + 1;
    }
}

// One stage on the targets in [start, last], segments ascending: c[i m] -= source[i].
template <std::uint32_t kStage>
void moebius_segment(std::uint32_t* c, const std::uint32_t* source, Sweep<kStage>& s, std::uint32_t start, std::uint32_t last) {
    for (std::uint32_t bottom = start; bottom <= last; bottom += kMoebiusPiece)
        moebius_small(c, source, s, 0, Sweep<kStage>::kTiny, std::min(last, bottom + kMoebiusPiece - 1));
    moebius_small(c, source, s, Sweep<kStage>::kTiny, Multipliers<kStage>::kSmall, last);
    for (std::uint32_t i = 1; i * (kSplit + 1) <= last; ++i) {
        const std::uint32_t v = source[i];
        std::uint32_t j = s.edge[i];
        for (; s.m.large[j] * i <= last; ++j) c[s.m.large[j] * i] = sub(c[s.m.large[j] * i], v);
        s.edge[i] = j;
    }
}

// x[i m] -= source[i] over the stage's m, i m <= last, i ascending (in place: x[i] final).
template <std::uint32_t kStage>
void moebius_by_source(std::uint32_t* x, const std::uint32_t* source, const Multipliers<kStage>& s, std::uint32_t last) {
    for (std::uint32_t i = 1; s.small[0] * i <= last; ++i) {
        const std::uint32_t v = source[i];
        for_multipliers(s, i, last, [&](std::uint32_t m) { x[m * i] = sub(x[m * i], v); });
    }
}

// The Moebius passes of all primes >= 17 (inverting zeta_rough): stage 2, then stage 1, each as
// c[i m] -= final c[i]. Stage 2's final values up to n / kStage2Min go to `prefix` first.
// Segment 0: stage 2, then stage 1 in place; then target segments ascend, so stage 1 sources
// (below the segment) are final when read.
void moebius_rough(std::uint32_t* c, std::uint32_t n, std::uint32_t* prefix) {
    const std::uint32_t prefix_last = n / kStage2Min;
    std::memcpy(prefix, c, (prefix_last + 1) * sizeof(std::uint32_t));
    moebius_by_source(prefix, prefix, kStage2, prefix_last);
    constexpr std::uint32_t kSegment = 1 << 16;  // 256 KiB
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
        moebius_segment(c, c, one, start, last);
        moebius_segment(c, prefix, two, start, last);
    }
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    // One region: the pairs (2 words each), with a parsed into their second half, later c; then
    // b, later the rough sweeps' prefix copies; then the output text.
    constexpr std::size_t kPad = 64;
    const std::size_t words = n + kPad;
    const std::size_t text_offset = (3 * words * sizeof(std::uint32_t) + 63) / 64 * 64;
    char* const base = mem::huge<char>(text_offset + fields::kTextBytes);
    auto* const region = reinterpret_cast<std::uint32_t*>(base);
    auto* const pairs = reinterpret_cast<std::uint64_t*>(region);
    std::uint32_t* const a = region + words;
    std::uint32_t* const b = region + 2 * words;
    io::read_bulk(in, a + 1, n);
    io::read_bulk(in, b + 1, n);

    interleave_zeta3(a, b, pairs, n);
    for (const std::uint32_t p : {5, 7, 11, 13}) zeta_pass(pairs, n, p);
    auto* const prefix = reinterpret_cast<std::uint64_t*>(b);  // b is dead after the interleave
    zeta_rough(pairs, n, prefix);
    std::uint32_t* const c = region;
    zeta2_product_moebius2(pairs, c, n);
    for (const std::uint32_t p : {3, 5, 7, 11, 13}) moebius_pass(c, n, p);
    moebius_rough(c, n, reinterpret_cast<std::uint32_t*>(prefix));

    io::Writer out;
    fields::write(out, c + 1, n, base + text_offset);
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
