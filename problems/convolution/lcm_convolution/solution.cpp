// c_k = sum over lcm(i, j) = k of a_i b_j mod 998244353, 1 <= k <= N <= 10^6.
// Sums over divisors (zeta) of a and b, pointwise product, then the inverse (Moebius). Both are
// products of commuting per-prime passes. Primes 2..13: one pass each (zeta: x_ip += x_i, i
// ascending; Moebius: x_ip -= x_i, i descending). Primes >= 17 together: one sweep over the
// multipliers m coprime to 30030 ("rough"), target segment by target segment, in L2.
// a and b are interleaved as pairs, so one load fetches both. Design and measurements: notes.md.
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "lib/io/io.hpp"
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

// Zeroed memory in 2 MiB pages where the kernel allows. Never freed.
char* allocate(std::size_t bytes) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    bytes = (bytes + kHuge - 1) / kHuge * kHuge;
    void* p = ::mmap(nullptr, bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) std::abort();
    const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(p) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
    ::madvise(reinterpret_cast<void*>(aligned), bytes, MADV_HUGEPAGE);
#endif
    return reinterpret_cast<char*>(aligned);
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
};

constexpr Wheel kRough;

// visit(i m) for the rough m with index in [t0, t1), t0 <= t1, ascending. i m < 2^32.
template <class Visit>
[[gnu::always_inline]] inline void for_multiples(std::uint32_t i, std::uint32_t t0, std::uint32_t t1, Visit visit) {
    std::uint32_t k = t0 % kSpokes, base = t0 / kSpokes * kWheel * i;
    for (std::uint32_t left = t1 - t0; left; k = 0, base += kWheel * i) {
        const std::uint32_t end = std::min(kSpokes, k + left);
        left -= end - k;
        for (; k < end; ++k) visit(base + i * kRough.spoke[k]);
    }
}

// floor(x / d) = (x r) >> 42 with r = reciprocal(d), for x, d < 2^20.
constexpr std::uint64_t reciprocal(std::uint32_t d) { return (std::uint64_t(1) << 42) / d + 1; }
inline std::uint32_t divide(std::uint32_t x, std::uint64_t r) { return std::uint32_t(x * r >> 42); }

// In a segment of targets, rough multipliers 17 <= m <= kSplit go by m over a run of sources i;
// larger ones by source i <= n / (kSplit + 1), over a run of m.
constexpr std::uint32_t kSplit = 2048;
constexpr std::uint32_t kSmall = kRough.index(kSplit + 1);  // small m: spoke[1..kSmall)
constexpr std::uint32_t kMaxN = 1 << 20;
constexpr std::uint32_t kMaxSources = kMaxN / (kSplit + 1) + 1;

constexpr auto kSmallReciprocal = [] {
    std::array<std::uint64_t, kSmall> r{};
    for (std::uint32_t k = 1; k < kSmall; ++k) r[k] = reciprocal(kRough.spoke[k]);
    return r;
}();
constexpr auto kSourceReciprocal = [] {
    std::array<std::uint64_t, kMaxSources> r{};
    for (std::uint32_t i = 1; i < kMaxSources; ++i) r[i] = reciprocal(i);
    return r;
}();

// The wheel indices of the large rough m with i m in [start, last], start >= 1.
struct Range {
    std::uint32_t begin, end;
};
inline Range large_multipliers(std::uint32_t i, std::uint32_t start, std::uint32_t last) {
    const std::uint64_t r = kSourceReciprocal[i];
    return {std::max(kSmall, kRough.index(divide(start - 1, r) + 1)), kRough.index(divide(last, r) + 1)};
}

// Zeta passes of the primes 3..13 fused into the interleave: kFused of them (a prefix of
// kOddPrimes). The full transform F of the fused primes satisfies F_k = x_k + sum over the
// squarefree products d > 1 of fused primes dividing k of -mu(d) F_k/d. d = 3 goes by vectors,
// the other terms by source.
#ifndef FUSED
#define FUSED 3
#endif
#ifndef FUSED_M
#define FUSED_M 3
#endif
constexpr std::uint32_t kOddPrimes[] = {3, 5, 7, 11, 13};
constexpr int kFusedZeta = FUSED, kFusedMoebius = FUSED_M;

struct Term {
    std::uint32_t d;
    bool odd;  // d has an odd number of prime factors: mu(d) = -1
};

// The squarefree products of the first Fused odd primes, except 1 and 3.
template <int Fused>
constexpr auto terms() {
    std::array<Term, (1 << Fused) - 2> terms{};
    for (int set = 2; set < (1 << Fused); ++set) {
        std::uint32_t d = 1;
        int primes = 0;
        for (int i = 0; i < Fused; ++i)
            if (set >> i & 1) d *= kOddPrimes[i], ++primes;
        terms[set - 2] = {d, primes % 2 == 1};
    }
    return terms;
}

constexpr auto kZetaTerms = terms<kFusedZeta>();
constexpr auto kMoebiusTerms = terms<kFusedMoebius>();

void sub_pair(std::uint64_t* dst, const std::uint64_t* src) {
    const Half d = _mm_sub_epi32(load_pair(dst), load_pair(src));
    store_pair(dst, _mm_min_epu32(d, _mm_add_epi32(d, _mm_set1_epi32(int(kP)))));
}

// pairs[k] = (a_k, b_k R mod P) for 1 <= k <= n, with the fused zeta passes, ascending. a lies in
// the second half of pairs: pair k overwrites only a values below k, which are read already.
// Blocks [k0, k1) with k1 <= 3 k0 take all sources from below k0, which are final.
void interleave_zeta(const AliasWord* a, const std::uint32_t* b, std::uint64_t* pairs, std::uint32_t n) {
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
    // Target k with the term d = 3.
    auto one = [&](std::uint32_t k) {
        const std::uint32_t br = std::uint32_t((std::uint64_t(b[k]) << 32) % kP);
        const std::uint64_t pair = a[k] | std::uint64_t(br) << 32;
        std::memcpy(pairs + k, &pair, sizeof pair);
        if (k % 3 == 0) add_pair(pairs + k, pairs + k / 3);
    };
    constexpr std::uint32_t kFirst = 24, kBlock = 1536;  // multiples of 24
    std::uint32_t k = 1;
    for (; k < kFirst && k <= n; ++k) {
        one(k);
        for (const Term& t : kZetaTerms)
            if (k % t.d == 0) (t.odd ? add_pair : sub_pair)(pairs + k, pairs + k / t.d);
    }
    std::uint32_t next[kZetaTerms.size()];  // per term: the next source
    for (std::size_t t = 0; t < kZetaTerms.size(); ++t) next[t] = (kFirst + kZetaTerms[t].d - 1) / kZetaTerms[t].d;
    while (k <= n) {
        const std::uint32_t end = std::min({k + kBlock, 3 * k, n + 1});
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
        for (std::size_t t = 0; t < kZetaTerms.size(); ++t) {
            const std::uint32_t d = kZetaTerms[t].d;
            std::uint32_t j = next[t];
            if (kZetaTerms[t].odd)
                for (; j * d < end; ++j) add_pair(pairs + j * d, pairs + j);
            else
                for (; j * d < end; ++j) sub_pair(pairs + j * d, pairs + j);
            next[t] = j;
        }
    }
}

// Zeta pass of p >= 5 on the pairs: pairs[i p] += pairs[i], i ascending.
void zeta_pass(std::uint64_t* pairs, std::uint32_t n, std::uint32_t p) {
    for (std::uint32_t i = 1, last = n / p; i <= last; ++i) add_pair(pairs + i * p, pairs + i);
}

// pairs[i m] += pairs[i] over rough m > 1, i m <= n, with the values before this pass: the zeta
// passes of all primes >= 17. Target segments descend, so sources (below the segment) are
// unchanged when read. Segment 0 goes by source, descending.
void zeta_rough(std::uint64_t* pairs, std::uint32_t n) {
    constexpr std::uint32_t kSegment = 1 << 15;  // 256 KiB of pairs
    if (n >= kSegment) {
        static std::uint32_t last_source[kSmall];  // per small m, for the next segment
        for (std::uint32_t k = 1; k < kSmall; ++k) last_source[k] = n / kRough.spoke[k];
        for (std::uint32_t start = n / kSegment * kSegment; start >= kSegment; start -= kSegment) {
            const std::uint32_t last = std::min(n, start + kSegment - 1);
            for (std::uint32_t k = 1; k < kSmall; ++k) {
                const std::uint32_t m = kRough.spoke[k], first = divide(start - 1, kSmallReciprocal[k]) + 1;
                for (std::uint32_t i = first; i <= last_source[k]; ++i) add_pair(pairs + i * m, pairs + i);
                last_source[k] = first - 1;
            }
            for (std::uint32_t i = 1; i * (kSplit + 1) <= last; ++i) {
                const Range r = large_multipliers(i, start, last);
                if (r.begin >= r.end) continue;
                const Half v = load_pair(pairs + i);
                for_multiples(i, r.begin, r.end, [&](std::uint32_t j) { store_pair(pairs + j, add(load_pair(pairs + j), v)); });
            }
        }
    }
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    for (std::uint32_t i = last0 / 17; i >= 1; --i) {
        const Half v = load_pair(pairs + i);
        for_multiples(i, 1, kRough.index(last0 / i + 1), [&](std::uint32_t j) { store_pair(pairs + j, add(load_pair(pairs + j), v)); });
    }
}

// Zeta pass of p = 2 fused with the product and the Moebius passes of 2 and of the first
// kFusedMoebius odd primes, ascending. A_k += A_k/2, then C_k = A_k B_k - A_k/2 B_k/2 (the second
// term for even k) into products, dwords over the pairs: C_k overwrites pair k / 2, which is
// used for the last time at target k. c_k = C_k + sum over the squarefree products d > 1 of the
// fused odd primes dividing k of mu(d) C_k/d: d = 3 by vectors, the rest by source per block.
void zeta2_product_moebius(std::uint64_t* pairs, std::uint32_t* c, std::uint32_t n) {
    AliasWord* const products = reinterpret_cast<AliasWord*>(pairs);
    auto one = [&](std::uint32_t k) {
        std::uint32_t ai = std::uint32_t(pairs[k]), bi = std::uint32_t(pairs[k] >> 32);
        std::uint32_t ck;
        if (k % 2) {
            ck = product(ai, bi);
        } else {
            const std::uint64_t s = pairs[k / 2];
            ai = add(ai, std::uint32_t(s));
            bi = add(bi, std::uint32_t(s >> 32));
            pairs[k] = ai | std::uint64_t(bi) << 32;
            ck = sub(product(ai, bi), product(std::uint32_t(s), std::uint32_t(s >> 32)));
        }
        products[k] = ck;
        c[k] = k % 3 ? ck : sub(ck, products[k / 3]);
    };
    constexpr std::uint32_t kFirst = 24, kBlock = 1536;  // multiples of 24
    std::uint32_t k = 1;
    for (; k < kFirst && k <= n; ++k) {
        one(k);
        for (const Term& t : kMoebiusTerms)
            if (k % t.d == 0) c[k] = t.odd ? sub(c[k], products[k / t.d]) : add(c[k], products[k / t.d]);
    }
    std::uint32_t next[kMoebiusTerms.size()];  // per term: the next source
    for (std::size_t t = 0; t < kMoebiusTerms.size(); ++t)
        next[t] = (kFirst + kMoebiusTerms[t].d - 1) / kMoebiusTerms[t].d;
    const Vec even_lanes = _mm256_setr_epi32(0, 4, 1, 4, 2, 4, 3, 4);  // dword 4 is zero
    // Multiples of 3 among 24 targets: per group of 8, the source lanes (8 = none).
    const Vec thirds[3] = {_mm256_setr_epi32(0, 8, 8, 1, 8, 8, 2, 8), _mm256_setr_epi32(8, 3, 8, 8, 4, 8, 8, 5),
                           _mm256_setr_epi32(8, 8, 6, 8, 8, 7, 8, 8)};
    while (k <= n) {
        const std::uint32_t end = std::min(k + kBlock, n + 1);
        for (; k + 24 <= end; k += 24) {
            const Vec third = load(products + k / 3);
            for (int g = 0; g < 3; ++g) {
                const std::uint32_t i = k + 8 * g;  // 8 targets, 4 sources i / 2 below them
                const Vec s = load(pairs + i / 2);
                const Vec t0 = add(load(pairs + i), spread<0x10, 0x33>(s));
                const Vec t1 = add(load(pairs + i + 4), spread<0x32, 0x33>(s));
                store(pairs + i, t0);
                store(pairs + i + 4, t1);
                const Vec halves = _mm256_permutevar8x32_epi32(high_dwords(product4(s), _mm256_setzero_si256()), even_lanes);
                const Vec ci = sub(high_dwords(product4(t0), product4(t1)), halves);
                store(products + i, ci);
                // vpermd uses the low 3 bits of each index: lanes with index 8 read lane 0, then cleared.
                const Vec from = _mm256_permutevar8x32_epi32(third, thirds[g]);
                const Vec mask = _mm256_cmpgt_epi32(broadcast(8), thirds[g]);
                store(c + i, sub(ci, _mm256_and_si256(from, mask)));
            }
        }
        for (; k < end; ++k) one(k);
        for (std::size_t t = 0; t < kMoebiusTerms.size(); ++t) {
            const std::uint32_t d = kMoebiusTerms[t].d;
            std::uint32_t j = next[t];
            if (kMoebiusTerms[t].odd)
                for (; j * d < end; ++j) c[j * d] = sub(c[j * d], products[j]);
            else
                for (; j * d < end; ++j) c[j * d] = add(c[j * d], products[j]);
            next[t] = j;
        }
    }
}

// Moebius pass of p on c: c[i p] -= c[i], i descending.
void moebius_pass(std::uint32_t* c, std::uint32_t n, std::uint32_t p) {
    for (std::uint32_t i = n / p; i >= 1; --i) c[i * p] = sub(c[i * p], c[i]);
}

// c[i m] -= c[i] over rough m > 1, i m <= n, with final values c[i]: the Moebius passes of all
// primes >= 17 (inverting zeta_rough). Segment 0 goes by source, ascending; then target
// segments ascend, so sources (below the segment) are final when read.
void moebius_rough(std::uint32_t* c, std::uint32_t n) {
    constexpr std::uint32_t kSegment = 1 << 16;  // 256 KiB
    const std::uint32_t last0 = std::min(n, kSegment - 1);
    for (std::uint32_t i = 1; 17 * i <= last0; ++i) {
        const std::uint32_t v = c[i];
        for_multiples(i, 1, kRough.index(last0 / i + 1), [&](std::uint32_t j) { c[j] = sub(c[j], v); });
    }
    if (n < kSegment) return;
    static std::uint32_t next_source[kSmall];  // per small m
    for (std::uint32_t k = 1; k < kSmall; ++k) next_source[k] = last0 / kRough.spoke[k] + 1;
    for (std::uint32_t start = kSegment; start <= n; start += kSegment) {
        const std::uint32_t last = std::min(n, start + kSegment - 1);
        for (std::uint32_t k = 1; k < kSmall; ++k) {
            const std::uint32_t m = kRough.spoke[k], last_source = divide(last, kSmallReciprocal[k]);
            for (std::uint32_t i = next_source[k]; i <= last_source; ++i) c[i * m] = sub(c[i * m], c[i]);
            next_source[k] = last_source + 1;
        }
        for (std::uint32_t i = 1; i * (kSplit + 1) <= last; ++i) {
            const Range r = large_multipliers(i, start, last);
            if (r.begin >= r.end) continue;
            const std::uint32_t v = c[i];
            for_multiples(i, r.begin, r.end, [&](std::uint32_t j) { c[j] = sub(c[j], v); });
        }
    }
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    // One region: the pairs (2 words each), with a parsed into their second half; then b, later
    // c; then the output text.
    constexpr std::size_t kPad = 64;
    const std::size_t words = n + kPad;
    const std::size_t text_offset = (3 * words * sizeof(std::uint32_t) + 63) / 64 * 64;
    char* const base = allocate(text_offset + fields::kTextBytes);
    auto* const region = reinterpret_cast<std::uint32_t*>(base);
    auto* const pairs = reinterpret_cast<std::uint64_t*>(region);
    std::uint32_t* const a = region + words;
    std::uint32_t* const b = region + 2 * words;
    in.read(a + 1, n);
    in.read(b + 1, n);

    interleave_zeta(a, b, pairs, n);
    for (int i = kFusedZeta; i < 5; ++i) zeta_pass(pairs, n, kOddPrimes[i]);
    zeta_rough(pairs, n);
    std::uint32_t* const c = b;
    zeta2_product_moebius(pairs, c, n);
    for (int i = kFusedMoebius; i < 5; ++i) moebius_pass(c, n, kOddPrimes[i]);
    moebius_rough(c, n);

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
