// c_k = sum over gcd(i, j) = k of a_i b_j mod 998244353, 1 <= k <= N <= 10^6.
// Sums over multiples (zeta) of a and b, pointwise product, then the inverse (Moebius). Both are
// products of commuting per-prime passes. Primes 2..13: one pass each (zeta: x_i += x_ip, i
// descending; Moebius: x_i -= x_ip, i ascending). Primes >= 17 together: one sweep over the
// multipliers m coprime to 30030 ("rough"), segment by segment so the sources stay in L2 (in L1
// for m < 256).
// a and b are interleaved as pairs, so one load fetches both.
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

// Sum over the rough m with index in [t0, t1) of term(i m), in two interleaved chains. i m < 2^32.
template <class T, class Term>
[[gnu::always_inline]] inline T sum_multiples(std::uint32_t i, std::uint32_t t0, std::uint32_t t1, T sum, Term term) {
    T other{};
    std::uint32_t k = t0 % kSpokes, base = t0 / kSpokes * kWheel * i;
    for (std::uint32_t left = t1 - t0; left; k = 0, base += kWheel * i) {
        const std::uint32_t end = std::min(kSpokes, k + left);
        left -= end - k;
        for (; k + 2 <= end; k += 2) {
            sum += term(base + i * kRough.spoke[k]);
            other += term(base + i * kRough.spoke[k + 1]);
        }
        if (k < end) sum += term(base + i * kRough.spoke[k]);
    }
    return sum + other;
}

// floor(x / d) = (x r) >> 42 with r = reciprocal(d), for x, d < 2^20.
constexpr std::uint64_t reciprocal(std::uint32_t d) { return (std::uint64_t(1) << 42) / d + 1; }
inline std::uint32_t divide(std::uint32_t x, std::uint64_t r) { return std::uint32_t(x * r >> 42); }

// Rough multipliers 17 <= m <= kSplit go by m, each over a run of targets i; larger ones by
// target i <= n / (kSplit + 1), into 64-bit sums. Sources come in segments that fit in L2.
constexpr std::uint32_t kSplit = 2048;
constexpr std::uint32_t kSmall = kRough.index(kSplit + 1);  // small m: spoke[1..kSmall)
// Tiny m (17..251, spoke[1..kTiny)) take their sources by L1-sized pieces of each segment.
constexpr std::uint32_t kTiny = kRough.index(256);
constexpr std::uint32_t kMaxN = 1 << 20;
constexpr std::uint32_t kMaxTargets = kMaxN / (kSplit + 1) + 1;

constexpr auto kSmallReciprocal = [] {
    std::array<std::uint64_t, kSmall> r{};
    for (std::uint32_t k = 1; k < kSmall; ++k) r[k] = reciprocal(kRough.spoke[k]);
    return r;
}();
constexpr auto kTargetReciprocal = [] {
    std::array<std::uint64_t, kMaxTargets> r{};
    for (std::uint32_t i = 1; i < kMaxTargets; ++i) r[i] = reciprocal(i);
    return r;
}();

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

// pairs[i] += sum of pairs[i m] over rough m > 1, im <= n: the zeta passes of all primes >= 17.
// Sources go segment by segment, ascending, so each is read before it changes: the targets of a
// segment's sources lie in earlier segments. Segment 0 goes by target, ascending.
void zeta_rough(std::uint64_t* pairs, std::uint32_t n) {
    constexpr std::uint32_t kSegment = 1 << 15, kPiece = 1 << 12;  // pairs: 256 KiB, 32 KiB
    auto widened = [&](std::uint32_t j) { return widen_pair(pairs + j); };
    const std::uint32_t end0 = std::min(n + 1, kSegment);
    for (std::uint32_t i = 1; 17 * i < end0; ++i)  // below 2^15 terms of 2^30 per lane
        add_sums(pairs + i, sum_multiples(i, 1, kRough.index((end0 - 1) / i + 1), Half{}, widened));
    if (n < kSegment) return;
    // Per small m: the next target. Per large target: the next multiplier index, and the sums
    // (below 2^18 terms of 2^30 per lane).
    static std::uint32_t next_target[kSmall], next_index[kMaxTargets];
    static Half sums[kMaxTargets];
    for (std::uint32_t k = 1; k < kSmall; ++k) next_target[k] = (kSegment - 1) / kRough.spoke[k] + 1;
    const std::uint32_t targets = n / (kSplit + 1);
    for (std::uint32_t i = 1; i <= targets; ++i)
        next_index[i] = std::max(kSmall, kRough.index((kSegment - 1) / i + 1));
    for (std::uint32_t start = kSegment; start <= n; start += kSegment) {
        const std::uint32_t last_source = std::min(n, start + kSegment - 1);
        // Multipliers [1, k_end) with sources up to last.
        auto by_multiplier = [&](std::uint32_t k_end, std::uint32_t last_source) {
            for (std::uint32_t k = 1; k < k_end; ++k) {
                const std::uint32_t m = kRough.spoke[k], last = divide(last_source, kSmallReciprocal[k]);
                std::uint32_t i = next_target[k];
                for (; i + 3 <= last; i += 4) store(pairs + i, add(load(pairs + i), gather_pairs(pairs + i * m, m)));
                for (; i <= last; ++i) add_pair(pairs + i, pairs + i * m);
                next_target[k] = i;
            }
        };
        for (std::uint32_t piece = start + kPiece; piece <= last_source; piece += kPiece) by_multiplier(kTiny, piece - 1);
        by_multiplier(kSmall, last_source);
        for (std::uint32_t i = 1; i * (kSplit + 1) <= last_source; ++i) {
            const std::uint32_t end = kRough.index(divide(last_source, kTargetReciprocal[i]) + 1);
            sums[i] = sum_multiples(i, next_index[i], end, sums[i], widened);
            next_index[i] = end;
        }
    }
    for (std::uint32_t i = 1; i <= targets; ++i) add_sums(pairs + i, sums[i]);
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

// c_i -= sum of c_im over rough m > 1, im <= n, with final values c_im: the Moebius passes of all
// primes >= 17 (inverting zeta_rough: x_i = y_i - sum over m > 1 of x_im). Segments descend, so a
// segment is final when read; segment 0 goes by target, descending.
void moebius_rough(std::uint32_t* c, std::uint32_t n) {
    constexpr std::uint32_t kSegment = 1 << 16, kPiece = 1 << 13;  // 256 KiB, 32 KiB
    auto value = [&](std::uint32_t j) { return std::uint64_t(c[j]); };
    if (n >= kSegment) {
        // Per small m: the last target. Per large target: the end of its multiplier indices,
        // and the sums (below 2^18 terms of 2^30).
        static std::uint32_t last_target[kSmall], end_index[kMaxTargets];
        static std::uint64_t sums[kMaxTargets];
        for (std::uint32_t k = 1; k < kSmall; ++k) last_target[k] = n / kRough.spoke[k];
        const std::uint32_t targets = n / (kSplit + 1);
        for (std::uint32_t i = 1; i <= targets; ++i) end_index[i] = kRough.index(n / i + 1);
        for (std::uint32_t start = n / kSegment * kSegment; start >= kSegment; start -= kSegment) {
            // Multipliers [1, k_end) with sources from first on.
            auto by_multiplier = [&](std::uint32_t k_end, std::uint32_t first) {
                for (std::uint32_t k = 1; k < k_end; ++k) {
                    const std::uint32_t m = kRough.spoke[k], last = last_target[k];
                    std::uint32_t i = divide(first - 1, kSmallReciprocal[k]) + 1;
                    last_target[k] = i - 1;
                    for (; i + 7 <= last; i += 8) store(c + i, sub(load(c + i), gather_dwords(c + i * m, m)));
                    for (; i <= last; ++i) c[i] = sub(c[i], c[i * m]);
                }
            };
            const std::uint32_t last_source = std::min(n, start + kSegment - 1);
            for (std::uint32_t piece = start + (last_source - start) / kPiece * kPiece; piece > start; piece -= kPiece)
                by_multiplier(kTiny, piece);
            by_multiplier(kSmall, start);
            for (std::uint32_t i = 1; i * (kSplit + 1) <= last_source; ++i) {
                const std::uint32_t begin =
                    std::max(kSmall, kRough.index(divide(start - 1, kTargetReciprocal[i]) + 1));
                if (begin < end_index[i]) sums[i] = sum_multiples(i, begin, end_index[i], sums[i], value);
                end_index[i] = begin;
            }
        }
        for (std::uint32_t i = 1; i <= targets; ++i) c[i] = sub(c[i], std::uint32_t(sums[i] % kP));
    }
    const std::uint32_t end0 = std::min(n + 1, kSegment);
    for (std::uint32_t i = (end0 - 1) / 17; i >= 1; --i) {
        const std::uint64_t sum = sum_multiples(i, 1, kRough.index((end0 - 1) / i + 1), std::uint64_t{}, value);
        c[i] = sub(c[i], std::uint32_t(sum % kP));
    }
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    // One region: the pairs, with a parsed into their first half; then b, later c. The pairs'
    // first huge page later holds the output text.
    constexpr std::size_t kPad = 64;
    const std::size_t words = n + kPad;
    static_assert(fields::kTextBytes <= std::size_t(1) << 21);
    auto* region = allocate<std::uint32_t>(std::max(3 * words, fields::kTextBytes / sizeof(std::uint32_t)));
    auto* pairs = reinterpret_cast<std::uint64_t*>(region);
    std::uint32_t* const b = region + 2 * words;
    in.read(region + 1, n);
    in.read(b + 1, n);

    interleave_zeta3(region, b, pairs, n);
    for (const std::uint32_t p : {5, 7, 11, 13}) zeta_pass(pairs, n, p);
    zeta_rough(pairs, n);
    std::uint32_t* const c = b;
    zeta2_product_moebius2(pairs, c, n);
    for (const std::uint32_t p : {3, 5, 7, 11, 13}) moebius_pass(c, n, p);
    moebius_rough(c, n);

    io::Writer out;
    fields::write(out, c + 1, n, reinterpret_cast<char*>(region));
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here), and _exit skips their teardown.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
