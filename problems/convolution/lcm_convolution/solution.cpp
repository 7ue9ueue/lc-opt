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

// Zeta pass of p = 2 fused with the product and the Moebius pass of 2, ascending:
// A_k += A_k/2, then c_k = A_k B_k - A_k/2 B_k/2 (the second term for even k).
void zeta2_product_moebius2(std::uint64_t* pairs, std::uint32_t* c, std::uint32_t n) {
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

    interleave_zeta3(a, b, pairs, n);
    for (const std::uint32_t p : {5, 7, 11, 13}) zeta_pass(pairs, n, p);
    zeta_rough(pairs, n);
    std::uint32_t* const c = b;
    zeta2_product_moebius2(pairs, c, n);
    for (const std::uint32_t p : {3, 5, 7, 11, 13}) moebius_pass(c, n, p);
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
