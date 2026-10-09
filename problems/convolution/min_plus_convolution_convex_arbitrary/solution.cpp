// c[k] = min over i + j = k of a[i] + b[j], a convex. Row k of the matrix b[j] + a[k - j] (columns
// j with 0 <= k - j < N) is Monge, so the leftmost argmin opt(k) is nondecreasing in k.
// Divide and conquer finds opt at every kGroup-th row (sample rows); each group of kGroup rows
// then takes its minima over the columns between its two sample opts, kGroup rows per column.
#include <sys/mman.h>
#include <unistd.h>

#include "lib/io/io.hpp"

namespace {

using u32 = std::uint32_t;

constexpr std::size_t kGroup = 16;         // rows per group: two vectors of 8
constexpr u32 kInf = 3'000'000'000u;       // a outside [0, N); kInf + b <= 4e9 < 2^32
constexpr std::size_t kAPad = kGroup + 8;  // kInf words before and after a

struct Problem {
    std::size_t n, m;  // lengths of a and b
    const u32* a;      // a[-kAPad, n + kAPad), kInf outside [0, n)
    const u32* b;      // b[0, m + 8)
    u32* opt;          // opt[t]: leftmost argmin of sample row t * kGroup; opt[groups] = m - 1
    u32* c;            // c[0, groups * kGroup)
};

u32 horizontal_min(__m256i v) {
    __m128i x = _mm_min_epu32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    x = _mm_min_epu32(x, _mm_shuffle_epi32(x, 0x4E));
    x = _mm_min_epu32(x, _mm_shuffle_epi32(x, 0xB1));
    return u32(_mm_cvtsi128_si32(x));
}

// b[j, j + 8) + a[k - j - 8 + 1, k - j + 1) reversed: row k at columns j..j+7.
__m256i row_chunk(const Problem& p, std::size_t k, std::size_t j) {
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.a + (k - j) - 7));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p.b + j));
    return _mm256_add_epi32(b, _mm256_permutevar8x32_epi32(a, reverse));
}

// Leftmost j in [lo, hi] minimizing b[j] + a[k - j]; every column in range is valid.
std::size_t argmin(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi) {
    if (hi - lo < 16) {
        std::size_t best = lo;
        u32 best_value = p.b[lo] + p.a[k - lo];
        for (std::size_t j = lo + 1; j <= hi; ++j) {
            const u32 value = p.b[j] + p.a[k - j];
            if (value < best_value) best = j, best_value = value;
        }
        return best;
    }
    // Chunks start at lo, lo + 8, ...; the last one at hi - 7 overlaps its predecessor.
    const std::size_t last = hi - 7;
    __m256i low = _mm256_set1_epi32(-1);
    for (std::size_t j = lo; j < last; j += 8) low = _mm256_min_epu32(low, row_chunk(p, k, j));
    low = _mm256_min_epu32(low, row_chunk(p, k, last));
    const __m256i target = _mm256_set1_epi32(int(horizontal_min(low)));
    for (std::size_t j = lo;; j = std::min(j + 8, last)) {
        const auto hits = u32(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(row_chunk(p, k, j), target))));
        if (hits) return j + std::countr_zero(hits);
    }
}

// opt of sample rows [t1, t2), known to lie in [lo, hi].
void sample(const Problem& p, std::size_t t1, std::size_t t2, std::size_t lo, std::size_t hi) {
    if (t1 == t2) return;
    const std::size_t t = (t1 + t2) / 2, k = t * kGroup;
    const std::size_t j = argmin(p, k, std::max(lo, k + 1 > p.n ? k + 1 - p.n : 0), std::min(hi, k));
    p.opt[t] = u32(j);
    sample(p, t1, t, lo, j);
    sample(p, t + 1, t2, j, hi);
}

// c[k0, k0 + kGroup) for group t: columns between the sample opts, clamped to rows' valid columns.
void group(const Problem& p, std::size_t t) {
    const std::size_t k0 = t * kGroup;
    const std::size_t lo = std::max(p.opt[t], k0 + 1 > p.n ? k0 + 1 - p.n : 0);
    const std::size_t hi = std::min(p.opt[t + 1], k0 + kGroup - 1);
    __m256i low0 = _mm256_set1_epi32(-1), low1 = low0;
    for (std::size_t j = lo; j <= hi; ++j) {
        const __m256i b = _mm256_set1_epi32(int(p.b[j]));
        const u32* a = p.a + (k0 - j);  // index >= -(kGroup - 1)
        low0 = _mm256_min_epu32(low0, _mm256_add_epi32(b, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))));
        low1 = _mm256_min_epu32(low1, _mm256_add_epi32(b, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 8))));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p.c + k0), low0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p.c + k0 + 8), low1);
}

// words u32 words, 2 MiB aligned, in huge pages where the kernel allows.
u32* allocate(std::size_t words) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    const std::size_t bytes = (words * sizeof(u32) + kHuge - 1) / kHuge * kHuge + kHuge;
    void* region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) std::abort();
    const std::uintptr_t start = (reinterpret_cast<std::uintptr_t>(region) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
    ::madvise(reinterpret_cast<void*>(start), bytes - kHuge, MADV_HUGEPAGE);
#endif
    return reinterpret_cast<u32*>(start);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>(), m = in.read<u32>(), count = n + m - 1;
    const std::size_t groups = (count + kGroup - 1) / kGroup;
    const std::size_t a_words = n + 2 * kAPad, b_words = m + 8, c_words = groups * kGroup;
    u32* const memory = allocate(a_words + b_words + c_words + groups + 1);
    u32* const a = memory;
    u32* const b = a + a_words;
    u32* const c = b + b_words;
    u32* const opt = c + c_words;
    std::fill_n(a, kAPad, kInf);
    in.read(a + kAPad, n);
    std::fill_n(a + kAPad + n, kAPad, kInf);
    in.read(b, m);

    const Problem p{n, m, a + kAPad, b, opt, c};
    sample(p, 0, groups, 0, m - 1);
    opt[groups] = u32(m - 1);
    for (std::size_t t = 0; t < groups; ++t) group(p, t);

    io::Writer out;
    out.write_array(c, count, ' ');
    out.write('\n');
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
