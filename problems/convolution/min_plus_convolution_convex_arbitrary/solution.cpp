// c[k] = min over i + j = k of a[i] + b[j], a convex. Row k of the matrix b[j] + a[k - j] (columns
// j with 0 <= k - j < N) is Monge, so the leftmost argmin opt(k) is nondecreasing in k.
// opt is found at every kGroup-th row (sample rows), coarse rows first, each searched between the
// opts of its neighbors; each group of kGroup rows then takes its minima over the columns between
// its two sample opts, kGroup rows per column.

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "lib/run/early.hpp"
#include "columns.hpp"

namespace {

using u32 = std::uint32_t;

constexpr std::size_t kGroup = 16;         // rows per group
constexpr std::size_t kVecs = kGroup / 8;  // vectors per group
constexpr u32 kInf = 3'000'000'000u;       // a outside [0, N); kInf + b <= 4e9 < 2^32
constexpr std::size_t kAPad = kGroup + 8;  // kInf words before and after a

struct Problem {
    std::size_t n, m;  // lengths of a and b
    const u32* a;      // a[-kAPad, n + kAPad), kInf outside [0, n)
    const u32* b;      // b[0, m)
    u32* opt;          // opt[t]: leftmost argmin of sample row t * kGroup; m - 1 for t >= groups
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

// Leftmost column among the lanes holding the least value.
u32 leftmost(__m256i value, __m256i column) {
    const __m256i least = _mm256_set1_epi32(int(horizontal_min(value)));
    const __m256i other = _mm256_xor_si256(_mm256_cmpeq_epi32(value, least), _mm256_set1_epi32(-1));
    return horizontal_min(_mm256_or_si256(column, other));
}

// Leftmost j in [lo, hi] minimizing b[j] + a[k - j]; every column in range is valid.
std::size_t argmin(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi) {
    if (hi - lo < 8) {
        std::size_t best = lo;
        u32 best_value = p.b[lo] + p.a[k - lo];
        for (std::size_t j = lo + 1; j <= hi; ++j) {
            const u32 value = p.b[j] + p.a[k - j];
            if (value < best_value) best = j, best_value = value;
        }
        return best;
    }
    // Per lane, the least value and its leftmost column. Chunks start at lo, lo + 8, ...; the
    // last one at hi - 7 overlaps its predecessor, still in increasing column order per lane.
    const std::size_t last = hi - 7;
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i column = _mm256_add_epi32(_mm256_set1_epi32(int(lo)), lanes);
    __m256i best = _mm256_set1_epi32(-1), best_column = column;
    const auto take = [&](std::size_t j) {
        const __m256i low = _mm256_min_epu32(best, row_chunk(p, k, j));
        best_column = _mm256_blendv_epi8(column, best_column, _mm256_cmpeq_epi32(low, best));
        best = low;
    };
    for (std::size_t j = lo; j < last; j += 8) {
        take(j);
        column = _mm256_add_epi32(column, _mm256_set1_epi32(8));
    }
    column = _mm256_add_epi32(_mm256_set1_epi32(int(last)), lanes);
    take(last);
    return leftmost(best, best_column);
}

// opt[0, size], size = bit_ceil(groups). Level by level: the rows of one level are independent,
// so their searches overlap in the core (a depth-first recursion was 13% slower).
void sample_levels(const Problem& p, std::size_t groups) {
    const std::size_t size = std::bit_ceil(groups);
    std::fill(p.opt + groups, p.opt + size + 1, u32(p.m - 1));
    p.opt[0] = 0;
    for (std::size_t step = size / 2; step; step /= 2) {
        for (std::size_t t = step; t < groups; t += 2 * step) {
            const std::size_t k = t * kGroup, lo = p.opt[t - step], hi = p.opt[t + step];
            p.opt[t] = u32(argmin(p, k, std::max(lo, k + 1 > p.n ? k + 1 - p.n : 0), std::min(hi, k)));
        }
    }
}

// c[k0, k0 + kGroup) for group t: columns between the sample opts, clamped to rows' valid columns.
void group(const Problem& p, std::size_t t) {
    const std::size_t k0 = t * kGroup;
    const std::size_t lo = std::max(std::size_t{p.opt[t]}, k0 + 1 > p.n ? k0 + 1 - p.n : 0);
    const std::size_t hi = std::min(std::size_t{p.opt[t + 1]}, k0 + kGroup - 1);
    __m256i low[kVecs];
    for (auto& v : low) v = _mm256_set1_epi32(-1);
    for (std::size_t j = lo; j <= hi; ++j) {
        const __m256i b = _mm256_set1_epi32(int(p.b[j]));
        const auto* a = reinterpret_cast<const __m256i_u*>(p.a + (k0 - j));  // index >= 1 - kGroup
        for (std::size_t v = 0; v < kVecs; ++v)
            low[v] = _mm256_min_epu32(low[v], _mm256_add_epi32(b, _mm256_loadu_si256(a + v)));
    }
    auto* c = reinterpret_cast<__m256i_u*>(p.c + k0);
    for (std::size_t v = 0; v < kVecs; ++v) _mm256_storeu_si256(c + v, low[v]);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>(), m = in.read<u32>(), count = n + m - 1;
    const std::size_t groups = (count + kGroup - 1) / kGroup;
    const std::size_t a_words = n + 2 * kAPad, b_words = m, c_words = groups * kGroup;
    const std::size_t text_words = columns::kTextBytes / sizeof(u32);
    u32* const memory = mem::huge<u32>(text_words + a_words + b_words + c_words + std::bit_ceil(groups) + 1);
    char* const text = reinterpret_cast<char*>(memory);
    u32* const a = memory + text_words;
    u32* const b = a + a_words;
    u32* const c = b + b_words;
    u32* const opt = c + c_words;
    std::fill_n(a, kAPad, kInf);
    io::read_bulk(in, a + kAPad, n);
    std::fill_n(a + kAPad + n, kAPad, kInf);
    io::read_bulk(in, b, m);

    const Problem p{n, m, a + kAPad, b, opt, c};
    sample_levels(p, groups);
    for (std::size_t t = 0; t < groups; ++t) group(p, t);

    std::fill(c + count, c + c_words, 0);
    io::Writer out;
    columns::write(out, c, count, text);
}

}  // namespace

RUN_EARLY(solve)
