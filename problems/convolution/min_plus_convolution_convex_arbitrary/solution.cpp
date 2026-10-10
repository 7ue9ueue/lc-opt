// c[k] = min over i + j = k of a[i] + b[j], a convex. f_k(j) = b[j] + a[k - j] on the valid columns
// max(0, k - N + 1) <= j <= min(k, M - 1) is Monge: for rows k < k' and columns i < j valid in both,
// f_k'(j) - f_k'(i) <= f_k(j) - f_k(i). So the leftmost argmin opt(k) is nondecreasing in k, and
// between rows k < k' a column j can be opt only if it is invalid in k' or below every column
// i < j valid in k' (a strict prefix record of row k'), and invalid in k or at most every column
// i > j valid in k (a suffix minimum of row k).
//
// opt is found at every kGroup-th row (sample rows), coarse rows first. Each node of the bisection
// holds the candidates for the rows strictly between its two ends: a dense range of columns, or a
// sorted list when the middle row's records are sparse (scans give up on dense records). A node
// whose ends share their opt is settled. Each group of kGroup rows then takes its minima over its
// node's candidates, kGroup rows per column, one output block at a time.
// Wide ranges skip blocks of kBlock columns by a lower bound: the least b in the block plus the
// least a over the two aligned kBlock-windows of a that the block reads in the row.
#include "lib/io/io.hpp"
#include "lib/io/sequential.hpp"
#include "lib/mem/huge.hpp"
#include "lib/run/early.hpp"
#include "columns.hpp"
#include "runs.hpp"

namespace {

using u32 = std::uint32_t;

constexpr std::size_t kGroup = 16;         // rows per group
constexpr std::size_t kVecs = kGroup / 8;  // vectors per group
constexpr u32 kInf = 3'000'000'000u;       // a outside [0, N), b past M; kInf + b <= 4e9 < 2^32
constexpr u32 kNone = INT32_MAX;           // above every value of a valid column (<= 2e9)
constexpr std::size_t kAPad = kGroup + 8;  // kInf words before a
constexpr std::size_t kBlock = 64;         // columns per bounded block; a multiple of 8
constexpr std::size_t kTail = kBlock;      // kInf words after a and after b
constexpr std::size_t kWide = 1024;        // ranges this wide (>= 9 blocks) use the bounds
constexpr std::size_t kNarrow = 128;      // dense ranges this narrow get dense children
constexpr std::size_t kListWords = std::size_t(1) << 16;  // capacity of a level's candidate lists

struct Problem {
    std::size_t n, m;    // lengths of a and b
    const u32* a;        // a[-kAPad, n + kTail)
    const u32* b;        // b[0, m + kTail)
    const u32* b_least;  // b_least[J] = min b[kBlock J, kBlock (J + 1)) for kBlock J < m
    const u32* a_least;  // a_least[w] = min a[kBlock (w - 1), kBlock (w + 1)), w <= ceil(n / kBlock)
};

u32 value(const Problem& p, std::size_t k, std::size_t j) { return p.b[j] + p.a[k - j]; }
std::size_t first_valid(const Problem& p, std::size_t k) { return k + 1 > p.n ? k + 1 - p.n : 0; }
std::size_t last_valid(const Problem& p, std::size_t k) { return std::min(k, p.m - 1); }

u32 horizontal_min(__m256i v) {
    __m128i x = _mm_min_epu32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    x = _mm_min_epu32(x, _mm_shuffle_epi32(x, 0x4E));
    x = _mm_min_epu32(x, _mm_shuffle_epi32(x, 0xB1));
    return u32(_mm_cvtsi128_si32(x));
}

// y[j, j + 8) + x[k - j - 8 + 1, k - j + 1) reversed: the min-plus row k of x and y at columns j..j+7.
__m256i row_chunk(const u32* x, const u32* y, std::size_t k, std::size_t j) {
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i lhs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + (k - j) - 7));
    const __m256i rhs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + j));
    return _mm256_add_epi32(rhs, _mm256_permutevar8x32_epi32(lhs, reverse));
}

// Lower bounds of blocks J..J+7 for the rows k with k / kBlock = q; bound <= 4e9.
__m256i block_bounds(const Problem& p, std::size_t q, std::size_t J) { return row_chunk(p.a_least, p.b_least, q, J); }

u32 block_bound(const Problem& p, std::size_t q, std::size_t J) { return p.b_least[J] + p.a_least[q - J]; }

// Lanes (bits 0-7) of v below limit, unsigned.
unsigned lanes_below(__m256i v, u32 limit) {
    const __m256i below = _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_max_epu32(v, _mm256_set1_epi32(int(limit))), v),
                                           _mm256_set1_epi32(-1));
    return unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(below)));
}

// Leftmost column among the lanes holding the least value.
u32 leftmost(__m256i value, __m256i column) {
    const __m256i least = _mm256_set1_epi32(int(horizontal_min(value)));
    const __m256i other = _mm256_xor_si256(_mm256_cmpeq_epi32(value, least), _mm256_set1_epi32(-1));
    return horizontal_min(_mm256_or_si256(column, other));
}

// Leftmost j in [lo, hi] minimizing f_k(j); every column in range is valid.
std::size_t argmin(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi) {
    if (hi - lo < 8) {
        std::size_t best = lo;
        u32 best_value = value(p, k, lo);
        for (std::size_t j = lo + 1; j <= hi; ++j) {
            const u32 v = value(p, k, j);
            if (v < best_value) best = j, best_value = v;
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
        const __m256i low = _mm256_min_epu32(best, row_chunk(p.a, p.b, k, j));
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

// Blocks [first, last) (last - first >= 8) in increasing order, or decreasing with Backward, whose
// bound is below limit(): visit(J) for each, re-reading limit() after each visit.
template <bool Backward, class Limit, class Visit>
void bounded_blocks(const Problem& p, std::size_t q, std::size_t first, std::size_t last, Limit limit, Visit visit) {
    const std::size_t vectors = (last - first + 7) / 8;
    for (std::size_t i = 0; i < vectors; ++i) {
        // The vector farthest along the scan overlaps the one before; its repeated lanes are masked.
        const std::size_t left = last - first - 8 * i;  // blocks not yet scanned
        const std::size_t start = Backward ? (left > 8 ? last - 8 * i - 8 : first) : std::min(first + 8 * i, last - 8);
        const std::size_t fresh = Backward ? last - 8 * i - start : start + 8 - (first + 8 * i);  // new lanes
        const unsigned mask = Backward ? (1u << fresh) - 1 : 0xFFu << (8 - fresh) & 0xFF;
        for (unsigned lanes = lanes_below(block_bounds(p, q, start), limit()) & mask; lanes;) {
            const unsigned lane = Backward ? 31 - unsigned(std::countl_zero(lanes)) : unsigned(std::countr_zero(lanes));
            lanes &= ~(1u << lane);
            const std::size_t J = start + lane;
            if (block_bound(p, q, J) < limit()) visit(J);
        }
    }
}

// A column and its value; better: smaller value, then smaller column.
struct Candidate {
    u32 value, column;
};

void keep_better(Candidate& best, Candidate x) {
    if (x.value < best.value || (x.value == best.value && x.column < best.column)) best = x;
}

Candidate candidate(const Problem& p, std::size_t k, std::size_t j) { return {value(p, k, j), u32(j)}; }

// Leftmost argmin of row k over best and the valid columns [lo, hi]. Wide ranges search only blocks
// whose bound is at most the best value found so far, runs of up to kRun adjacent blocks at once.
std::size_t search(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi, Candidate best = {kNone, 0}) {
    constexpr std::size_t kRun = 64;
    if (hi - lo < kWide) {
        keep_better(best, candidate(p, k, argmin(p, k, lo, hi)));
        return best.column;
    }
    const std::size_t first = (lo + kBlock - 1) / kBlock, last = (hi + 1) / kBlock;  // whole blocks
    keep_better(best, candidate(p, k, hi));
    keep_better(best, candidate(p, k, lo));
    if (lo < first * kBlock) keep_better(best, candidate(p, k, argmin(p, k, lo, first * kBlock - 1)));
    if (last * kBlock <= hi) keep_better(best, candidate(p, k, argmin(p, k, last * kBlock, hi)));
    std::size_t run_first = 0, run_end = 0;  // pending blocks [run_first, run_end)
    const auto flush = [&] {
        if (run_end > run_first) keep_better(best, candidate(p, k, argmin(p, k, run_first * kBlock, run_end * kBlock - 1)));
        run_first = run_end;
    };
    bounded_blocks<false>(p, k / kBlock, first, last, [&] { return best.value + 1; }, [&](std::size_t J) {
        if (J != run_end || run_end - run_first == kRun) flush(), run_first = J;
        run_end = J + 1;
    });
    flush();
    return best.column;
}

// Records of row k in scan order: forward, the columns below every value before them; backward,
// the columns at most every value after them. Stops when there are more than 32 + 1/16 of the
// columns scanned: the candidates are then dense.
struct Records {
    u32* out;            // 8 words of slack
    std::size_t origin;  // first column scanned
    u32 limit = kNone;   // a column is a record if its value is below limit
    u32 last = 0;        // the last record
    std::size_t count = 0;

    template <bool Backward>
    bool add(std::size_t j, u32 v) {
        limit = v + Backward, last = u32(j);
        out[count++] = u32(j);
        return count <= 32 + (Backward ? origin - j : j - origin) / 16;
    }
};

// Lane numbers of the set bits of each 8-bit mask, one per byte: ascending, then descending.
constexpr auto kLanes = [] {
    std::array<std::array<std::uint64_t, 256>, 2> t{};
    for (unsigned mask = 0; mask < 256; ++mask)
        for (unsigned i = 0, up = 0, down = 0; i < 8; ++i) {
            if (mask >> i & 1) t[0][mask] |= std::uint64_t(i) << 8 * up++;
            if (mask >> (7 - i) & 1) t[1][mask] |= std::uint64_t(7 - i) << 8 * down++;
        }
    return t;
}();

// Record lanes (bits 0-7) of chunk v given limit (every lane): forward, v[i] < min(limit, v[0, i));
// backward, v[i] < min(limit, v(i, 8) + 1). least: the least value of the chunk, every lane.
// Values < 2^31: signed compares are exact.
template <bool Backward>
unsigned record_lanes(__m256i v, __m256i limit, __m256i& least) {
    const auto spread = [](__m256i x, __m256i from) { return _mm256_min_epu32(x, _mm256_permutevar8x32_epi32(x, from)); };
    if constexpr (!Backward) {
        const __m256i down = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
        const __m256i prefix = spread(spread(spread(v, down), _mm256_setr_epi32(0, 1, 0, 1, 2, 3, 4, 5)),
                                      _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3));
        const __m256i before = _mm256_min_epu32(limit, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(prefix, down), limit, 0x01));
        least = _mm256_permutevar8x32_epi32(prefix, _mm256_set1_epi32(7));
        return unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(before, v))));
    } else {
        const __m256i up = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);
        const __m256i suffix = spread(spread(spread(v, up), _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 6, 7)),
                                      _mm256_setr_epi32(4, 5, 6, 7, 4, 5, 6, 7));
        const __m256i after_plus = _mm256_permutevar8x32_epi32(_mm256_add_epi32(suffix, _mm256_set1_epi32(1)), up);
        const __m256i after = _mm256_min_epu32(limit, _mm256_blend_epi32(after_plus, limit, 0x80));
        least = _mm256_permutevar8x32_epi32(suffix, _mm256_setzero_si256());
        return unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(after, v))));
    }
}

// Records of row k among the lanes of chunk j..j+7 in mask (bits 0-7); false if they overflow.
template <bool Backward>
bool chunk_records(const Problem& p, std::size_t k, std::size_t j, unsigned mask, Records& r) {
    const __m256i v = row_chunk(p.a, p.b, k, j), limit = _mm256_set1_epi32(int(r.limit));
    if (!(unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(limit, v)))) & mask)) [[likely]] return true;
    __m256i least;
    const unsigned lanes = record_lanes<Backward>(v, limit, least) & mask;  // not 0: the first lane below limit
    const __m256i numbers = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(std::int64_t(kLanes[Backward][lanes])));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r.out + r.count), _mm256_add_epi32(numbers, _mm256_set1_epi32(int(j))));
    r.count += unsigned(std::popcount(lanes));
    r.limit = u32(_mm256_cvtsi256_si32(least)) + Backward;
    r.last = u32(j + (Backward ? unsigned(std::countr_zero(lanes)) : 31 - unsigned(std::countl_zero(lanes))));
    return r.count <= 32 + (Backward ? r.origin - j : j - r.origin) / 16;
}

// Records of row k over the valid columns [lo, hi], in chunks; false if they overflow.
template <bool Backward>
bool range_records(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi, Records& r) {
    if (hi - lo < 8) {
        for (std::size_t i = 0; i <= hi - lo; ++i) {
            const std::size_t j = Backward ? hi - i : lo + i;
            const u32 v = value(p, k, j);
            if (v < r.limit && !r.add<Backward>(j, v)) return false;
        }
        return true;
    }
    // The chunk farthest along the scan overlaps the one before; its repeated lanes are masked.
    const std::size_t chunks = (hi - lo + 8) / 8;
    for (std::size_t i = 0; i < chunks; ++i) {
        const std::size_t j = Backward ? std::max(lo, hi - 7 - std::min(hi - 7, 8 * i)) : std::min(lo + 8 * i, hi - 7);
        const std::size_t fresh = Backward ? hi - 8 * i - j + 1 : j + 8 - (lo + 8 * i);
        const unsigned mask = Backward ? (1u << std::min<std::size_t>(fresh, 8)) - 1 : 0xFFu << (8 - fresh) & 0xFF;
        if (!chunk_records<Backward>(p, k, j, mask, r)) return false;
    }
    return true;
}

// Records of row k over the valid columns [lo, hi]; wide ranges skip blocks whose bound is not
// below the limit. False if they overflow.
template <bool Backward>
bool records(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi, Records& r) {
    if (hi - lo < kWide) return range_records<Backward>(p, k, lo, hi, r);
    const std::size_t first = (lo + kBlock - 1) / kBlock, last = (hi + 1) / kBlock;  // whole blocks
    const auto head = [&] { return lo == first * kBlock || range_records<Backward>(p, k, lo, first * kBlock - 1, r); };
    const auto tail = [&] { return hi < last * kBlock || range_records<Backward>(p, k, last * kBlock, hi, r); };
    if (!(Backward ? tail() : head())) return false;
    bool fits = true;
    bounded_blocks<Backward>(p, k / kBlock, first, last, [&] { return fits ? r.limit : 0; }, [&](std::size_t J) {
        fits = range_records<Backward>(p, k, J * kBlock, J * kBlock + kBlock - 1, r);
    });
    return fits && (Backward ? head() : tail());
}

// Whether the first 8 columns scanned, from lo or back from hi (all valid), are all records:
// then most columns are (monotone rows), and a scan for records would overflow.
template <bool Backward>
bool starts_dense(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi) {
    if (hi - lo < 8) return false;
    const __m256i v = row_chunk(p.a, p.b, k, Backward ? hi - 7 : lo);
    const __m256i next = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7));
    const auto falls = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, next)))) & 0x7F;
    return falls == (Backward ? 0 : 0x7F);
}

// Candidate columns of a node: dense, the range [first, first + count); else count sorted columns
// at first in the level's list buffer.
struct Candidates {
    u32 first;
    u32 count : 31;
    u32 dense : 1;
};

Candidates dense(std::size_t lo, std::size_t hi) { return {u32(lo), u32(hi - lo + 1), 1}; }

struct ListBuffer {
    u32* words;  // kListWords
    std::size_t used = 0;

    bool fits(std::size_t count) const { return used + count <= kListWords; }
    void range(std::size_t lo, std::size_t hi) {
        for (std::size_t j = lo; j < hi; ++j) words[used++] = u32(j);
    }
    void list(const u32* from, std::size_t count) {
        std::copy_n(from, count, words + used);
        used += count;
    }
    // The candidates appended since used was first.
    Candidates since(std::size_t first) const { return {u32(first), u32(used - first), 0}; }
};

struct Children {
    Candidates left, right;
};

// Row k over the dense candidates [lo, hi]: its opt, and the children's candidates in next.
// scratch: kListWords words.
Children split_dense(const Problem& p, std::size_t k, std::size_t lo, std::size_t hi, u32& opt, ListBuffer& next,
                     u32* scratch) {
    const std::size_t valid_lo = std::max(lo, first_valid(p, k)), valid_hi = std::min(hi, last_valid(p, k));
    if (valid_hi - valid_lo < kNarrow) {
        opt = u32(argmin(p, k, valid_lo, valid_hi));
        return {dense(lo, opt), dense(opt, hi)};
    }
    // A sparse child holds at most 1/8 of its range; columns invalid in row k are kept.
    const auto sparse = [](std::size_t kept, std::size_t width) { return kept * 8 <= width; };
    // Left: the columns invalid in row k, then the strict prefix records; the last record is opt.
    Records forward{scratch, valid_lo};
    bool listed = false;
    if (!sparse(valid_lo - lo, valid_hi - lo + 1) || starts_dense<false>(p, k, valid_lo, valid_hi)) {
        opt = u32(search(p, k, valid_lo, valid_hi));
    } else if (!records<false>(p, k, valid_lo, valid_hi, forward)) {  // dense: the argmin of the rest
        const Candidate best{forward.limit, forward.last};
        opt = forward.last < valid_hi ? u32(search(p, k, forward.last + 1, valid_hi, best)) : forward.last;
    } else {
        opt = forward.last;
        const std::size_t kept = valid_lo - lo + forward.count;
        listed = sparse(kept, opt - lo + 1) && next.fits(kept);
    }
    Children children{dense(lo, opt), dense(opt, hi)};
    if (listed) {
        const std::size_t first = next.used;
        next.range(lo, valid_lo);
        next.list(scratch, forward.count);
        children.left = next.since(first);
    }
    // Right: the suffix minima from opt on, then the columns invalid in row k.
    Records backward{scratch, valid_hi};
    const std::size_t width = hi - opt + 1;
    if (sparse(hi - valid_hi, width) && !starts_dense<true>(p, k, opt, valid_hi) &&
        records<true>(p, k, opt, valid_hi, backward) && sparse(backward.count + hi - valid_hi, width) &&
        next.fits(backward.count + hi - valid_hi)) {
        const std::size_t first = next.used;
        std::reverse(scratch, scratch + backward.count);
        next.list(scratch, backward.count);
        next.range(valid_hi + 1, hi + 1);
        children.right = next.since(first);
    }
    return children;
}

// Row k over the sorted candidate list[0, count): its opt, and the children's lists in next.
// values: kListWords words.
Children split_list(const Problem& p, std::size_t k, const u32* list, std::size_t count, u32& opt, ListBuffer& next,
                    u32* values) {
    const std::size_t lo = first_valid(p, k), hi = last_valid(p, k);
    std::size_t begin = 0, end = count;  // the valid columns list[begin, end)
    while (list[begin] < lo) ++begin;
    while (list[end - 1] > hi) --end;
    if (!next.fits(count + 1)) {  // children hold at most count + 1 columns
        std::size_t best = begin;
        for (std::size_t i = begin; i < end; ++i)
            if ((values[i] = value(p, k, list[i])) < values[best]) best = i;
        opt = list[best];
        return {dense(list[0], opt), dense(opt, list[count - 1])};
    }
    // Left: the invalid columns before, then the strict prefix records; the last record is opt.
    const std::size_t left_first = next.used;
    next.list(list, begin);
    u32 least = kNone;
    std::size_t best = begin;
    for (std::size_t i = begin; i < end; ++i) {
        values[i] = value(p, k, list[i]);
        if (values[i] < least) least = values[i], best = i, next.words[next.used++] = list[i];
    }
    opt = list[best];
    const Candidates left = next.since(left_first);
    // Right: the suffix minima from opt on, then the invalid columns after.
    const std::size_t right_first = next.used;
    least = kNone;
    for (std::size_t i = end; i-- > best;)
        if (values[i] <= least) least = values[i], next.words[next.used++] = list[i];
    std::reverse(next.words + right_first, next.words + next.used);
    next.list(list + end, count - end);
    return {left, next.since(right_first)};
}

// The bisection over sample rows t * kGroup, t < groups; rows t >= groups are virtual with opt
// m - 1. node[t]: candidates of the active node whose left end is t at the current level. On return,
// lists holds the last level's lists (node[t] of every unsettled group t).
struct Sampler {
    const Problem& p;
    std::size_t groups;
    u32* opt;            // [0, bit_ceil(groups)]
    Candidates* node;    // [0, bit_ceil(groups)]
    ListBuffer lists[2];
    u32* scratch;        // kListWords
    const u32* last_lists = nullptr;

    void run() {
        const std::size_t size = std::bit_ceil(groups);
        std::fill(opt + groups, opt + size + 1, u32(p.m - 1));
        opt[0] = 0;
        node[0] = dense(0, p.m - 1);
        settle(0, size);
        int current = 0;
        for (std::size_t step = size / 2; step; step /= 2) {
            ListBuffer& next = lists[current ^ 1];
            next.used = 0;
            for (std::size_t left = 0; left < size; left += 2 * step) {
                if (opt[left] == opt[left + 2 * step]) continue;  // settled
                const std::size_t t = left + step;
                const Candidates c = node[left];
                const u32* const list = c.dense ? nullptr : lists[current].words + c.first;
                if (t >= groups) {  // virtual middle: the left child keeps the candidates
                    if (c.dense) {
                        node[left] = c;
                    } else if (next.fits(c.count)) {
                        const std::size_t first = next.used;
                        next.list(list, c.count);
                        node[left] = next.since(first);
                    } else {
                        node[left] = dense(list[0], list[c.count - 1]);
                    }
                    settle(left, t);
                    continue;
                }
                const std::size_t k = t * kGroup;
                const Children children = c.dense ? split_dense(p, k, c.first, c.first + c.count - 1, opt[t], next, scratch)
                                                  : split_list(p, k, list, c.count, opt[t], next, scratch);
                node[left] = children.left;
                node[t] = children.right;
                settle(left, t);
                settle(t, left + 2 * step);
            }
            current ^= 1;
        }
        last_lists = lists[current].words;
    }

    // A node whose ends share their opt: every row between has it.
    void settle(std::size_t left, std::size_t right) {
        if (opt[left] == opt[right]) std::fill(opt + left + 1, opt + right, opt[left]);
    }
};

// low[v] lane i = min(low[v] lane i, f_{k0 + 8v + i}(j)), kInf-padded a for invalid columns.
void take_column(const Problem& p, std::size_t k0, std::size_t j, __m256i (&low)[kVecs]) {
    const __m256i b = _mm256_set1_epi32(int(p.b[j]));
    const auto* a = reinterpret_cast<const __m256i_u*>(p.a + (k0 - j));  // index >= 1 - kGroup
    for (std::size_t v = 0; v < kVecs; ++v) low[v] = _mm256_min_epu32(low[v], _mm256_add_epi32(b, _mm256_loadu_si256(a + v)));
}

// c[k0, k0 + kGroup) for group t into c: one column if settled, else its node's candidates, the
// columns of a dense range clamped to the rows' valid columns.
void group(const Problem& p, const Sampler& s, std::size_t t, u32* c) {
    const std::size_t k0 = t * kGroup;
    __m256i low[kVecs];
    for (auto& v : low) v = _mm256_set1_epi32(-1);
    if (s.opt[t] == s.opt[t + 1]) {
        take_column(p, k0, s.opt[t], low);
    } else if (const Candidates nc = s.node[t]; nc.dense) {
        const std::size_t lo = std::max(std::size_t{nc.first}, first_valid(p, k0));
        const std::size_t hi = std::min(std::size_t{nc.first} + nc.count - 1, k0 + kGroup - 1);
        for (std::size_t j = lo; j <= hi; ++j) take_column(p, k0, j, low);
    } else {
        for (std::size_t i = 0; i < nc.count; ++i) take_column(p, k0, s.last_lists[nc.first + i], low);
    }
    auto* out = reinterpret_cast<__m256i_u*>(c);
    for (std::size_t v = 0; v < kVecs; ++v) _mm256_storeu_si256(out + v, low[v]);
}

// least[J] = min x[kBlock J, kBlock (J + 1)) for J < blocks; x readable over the whole blocks.
void block_minima(const u32* x, std::size_t blocks, u32* least) {
    for (std::size_t J = 0; J < blocks; ++J) {
        const auto* v = reinterpret_cast<const __m256i_u*>(x + J * kBlock);
        __m256i low = _mm256_loadu_si256(v);
        for (std::size_t i = 1; i < kBlock / 8; ++i) low = _mm256_min_epu32(low, _mm256_loadu_si256(v + i));
        least[J] = horizontal_min(low);
    }
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>(), m = in.read<u32>(), count = n + m - 1;
    io::advise_sequential(in);
    const std::size_t groups = (count + kGroup - 1) / kGroup, nodes = std::bit_ceil(groups) + 1;
    const std::size_t a_blocks = (n + kBlock - 1) / kBlock, b_blocks = (m + kBlock - 1) / kBlock;
    mem::Arena arena(columns::kTextBytes + 4 * (kAPad + n + kTail + m + kTail + nodes + b_blocks + a_blocks + 1 +
                                                3 * kListWords + columns::kBlock + kGroup) +
                     sizeof(Candidates) * nodes + 64 * 12);
    char* const text = arena.take<char>(columns::kTextBytes);
    u32* const a = arena.take<u32>(kAPad + n + kTail);
    u32* const b = arena.take<u32>(m + kTail);
    std::fill_n(a, kAPad, kInf);
    runs::read(in, a + kAPad, n);
    std::fill_n(a + kAPad + n, kTail, kInf);
    runs::read(in, b, m);
    std::fill_n(b + m, kTail, kInf);

    // a_least[w] = min(window w - 1, window w), windows outside [0, n) all kInf.
    u32* const b_least = arena.take<u32>(b_blocks);
    u32* const a_least = arena.take<u32>(a_blocks + 1);
    block_minima(b, b_blocks, b_least);
    block_minima(a + kAPad, a_blocks, a_least + 1);
    a_least[0] = a_least[1];
    for (std::size_t w = 1; w < a_blocks; ++w) a_least[w] = std::min(a_least[w], a_least[w + 1]);

    // The list buffers last: their tails are rarely touched, so they seldom fault in another huge page.
    const Problem p{n, m, a + kAPad, b, b_least, a_least};
    u32* const opt = arena.take<u32>(nodes);
    Candidates* const node = arena.take<Candidates>(nodes);
    u32* const c = arena.take<u32>(columns::kBlock + kGroup);
    Sampler sampler{p, groups, opt, node, {{arena.take<u32>(kListWords)}, {arena.take<u32>(kListWords)}},
                    arena.take<u32>(kListWords)};
    sampler.run();

    // c by output blocks, each formatted and written while it is in cache.
    io::Writer out;
    for (std::size_t k0 = 0; k0 < count; k0 += columns::kBlock) {
        const std::size_t size = std::min(columns::kBlock, count - k0), end = (size + kGroup - 1) / kGroup * kGroup;
        for (std::size_t i = 0; i < end; i += kGroup) group(p, sampler, (k0 + i) / kGroup, c + i);
        std::fill(c + size, c + end, 0);
        columns::write(out, c, size, text);
    }
}

}  // namespace

RUN_EARLY(solve)
