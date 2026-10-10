// c_k = min over i + j = k of a_i + b_j, with a concave (N, M <= 2^19, values <= 10^9, so
// c_k < 2^31). Column j is the curve f_j(x) = a_{x-j} + b_j on [j, j + N). For j < j', f_j' - f_j
// is non-decreasing on their common domain (a's slopes fall): f_j' wins a prefix of it.
//
// Columns go in blocks [j0, j1) of at most N. Rows [j0, j0 + N) see no column end; a forward
// sweep adds column x at row x, and the newest column wins a prefix of the rows left, so the
// envelope is a stack, newest on top, each column owning the rows from where the one above it
// stops to where it starts losing to the one below. Rows [j1 - 1, j1 + N - 1) see no column start;
// a backward sweep is the mirror image. The two sweeps cover the block's rows.
//
// Each entry keeps its exact crossing with the entry below. A new column's crossing with the entry
// it lands on is bracketed first: from below by the crossing of the entry it popped last (close
// for far columns), from both sides by a rank among a's slopes (narrow for near columns); then
// bisection down to 8 rows and one 8-row vector compare.
#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "lib/run/early.hpp"
#include "../min_plus_convolution_convex_arbitrary/columns.hpp"

namespace {

using u32 = std::uint32_t;

// Bounds on R(theta), the number of slopes s_y = a[y + 1] - a[y] (y < n - 1, non-increasing) with
// s_y >= theta. For columns d apart whose b values differ by g (newer minus older), the newer one
// wins where a[y + d] - a[y] >= g, y its offset into a. That sum of d slopes lies in
// [d s_{y+d-1}, d s_y], so with R = R(ceil(g / d)) the newer column wins at y <= R - d and loses
// at y >= R. Theta values go in buckets of 2^shift; the bounds are exact when shift = 0, that is
// when the slopes span fewer than kBuckets values.
class SlopeRank {
public:
    static constexpr u32 kBuckets = 1 << 15;

    struct Bounds {
        u32 low, high;  // low <= R(theta) <= high
    };

    // table: kBuckets entries.
    SlopeRank(const u32* a, u32 n, Bounds* table) : table_(table) {
        const u32 count = n - 1;
        if (count == 0) return;
        max_ = slope(a, 0);
        min_ = slope(a, count - 1);
        const u32 span = u32(max_ - min_) + 1;  // theta in [min_, max_ + 1]
        while (span >> shift_ >= kBuckets) ++shift_;
        u32 rank = count;  // R(theta) for the theta last queried
        const auto at_least = [&](std::int64_t theta) {
            while (rank > 0 && slope(a, rank - 1) < theta) --rank;
            return rank;
        };
        for (u32 i = 0; i <= span >> shift_; ++i) {
            const std::int64_t first = min_ + (std::int64_t(i) << shift_);
            table[i].high = at_least(first);
            table[i].low = at_least(first + (std::int64_t(1) << shift_) - 1);
        }
    }

    Bounds operator()(std::int32_t theta) const {
        return table_[u32(std::clamp(theta, min_, max_ + 1) - min_) >> shift_];
    }

private:
    static std::int32_t slope(const u32* a, u32 y) { return std::int32_t(a[y + 1] - a[y]); }

    const Bounds* table_;
    std::int32_t min_ = 0, max_ = 0;
    u32 shift_ = 0;
};

// A column in a sweep: its value at row offset t is a[offset + Step t] + bias (offset wraps mod
// 2^32). It beats the entry below up to row end and loses after; the bottom entry has end = len - 1.
struct Entry {
    u32 offset;
    u32 bias;  // b of the column
    u32 end;
};

// Sweeps len rows: first, first + Step, ... (Step = +1 or -1). Column enter_first + Step t joins
// at offset t for t < enter_count. c[x] = min(c[x], envelope at x). a must be readable 8 values
// beyond both ends.
template <int Step>
class Sweep {
public:
    Sweep(const u32* a, const u32* b, const SlopeRank& rank, u32 first, u32 len, Entry* stack)
        : a_(a), b_(b), rank_(rank), first_(first), len_(len), stack_(stack) {}

    void run(u32* c, u32 enter_first, u32 enter_count) {
        // A joining column is at a[first - enter_first] (a's first or last value).
        const u32 joining = a_[first_ - enter_first];
        Entry top = {first_ - enter_first, b_[enter_first], len_ - 1};
        c[first_] = std::min(c[first_], joining + top.bias);
        u32 t = 1;
        for (; t < enter_count; ++t) {
            while (top.end < t) top = stack_[--below_];
            u32 at_t = value(top, t);
            const u32 k = enter_first + Step * t;
            if (const u32 v = joining + b_[k]; v <= at_t) {
                top = insert({first_ - k, b_[k], 0}, t, top);
                at_t = v;
            }
            const u32 x = first_ + Step * t;
            c[x] = std::min(c[x], at_t);
        }
        // No more columns join: each entry owns its rows up to its end.
        stack_[below_++] = top;
        for (; t < len_; --below_) {
            const Entry& e = stack_[below_ - 1];
            if (e.end < t) continue;
            const u32 x0 = first_ + Step * t, x1 = first_ + Step * e.end;
            t = e.end + 1;
            const u32 column = first_ - e.offset;  // a's index at row x is x - column
            for (u32 x = std::min(x0, x1); x <= std::max(x0, x1); ++x) c[x] = std::min(c[x], a_[x - column] + e.bias);
        }
    }

private:
    static constexpr std::int32_t kNear = 1024;  // columns at most this far apart get rank brackets

    u32 value(const Entry& e, u32 t) const { return a_[e.offset + Step * t] + e.bias; }
    bool beats(const Entry& e, const Entry& o, u32 t) const { return value(e, t) <= value(o, t); }

    // Rows t, t + 1, ... where k beats q, counted up to the first loss, at most 8.
    u32 window(const Entry& k, const Entry& q, u32 t) const {
        constexpr int kShift = Step == 1 ? 0 : -7;  // Step -1: the rows' values run backwards
        const auto at = [&](const Entry& e) {
            const auto* p = reinterpret_cast<const __m256i*>(a_ + (e.offset + Step * t) + kShift);
            return _mm256_add_epi32(_mm256_loadu_si256(p), _mm256_set1_epi32(int(e.bias)));
        };
        const u32 loses = u32(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(at(k), at(q)))));
        if constexpr (Step == 1) return std::countr_zero(loses | 0x100);
        else return std::countl_zero(loses << 24 | 0x800000);
    }

    // The last row in [lo, hi) where k beats q; k beats q at lo and loses at hi.
    u32 crossing(const Entry& k, const Entry& q, u32 lo, u32 hi) const {
        if (const std::int32_t d = Step * std::int32_t(q.offset - k.offset); d <= kNear) {
            const std::int32_t gap = std::int32_t(k.bias - q.bias);
            std::int32_t theta = gap / d;
            theta += theta * d < gap;  // ceil(gap / d)
            // In the backward sweep k's offset into a falls as t grows: k wins at offsets >= R + d
            // and loses at offsets <= R, for R = R(1 - theta).
            const auto r = rank_(Step == 1 ? theta : 1 - theta);
            const std::int32_t offset = std::int32_t(k.offset);
            const std::int32_t wins = Step == 1 ? std::int32_t(r.low) - d - offset : offset - std::int32_t(r.high) - d;
            const std::int32_t loses = Step == 1 ? std::int32_t(r.high) - offset : offset - std::int32_t(r.low);
            lo = u32(std::max(std::int32_t(lo), wins));
            hi = u32(std::min(std::int32_t(hi), loses));
        }
        u32 m = window(k, q, lo);
        if (m == 8 && hi - lo > 8) {
            lo += 7;
            while (hi - lo > 8) {
                const u32 mid = lo + (hi - lo) / 2;
                (beats(k, q, mid) ? lo : hi) = mid;
            }
            m = window(k, q, lo);
        }
        return lo + m - 1;
    }

    // Column k joins at row t, where it beats q, the top; returns the new top.
    Entry insert(Entry k, u32 t, Entry q) {
        u32 lo = t;  // a row where k beats q
        for (;;) {
            if (!beats(k, q, q.end)) {
                k.end = crossing(k, q, lo, q.end);
                stack_[below_++] = q;
                return k;
            }
            lo = q.end;  // k beats q there, and q the entry below
            if (below_ == 0) {
                k.end = len_ - 1;
                return k;
            }
            q = stack_[--below_];
        }
    }

    const u32* a_;
    const u32* b_;
    const SlopeRank& rank_;
    u32 first_, len_;
    Entry* stack_;  // the entries below the top
    u32 below_ = 0;
};

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    const auto m = in.read<std::uint32_t>();
    std::uint32_t* const a = mem::huge<std::uint32_t>(n + 16) + 8;  // 8 readable values on each side
    std::uint32_t* const b = mem::huge<std::uint32_t>(m);
    const std::size_t count = n + m - 1;
    std::uint32_t* const c = mem::huge<std::uint32_t>((count + 15) / 16 * 16);  // tail stays 0
    char* const text = mem::huge<char>(columns::kTextBytes);
    mem::Arena arena(SlopeRank::kBuckets * sizeof(SlopeRank::Bounds) + std::min(n, m) * sizeof(Entry) + 64);
    auto* const table = arena.take<SlopeRank::Bounds>(SlopeRank::kBuckets);
    Entry* const stack = arena.take<Entry>(std::min(n, m));
    io::read_bulk(in, a, n);
    io::read_bulk(in, b, m);
    std::fill(c, c + count, ~0u);

    const SlopeRank rank(a, n, table);
    for (std::uint32_t j0 = 0; j0 < m; j0 += n) {
        const std::uint32_t j1 = std::min(m, j0 + n);
        Sweep<1>(a, b, rank, j0, n, stack).run(c, j0, j1 - j0);
        Sweep<-1>(a, b, rank, j1 + n - 2, n, stack).run(c, j1 - 1, j1 - j0);
    }

    io::Writer out;
    columns::write(out, c, count, text);
}

}  // namespace

RUN_EARLY(solve)
