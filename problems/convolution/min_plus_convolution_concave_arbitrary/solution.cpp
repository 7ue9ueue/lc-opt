// c_k = min over i + j = k of a_i + b_j, with a concave (N, M <= 2^19, values <= 10^9, so
// c_k < 2^31). Column j is the curve f_j(x) = a_{x-j} + b_j on [j, j + N). For j < j', f_j' - f_j
// is non-decreasing on their common domain (a's slopes fall): f_j' wins a prefix of it.
//
// Columns go in blocks [j0, j1) of at most N. Rows [j0, j0 + N) see no column end; a forward
// sweep adds column x at row x, and the newest column wins a prefix of the rows left, so the
// envelope is a stack of segments, newest on top. Rows [j1 - 1, j1 + N - 1) see no column start;
// a backward sweep is the mirror image. The two sweeps cover the block's rows.
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "lib/io/io.hpp"

namespace {

struct Segment {
    std::uint32_t column;
    std::uint32_t last;  // last row it owns, in sweep order
};

// The last t in [0, len) with gap(t) <= 0, for a non-decreasing gap with gap(0) = g0 <= 0 and
// gap(len) = g1 > 0. On the judge's data the gap is nearly linear, so interpolation lands within
// a row or two; each probe also tests its neighbour on the far side. A probe that leaves more
// than half of the interval is followed by a bisection step.
template <class Gap>
std::uint32_t last_nonpositive(Gap gap, std::uint32_t len, std::int64_t g0, std::int64_t g1) {
    std::uint32_t lo = 0, hi = len;
    bool bisect = false;
    while (hi - lo > 1) {
        const std::uint32_t width = hi - lo;
        std::uint32_t t = lo + width / 2;
        if (!bisect) {
            const double guess = double(lo) + double(-g0) * double(width) / double(g1 - g0);
            t = std::clamp(std::uint32_t(guess), lo + 1, hi - 1);
        }
        if (const std::int64_t g = gap(t); g <= 0) {
            lo = t, g0 = g;
            if (t + 1 < hi) {
                if (const std::int64_t g = gap(t + 1); g > 0) return t;
                else lo = t + 1, g0 = g;
            }
        } else {
            hi = t, g1 = g;
            if (t - 1 > lo) {
                if (const std::int64_t g = gap(t - 1); g <= 0) return t - 1;
                else hi = t - 1, g1 = g;
            }
        }
        bisect = !bisect && 2 * (hi - lo) > width;
    }
    return lo;
}

// Sweeps rows first, first + Step, ..., last (Step = +1 or -1). Column enter_first + Step k joins
// at the k-th row for k < enter_count; it beats every older column on a prefix of the rows left.
// c[x] = min(c[x], envelope at x).
template <int Step>
void sweep(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* c, std::uint32_t first,
           std::uint32_t last, std::uint32_t enter_first, std::uint32_t enter_count, Segment* stack) {
    auto value = [&](std::uint32_t j, std::uint32_t x) { return a[x - j] + b[j]; };
    auto before = [](std::uint32_t x, std::uint32_t y) { return Step > 0 ? x < y : x > y; };
    std::uint32_t top = 0;
    for (std::uint32_t x = first, k = 0;; x += Step, ++k) {
        while (top && before(stack[top - 1].last, x)) --top;
        if (k < enter_count) {
            const std::uint32_t j = enter_first + Step * k;
            std::uint32_t won = x - Step;  // the new column wins rows up to here
            std::uint32_t from = x;
            while (top) {
                const auto [o, end] = stack[top - 1];
                auto gap = [&](std::uint32_t t) {
                    const std::uint32_t y = from + Step * t;
                    return std::int64_t(value(j, y)) - std::int64_t(value(o, y));
                };
                const std::uint32_t len = Step > 0 ? end - from : from - end;
                const std::int64_t at_end = gap(len);
                if (at_end <= 0) {
                    won = end;
                    from = end + Step;
                    --top;
                    continue;
                }
                if (const std::int64_t at_from = gap(0); at_from <= 0)
                    won = from + Step * last_nonpositive(gap, len, at_from, at_end);
                break;
            }
            if (!top) won = last;
            if (won != x - Step) stack[top++] = {j, won};
        }
        const std::uint32_t o = stack[top - 1].column;
        c[x] = std::min(c[x], value(o, x));
        if (x == last) break;
    }
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    const auto m = in.read<std::uint32_t>();
    std::vector<std::uint32_t> a(n), b(m), c(n + m - 1, ~0u);
    in.read(a.data(), n);
    in.read(b.data(), m);

    std::vector<Segment> stack(std::min(n, m));
    for (std::uint32_t j0 = 0; j0 < m; j0 += n) {
        const std::uint32_t j1 = std::min(m, j0 + n);
        sweep<1>(a.data(), b.data(), c.data(), j0, j0 + n - 1, j0, j1 - j0, stack.data());
        sweep<-1>(a.data(), b.data(), c.data(), j1 + n - 2, j1 - 1, j1 - 1, j1 - j0, stack.data());
    }

    io::Writer out;
    out.write_array(c.data(), c.size(), ' ');
    out.write('\n');
}

#ifdef __ELF__
// Runs before the C++ runtime initializes iostreams and locales; _exit skips their teardown.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
