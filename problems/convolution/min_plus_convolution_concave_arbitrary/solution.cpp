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

// Sweeps rows first, first + step, ..., up to and including row last (step = +1 or -1). Column
// enter(x) joins at row x when it is in range; it beats every older column on a prefix of the
// rows left. c[x] = min(c[x], envelope at x).
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
            const std::uint32_t at_x = value(j, x);
            std::uint32_t won = x - Step;  // the new column wins rows up to here
            std::uint32_t from = x;
            while (top) {
                const Segment s = stack[top - 1];
                if (value(j, s.last) <= value(s.column, s.last)) {
                    won = s.last;
                    from = s.last + Step;
                    --top;
                    continue;
                }
                // Wins at from? Then the last win in [from, s.last).
                if (from == x ? at_x <= value(s.column, x) : value(j, from) <= value(s.column, from)) {
                    const std::uint32_t o = s.column;
                    auto wins = [&](std::uint32_t t) { return value(j, from + Step * t) <= value(o, from + Step * t); };
                    // Wins at offset base, loses at base + len. Short wins are common: try 1 first.
                    std::uint32_t base = 0, len = Step > 0 ? s.last - from : from - s.last;
                    if (len > 1 && wins(1)) {
                        base = 1;
                        --len;
                        while (len > 1) {
                            const std::uint32_t half = len / 2, next = (len - half) / 2;
                            for (const std::uint32_t t : {base + next, base + half + next}) {
                                __builtin_prefetch(a + (from + Step * t - j));
                                __builtin_prefetch(a + (from + Step * t - o));
                            }
                            base = wins(base + half) ? base + half : base;
                            len -= half;
                        }
                    }
                    won = from + Step * base;
                }
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
