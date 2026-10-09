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
// Crossings are found lazily: each column keeps a bracket around the last row where it beats the
// one below, narrowed by bisection only when an insertion needs it, and for free as the sweep
// passes.
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "lib/io/io.hpp"

namespace {

// Rows are sweep offsets t in [0, len). The last row where the column beats the entry below is in
// [lo, hi); the bottom entry has [len - 1, len).
struct Entry {
    std::uint32_t column;
    std::uint32_t lo;  // beats the entry below here
    std::uint32_t hi;  // loses to it here
};

// Sweeps len rows: first, first + Step, ... (Step = +1 or -1). Column enter_first + Step t joins
// at offset t for t < enter_count. c[x] = min(c[x], envelope at x).
template <int Step>
void sweep(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t* c, std::uint32_t first,
           std::uint32_t len, std::uint32_t enter_first, std::uint32_t enter_count, Entry* stack) {
    auto value = [&](std::uint32_t j, std::uint32_t t) { return a[first + Step * t - j] + b[j]; };
    auto beats = [&](std::uint32_t j, std::uint32_t o, std::uint32_t t) { return value(j, t) <= value(o, t); };
    std::uint32_t top = 0;
    // Column k joins at row t, where every entry still owns rows from t on.
    auto insert = [&](std::uint32_t k, std::uint32_t t) {
        if (top && !beats(k, stack[top - 1].column, t)) return;  // loses at once: never wins
        while (top) {
            Entry& q = stack[top - 1];
            // Pop q if k beats it at q's last row; else q keeps rows and k loses to q from there.
            // Narrow q's bracket until one of the two is known.
            while (!beats(k, q.column, q.hi - 1)) {
                for (;;) {
                    const std::uint32_t lo = std::max(q.lo, t);
                    if (lo == q.hi - 1) {
                        stack[top++] = {k, t, lo};
                        return;
                    }
                    if (q.lo > t && !beats(k, q.column, q.lo)) {
                        stack[top++] = {k, t, q.lo};
                        return;
                    }
                    const std::uint32_t mid = lo + (q.hi - lo) / 2;
                    if (!beats(q.column, stack[top - 2].column, mid)) {
                        q.hi = mid;
                        break;
                    }
                    q.lo = mid;
                }
            }
            --top;
        }
        stack[top++] = {k, len - 1, len};
    };
    for (std::uint32_t t = 0; t < len; ++t) {
        // The top owns row t while it beats the entry below.
        while (top > 1) {
            Entry& e = stack[top - 1];
            if (t < e.hi && (t <= e.lo || beats(e.column, stack[top - 2].column, t))) {
                e.lo = std::max(e.lo, t);
                break;
            }
            --top;
        }
        if (t < enter_count) insert(enter_first + Step * t, t);
        const std::uint32_t x = first + Step * t;
        c[x] = std::min(c[x], value(stack[top - 1].column, t));
    }
}

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    const auto m = in.read<std::uint32_t>();
    std::vector<std::uint32_t> a(n), b(m), c(n + m - 1, ~0u);
    in.read(a.data(), n);
    in.read(b.data(), m);

    std::vector<Entry> stack(std::min(n, m));
    for (std::uint32_t j0 = 0; j0 < m; j0 += n) {
        const std::uint32_t j1 = std::min(m, j0 + n);
        sweep<1>(a.data(), b.data(), c.data(), j0, n, j0, j1 - j0, stack.data());
        sweep<-1>(a.data(), b.data(), c.data(), j1 + n - 2, n, j1 - 1, j1 - j0, stack.data());
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
