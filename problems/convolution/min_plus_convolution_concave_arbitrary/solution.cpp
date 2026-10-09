// c_k = min over i + j = k of a_i + b_j, with a concave (N, M <= 2^19, values <= 10^9, so
// c_k < 2^31). Each b_j gives the curve f_j(x) = a_{x-j} + b_j on [j, j + N). For j < j', f_j - f_j'
// is non-increasing on their common domain (a's slopes fall), so f_j' is below the envelope of the
// earlier curves on a suffix of its domain. The envelope is a deque of (owner, start) segments:
// a new curve pops the back segments it beats at their start, then binary-searches its start.
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "lib/io/io.hpp"

namespace {

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    const auto m = in.read<std::uint32_t>();
    std::vector<std::uint32_t> a(n), b(m), c(n + m - 1);
    in.read(a.data(), n);
    in.read(b.data(), m);

    std::vector<std::uint32_t> owner(m), start(m);
    std::uint32_t head = 0, tail = 0;  // segments [head, tail)
    for (std::uint32_t x = 0; x < n + m - 1; ++x) {
        if (x < m) {
            const std::uint32_t j = x;
            auto beats = [&](std::uint32_t o, std::uint32_t y) { return a[y - j] + b[j] <= a[y - o] + b[o]; };
            std::uint32_t first = j;  // where the new curve starts to win
            while (tail > head) {
                const std::uint32_t o = owner[tail - 1];
                const std::uint32_t s = std::max(start[tail - 1], j);
                if (s >= o + n || beats(o, s)) {  // o expired, or beaten from s on
                    --tail;
                    continue;
                }
                std::uint32_t lo = s, hi = o + n;  // loses at lo; wins at hi (past o's domain)
                while (hi - lo > 1) {
                    const std::uint32_t mid = lo + (hi - lo) / 2;
                    if (!beats(o, mid)) lo = mid;
                    else hi = mid;
                }
                first = hi;
                break;
            }
            owner[tail] = j;
            start[tail] = first;
            ++tail;
        }
        while (tail - head > 1 && start[head + 1] <= x) ++head;
        const std::uint32_t o = owner[head];
        c[x] = a[x - o] + b[o];
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
