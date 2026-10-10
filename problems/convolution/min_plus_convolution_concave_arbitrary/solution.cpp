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
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/io/io.hpp"
#include "../min_plus_convolution_convex_arbitrary/columns.hpp"

namespace {

// A column in a sweep: its value at row offset t is a[offset + Step t] + bias (offset wraps mod
// 2^32). The last row where it beats the entry below is in [lo, hi); the bottom entry has
// [len - 1, len).
struct Entry {
    std::uint32_t offset;
    std::uint32_t bias;  // b of the column
    std::uint32_t lo;    // beats the entry below here
    std::uint32_t hi;    // loses to it here
};

// Sweeps len rows: first, first + Step, ... (Step = +1 or -1). Column enter_first + Step t joins
// at offset t for t < enter_count. c[x] = min(c[x], envelope at x).
template <int Step>
class Sweep {
public:
    Sweep(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t first, std::uint32_t len, Entry* stack)
        : a_(a), b_(b), first_(first), len_(len), stack_(stack) {}

    void run(std::uint32_t* c, std::uint32_t enter_first, std::uint32_t enter_count) {
        // A joining column is at a[first - enter_first] (a's first or last value).
        const std::uint32_t joining = a_[first_ - enter_first];
        for (std::uint32_t t = 0; t < len_; ++t) {
            std::uint32_t at_t = expire(t);
            if (t < enter_count) {
                const std::uint32_t k = enter_first + Step * t;
                if (const std::uint32_t v = joining + b_[k]; v <= at_t) {
                    insert({first_ - k, b_[k], 0, 0}, t);
                    at_t = v;
                }
            }
            const std::uint32_t x = first_ + Step * t;
            c[x] = std::min(c[x], at_t);
        }
    }

private:
    std::uint32_t value(const Entry& e, std::uint32_t t) const { return a_[e.offset + Step * t] + e.bias; }
    bool beats(const Entry& e, const Entry& o, std::uint32_t t) const { return value(e, t) <= value(o, t); }

    // Pops the entries that stopped owning rows before t; returns the top's value at t (any value
    // if the stack is empty).
    std::uint32_t expire(std::uint32_t t) {
        while (top_) {
            Entry& e = stack_[top_ - 1];
            const std::uint32_t v = value(e, t);
            if (top_ == 1 || t <= e.lo) return v;
            if (t < e.hi && v <= value(stack_[top_ - 2], t)) {
                e.lo = t;
                return v;
            }
            --top_;
        }
        return ~0u;
    }

    // Column k joins at row t, where it beats the top; every entry owns rows from t on.
    void insert(Entry k, std::uint32_t t) {
        k.lo = t;
        while (top_) {
            Entry& q = stack_[top_ - 1];
            // Pop q if k beats it at q's last row; else q keeps rows and k loses to q from there.
            // Narrow q's bracket until one of the two is known.
            while (!beats(k, q, q.hi - 1)) {
                for (;;) {
                    const std::uint32_t lo = std::max(q.lo, t);
                    if (lo == q.hi - 1) {
                        k.hi = lo;
                        stack_[top_++] = k;
                        return;
                    }
                    if (q.lo > t && !beats(k, q, q.lo)) {
                        k.hi = q.lo;
                        stack_[top_++] = k;
                        return;
                    }
                    const std::uint32_t mid = lo + (q.hi - lo) / 2;
                    if (!beats(q, stack_[top_ - 2], mid)) {
                        q.hi = mid;
                        break;
                    }
                    q.lo = mid;
                }
            }
            --top_;
        }
        k.lo = len_ - 1;
        k.hi = len_;
        stack_[top_++] = k;
    }

    const std::uint32_t* a_;
    const std::uint32_t* b_;
    std::uint32_t first_, len_;
    Entry* stack_;
    std::uint32_t top_ = 0;
};

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

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>();
    const auto m = in.read<std::uint32_t>();
    std::uint32_t* const a = allocate<std::uint32_t>(n);
    std::uint32_t* const b = allocate<std::uint32_t>(m);
    const std::size_t count = n + m - 1;
    std::uint32_t* const c = allocate<std::uint32_t>((count + 15) / 16 * 16);  // tail stays 0
    char* const text = allocate<char>(columns::kTextBytes);
    Entry* const stack = allocate<Entry>(std::min(n, m));
    in.read(a, n);
    in.read(b, m);
    std::fill(c, c + count, ~0u);

    for (std::uint32_t j0 = 0; j0 < m; j0 += n) {
        const std::uint32_t j1 = std::min(m, j0 + n);
        Sweep<1>(a, b, j0, n, stack).run(c, j0, j1 - j0);
        Sweep<-1>(a, b, j1 + n - 2, n, stack).run(c, j1 - 1, j1 - j0);
    }

    io::Writer out;
    columns::write(out, c, count, text);
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
