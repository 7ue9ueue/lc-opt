// c[k] = min over i + j = k of a[i] + b[j], a and b convex: c[0] = a[0] + b[0], and c's slopes are
// the slopes of a and b merged in sorted order. The merge runs as kChains independent chains over
// consecutive ranges of k, interleaved so their load latencies overlap; each chain starts from an
// argmin found by binary search. From any argmin (i, k - i) of c[k], the cheaper of a[i + 1] and
// b[k - i + 1] gives c[k + 1]: the slopes taken so far are k smallest ones.
#include <sys/mman.h>
#include <unistd.h>

#include <climits>

#include "lib/io/io.hpp"
#include "../min_plus_convolution_convex_arbitrary/columns.hpp"

namespace {

using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t kChains = 8;
constexpr i32 kEnd = INT32_MAX;   // slope past the last element; above every real slope
constexpr std::size_t kPad = 64;  // kEnd slopes after each array; chains overrun by < kChains

struct Chain {
    std::size_t i;  // c[k] = a[i] + b[k - i]
    u32 value;      // c[k]
};

// Leftmost i minimizing a[i] + b[k - i]. Its forward difference (a[i + 1] - a[i]) - (b[k - i] -
// b[k - i - 1]) is nondecreasing in i; values are <= 1e9, so slopes fit i32.
Chain start(const u32* a, std::size_t n, const u32* b, std::size_t m, std::size_t k) {
    std::size_t lo = k + 1 > m ? k + 1 - m : 0, hi = std::min(k, n - 1);
    while (lo < hi) {
        const std::size_t i = (lo + hi) / 2, j = k - i;
        if (i32(a[i + 1] - a[i]) >= i32(b[j] - b[j - 1])) hi = i;
        else lo = i + 1;
    }
    return {lo, a[lo] + b[k - lo]};
}

// x[i] = x[i + 1] - x[i] for i < size - 1, then kPad slopes kEnd from x[size - 1] on.
void to_slopes(u32* x, std::size_t size) {
    for (std::size_t i = 0; i + 1 < size; ++i) x[i] = x[i + 1] - x[i];
    std::fill_n(x + size - 1, kPad, u32(kEnd));
}

// c[0, kChains * length): chain s covers c[s * length, (s + 1) * length). Slots from n + m - 1 on
// receive garbage.
void merge(const i32* da, const i32* db, const Chain (&first)[kChains], std::size_t length, u32* c) {
    std::size_t i[kChains];
    u32 value[kChains];
    for (std::size_t s = 0; s < kChains; ++s) {
        i[s] = first[s].i, value[s] = first[s].value;
        c[s * length] = value[s];
    }
    for (std::size_t step = 1; step < length; ++step) {
        for (std::size_t s = 0; s < kChains; ++s) {
            const std::size_t k = s * length + step - 1;  // c[k + 1] - c[k] = min(da[i], db[k - i])
            const i32 x = da[i[s]], y = db[k - i[s]];
            const bool take_a = x <= y;
            value[s] += u32(take_a ? x : y);
            i[s] += take_a;
            c[k + 1] = value[s];
        }
    }
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
    const std::size_t length = (count + kChains - 1) / kChains;
    const std::size_t c_words = (kChains * length + 15) / 16 * 16;
    const std::size_t text_words = columns::kTextBytes / sizeof(u32);
    u32* const memory = allocate(text_words + (n + kPad) + (m + kPad) + c_words);
    char* const text = reinterpret_cast<char*>(memory);
    u32* const a = memory + text_words;
    u32* const b = a + n + kPad;
    u32* const c = b + m + kPad;
    in.read(a, n);
    in.read(b, m);

    Chain first[kChains];
    for (std::size_t s = 0; s < kChains; ++s) first[s] = start(a, n, b, m, std::min(s * length, count - 1));
    to_slopes(a, n);
    to_slopes(b, m);
    merge(reinterpret_cast<const i32*>(a), reinterpret_cast<const i32*>(b), first, length, c);

    std::fill(c + count, c + c_words, 0);
    io::Writer out;
    columns::write(out, c, count, text);
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
