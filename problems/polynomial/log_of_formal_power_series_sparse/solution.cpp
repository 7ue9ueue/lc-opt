// log(f) mod x^N for f with K <= 10 nonzero terms (f[0] = 1), N <= 10^6. From f g' = f':
// G = n g solves G[n] = n f[n] - sum_k a_k G[n - i_k] (lib/poly/sparse.hpp), then g[n] = G[n] / n
// (lib/poly/divider.hpp), solved and printed in chunks that stay in the L2 cache. Output:
// fixed-width fields (fields.hpp), and groups of 16 zeros as "0 0 ... 0 " (judge-specific: the
// checker compares tokens).
#include <sys/mman.h>
#include <unistd.h>

#include "lib/io/io.hpp"
#include "lib/poly/divider.hpp"
#include "lib/poly/sparse.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

namespace sparse = poly::sparse;

// Coefficients per chunk, a multiple of 64 (Divider::kStep). Its text (up to 256 KB) is longer
// than the Writer's buffer, so the Writer hands it to write(2) directly.
constexpr std::size_t kChunk = 25600;
constexpr std::size_t kGroup = 16;  // values per zero test

// Zero-filled memory in transparent huge pages.
std::uint32_t* allocate(std::size_t words) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    const std::size_t bytes = (words * sizeof(std::uint32_t) + kHuge - 1) / kHuge * kHuge + kHuge;
    void* const region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) std::abort();
    const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region) + kHuge - 1) & ~(kHuge - 1);
    ::madvise(reinterpret_cast<void*>(aligned), bytes - kHuge, MADV_HUGEPAGE);
    return reinterpret_cast<std::uint32_t*>(aligned);
}

bool zero_group(const std::uint32_t* x) {
    const __m256i v = _mm256_or_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)),
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + 8)));
    return _mm256_testz_si256(v, v);
}

// Text of x[0, count) at p (16-byte aligned), x readable up to count rounded up to 16: each group
// of 16 values as 16 "0 " if all are zero, else as 16 fields of 10 bytes, each run of such groups
// by fields.hpp's pipelined loop. Returns the end.
char* format(const std::uint32_t* x, std::size_t count, char* p) {
    static constexpr char kZeros[] = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    const std::size_t extra = (kGroup - count % kGroup) % kGroup;  // values past count in the last group
    const std::uint32_t* const end = x + count + extra;
    std::size_t width = 10;  // bytes per value in the last group
    while (x < end) {
        const std::uint32_t* const run = x;
        while (x < end && !zero_group(x)) x += kGroup;
        if (x > run) {
            fields::detail::format(p, run, std::size_t(x - run), fields::detail::kConstants);
            p += 10 * (x - run);
            width = 10;
        }
        for (; x < end && zero_group(x); x += kGroup, p += 2 * kGroup) {
            std::memcpy(p, kZeros, 2 * kGroup);
            width = 2;
        }
    }
    return p - extra * width;
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), k = in.read<std::uint32_t>();
    // G[n] = n f[n] - sum_k a_k G[n - i_k] over k >= 1 (i_0 = 0, a_0 = 1).
    std::array<sparse::Term, 9> taps, rhs;
    in.read<std::uint32_t>(), in.read<std::uint32_t>();
    for (std::size_t i = 1; i < k; ++i) {
        const std::uint32_t index = in.read<std::uint32_t>(), a = in.read<std::uint32_t>();
        taps[i - 1] = {index, sparse::kModulus - a};
        rhs[i - 1] = {index, sparse::multiply(index, a)};
    }
    sparse::Recurrence recurrence(std::span<const sparse::Term>(taps.data(), k - 1),
                                  std::span<const sparse::Term>(rhs.data(), k - 1));
    sparse::Divider divider(n);

    // G in a ring of history + kChunk words if the taps reach back at most a chunk, divided in
    // place once the next chunk's history is saved; else G in one array after kPadding zeros and
    // g in a chunk of its own. The text follows, 16-byte aligned.
    constexpr std::size_t kStep = sparse::Divider::kStep;
    const auto round_up = [](std::size_t x) { return (x + kStep - 1) / kStep * kStep; };
    const std::size_t history = recurrence.history();
    const bool ring = history <= kChunk;
    const std::size_t before = ring ? history : sparse::Recurrence::kPadding;
    const std::size_t words = round_up(before + (ring ? kChunk : round_up(n) + kChunk));
    std::uint32_t* const area = allocate(words + 10 * kChunk / sizeof(std::uint32_t));
    char* const text = reinterpret_cast<char*>(area + words);
    std::uint32_t* const quotient = ring ? area + before : area + words - kChunk;
    io::Writer out;
    std::uint32_t* G = area + before;
    for (std::size_t i = 0; i < n; i += kChunk) {
        const std::size_t m = std::min(kChunk, n - i);
        recurrence.next(G, m);
        if (ring) std::copy(G + kChunk - history, G + kChunk, area);
        divider.divide(G, quotient, i, m);
        char* const end = format(quotient, m, text);
        if (i + m == n) end[-1] = '\n';
        out.write(std::string_view(text, std::size_t(end - text)));
        if (!ring) G += m;
    }
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
