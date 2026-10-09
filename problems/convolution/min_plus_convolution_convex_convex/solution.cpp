// c[k] = min over i + j = k of a[i] + b[j], a and b convex: c[0] = a[0] + b[0], and c's slopes are
// the slopes of a and b merged in ascending order, so c[k] is c[0] plus the k smallest slopes.
// Output blocks of columns::kBlock values are computed and written in turn. In a block, kChains
// independent chains each cover a range of k, interleaved so their latencies overlap. A chain
// starts from an argmin (i, k - i) of c[k], found by binary search: the slopes before it in a and
// b are k smallest ones, and the rest are no smaller. It then merges slopes eight at a time with
// a bitonic network (AVX2 min/max) and adds them up.
#include <sys/mman.h>
#include <unistd.h>

#include <climits>

#include "lib/io/io.hpp"
#include "../min_plus_convolution_convex_arbitrary/columns.hpp"

namespace {

using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t kChains = 4;
constexpr u32 kEnd = INT32_MAX;   // slope past the last element; above every real slope
constexpr std::size_t kPad = 96;  // elements after a and b with slope kEnd; chains read < 64 past

// Slopes are differences mod 2^32 read as i32: values are <= 1e9, so real slopes fit.
i32 slope(const u32* p) { return i32(p[1] - p[0]); }

__m256i slopes(const u32* p) {
    const auto at = [p](int i) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)); };
    return _mm256_sub_epi32(at(1), at(0));
}

// x[size, size + kPad): slope kEnd after x[size - 1].
void extend(u32* x, std::size_t size) {
    for (std::size_t i = size; i < size + kPad; ++i) x[i] = x[i - 1] + kEnd;
}

struct Chain {
    const u32* a;  // c[k] = *a + *b
    const u32* b;
};

// Argmin (i, k - i) of c[k], i leftmost: slope(a + i) - slope(b + k - i - 1) is nondecreasing in i.
Chain start(const u32* a, std::size_t n, const u32* b, std::size_t m, std::size_t k) {
    std::size_t lo = k + 1 > m ? k + 1 - m : 0, hi = std::min(k, n - 1);
    while (lo < hi) {
        const std::size_t i = (lo + hi) / 2;
        if (slope(a + i) >= slope(b + (k - i - 1))) hi = i;
        else lo = i + 1;
    }
    return {a + lo, b + (k - lo)};
}

// A bitonic vector sorted ascending: half-cleaners at distances 4, 2, 1.
__m256i sort_bitonic(__m256i v) {
    __m256i p = _mm256_permute2x128_si256(v, v, 0x01);
    v = _mm256_blend_epi32(_mm256_min_epi32(v, p), _mm256_max_epi32(v, p), 0xF0);
    p = _mm256_shuffle_epi32(v, 0x4E);
    v = _mm256_blend_epi32(_mm256_min_epi32(v, p), _mm256_max_epi32(v, p), 0xCC);
    p = _mm256_shuffle_epi32(v, 0xB1);
    return _mm256_blend_epi32(_mm256_min_epi32(v, p), _mm256_max_epi32(v, p), 0xAA);
}

// carry + inclusive prefix sums of s.
__m256i prefix_sums(__m256i s, __m256i carry) {
    s = _mm256_add_epi32(s, _mm256_slli_si256(s, 4));
    s = _mm256_add_epi32(s, _mm256_slli_si256(s, 8));
    const __m256i low_total = _mm256_shuffle_epi32(s, 0xFF);
    s = _mm256_add_epi32(s, _mm256_permute2x128_si256(low_total, low_total, 0x08));  // into the high half
    return _mm256_add_epi32(s, carry);
}

// c[0, size) = values k0 + [0, size) for 1 <= size <= columns::kBlock, k0 + size <= n + m - 1;
// c[size, size + 8 * kChains + 1) receives garbage. Chain s writes c[t + 1, t + length + 1) from
// its start k0 + t, t = s * length (clamped to the block). Per chain, `held` keeps the 8 largest
// slopes seen, sorted; each step loads 8 slopes from the input with the smaller next one, and the
// 8 smallest of the 16 are the next slopes (the classic SIMD merge).
void block(const u32* a, std::size_t n, const u32* b, std::size_t m, std::size_t k0, std::size_t size, u32* c) {
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0), last = _mm256_set1_epi32(7);
    const std::size_t length = ((size - 1 + kChains - 1) / kChains + 7) / 8 * 8;
    const u32* pa[kChains];
    const u32* pb[kChains];
    __m256i held[kChains], carry[kChains];
    for (std::size_t s = 0; s < kChains; ++s) {
        const Chain first = start(a, n, b, m, k0 + std::min(s * length, size - 1));
        pa[s] = first.a, pb[s] = first.b;
        carry[s] = _mm256_set1_epi32(i32(*pa[s] + *pb[s]));
        held[s] = slopes(pa[s]), pa[s] += 8;
    }
    c[0] = u32(_mm256_cvtsi256_si32(carry[0]));
    for (std::size_t step = 0; step < length; step += 8) {
#pragma GCC unroll 16
        for (std::size_t s = 0; s < kChains; ++s) {
            // Which input to read is data-dependent and unpredictable: select it with masks, not a branch.
            const std::size_t take_b = slope(pa[s]) > slope(pb[s]);
            const std::uintptr_t mask = 0 - take_b;
            const auto* from = reinterpret_cast<const u32*>((reinterpret_cast<std::uintptr_t>(pa[s]) & ~mask) |
                                                            (reinterpret_cast<std::uintptr_t>(pb[s]) & mask));
            const __m256i next = _mm256_permutevar8x32_epi32(slopes(from), reverse);
            pa[s] += 8 - 8 * take_b, pb[s] += 8 * take_b;
            const __m256i low = sort_bitonic(_mm256_min_epi32(held[s], next));
            held[s] = sort_bitonic(_mm256_max_epi32(held[s], next));
            const __m256i values = prefix_sums(low, carry[s]);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + s * length + step + 1), values);
            carry[s] = _mm256_permutevar8x32_epi32(values, last);
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
    constexpr std::size_t kValues = columns::kBlock + 8 * kChains + 16;  // block, garbage, zeros
    const std::size_t text_words = columns::kTextBytes / sizeof(u32);
    u32* const memory = allocate(text_words + kValues + (n + kPad) + (m + kPad));
    char* const text = reinterpret_cast<char*>(memory);
    u32* const c = memory + text_words;
    u32* const a = c + kValues;
    u32* const b = a + n + kPad;
    in.read(a, n);
    in.read(b, m);
    extend(a, n);
    extend(b, m);

    io::Writer out;
    for (std::size_t k0 = 0; k0 < count; k0 += columns::kBlock) {
        const std::size_t size = std::min(columns::kBlock, count - k0);
        block(a, n, b, m, k0, size, c);
        std::fill_n(c + size, 16, 0);
        columns::write(out, c, size, text);
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
