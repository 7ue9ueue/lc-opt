// c[k] = min over i + j = k of a[i] + b[j], a and b convex: c[0] = a[0] + b[0], and c's slopes are
// the slopes of a and b merged in ascending order, so c[k] is c[0] plus the k smallest slopes.
// The slopes are merged eight at a time with a bitonic network (AVX2 min/max), in kChains
// independent chains over consecutive ranges of k so their latencies overlap. Each chain starts
// from an argmin (i, k - i) of c[k], found by binary search: the slopes before it in a and b are k
// smallest ones, and the rest are no smaller.
#include <sys/mman.h>
#include <unistd.h>

#include <climits>

#include "lib/io/io.hpp"
#include "../min_plus_convolution_convex_arbitrary/columns.hpp"

namespace {

using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t kChains = 4;
constexpr i32 kEnd = INT32_MAX;    // slope past the last element; above every real slope
constexpr std::size_t kPad = 128;  // kEnd slopes after each array

struct Chain {
    std::size_t k, i;  // c[k] = a[i] + b[k - i]
    u32 value;         // c[k]
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
    return {k, lo, a[lo] + b[k - lo]};
}

// x[i] = x[i + 1] - x[i] for i < size - 1, then kPad slopes kEnd from x[size - 1] on.
void to_slopes(u32* x, std::size_t size) {
    for (std::size_t i = 0; i + 1 < size; ++i) x[i] = x[i + 1] - x[i];
    std::fill_n(x + size - 1, kPad, u32(kEnd));
}

__m256i load(const i32* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }

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

// c[1, kChains * length + 1), length a multiple of 8: chain s writes c[k + 1, k + length + 1) for
// its start k = s * length. Values from n + m - 1 on are garbage. Per chain, `held` keeps the 8
// largest slopes seen, sorted; each step loads 8 slopes from the input with the smaller next one,
// and the 8 smallest of the 16 are the next slopes (the classic SIMD merge).
void merge(const i32* da, const i32* db, const Chain (&first)[kChains], std::size_t length, u32* c) {
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0), last = _mm256_set1_epi32(7);
    const i32* pa[kChains];
    const i32* pb[kChains];
    __m256i held[kChains], carry[kChains];
    for (std::size_t s = 0; s < kChains; ++s) {
        pa[s] = da + first[s].i, pb[s] = db + (first[s].k - first[s].i);
        held[s] = load(pa[s]), pa[s] += 8;
        carry[s] = _mm256_set1_epi32(i32(first[s].value));
    }
    for (std::size_t step = 0; step < length; step += 8) {
        for (std::size_t s = 0; s < kChains; ++s) {
            const bool take_a = *pa[s] <= *pb[s];
            const __m256i next = _mm256_permutevar8x32_epi32(load(take_a ? pa[s] : pb[s]), reverse);
            pa[s] += take_a ? 8 : 0, pb[s] += take_a ? 0 : 8;
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
    const std::size_t length = ((count - 1 + kChains - 1) / kChains + 7) / 8 * 8;
    const std::size_t c_words = (kChains * length + 1 + 15) / 16 * 16;
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
    c[0] = first[0].value;
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
