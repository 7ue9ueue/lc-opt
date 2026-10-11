// sqrt(f) mod x^N for f with K <= 10 nonzero terms, N <= 10^6. With f = a x^k (1 + h): no root if
// k is odd or a is not a square mod P. Else g = x^(k/2) s with s = c (1 + h)^(1/2), c^2 = a, and
// g^2 mod x^N depends on s mod x^(N - k) only: g's last k/2 coefficients are left zero.
// (1 + h) s' = h' s / 2 gives n s[n] = sum_d ((3/2) d c_d - c_d n) s[n - d], c_d the coefficients
// of h, s[0] = c (lib/poly/holonomic.hpp). s is solved and printed in chunks that stay in the L2
// cache. Output: fixed-width fields (fields.hpp), and groups of 16 zeros as "0 0 ... 0 "
// (judge-specific: the checker compares tokens).
#include <sys/mman.h>
#include <unistd.h>

#include <optional>

#include "lib/io/io.hpp"
#include "lib/poly/holonomic.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

namespace sparse = poly::sparse;

constexpr std::uint32_t kP = sparse::kModulus;

// Coefficients per chunk, a multiple of 16. Its text (up to 256 KB) is longer than the Writer's
// buffer, so the Writer hands it to write(2) directly. Chunks of 6400 take 0.35 ms more (lc-bench).
constexpr std::size_t kChunk = 25600;
constexpr std::size_t kGroup = 16;  // values per zero test
constexpr std::size_t kTextBytes = 10 * kChunk;
// Words for the ring of coefficients (history() <= kChunk) and the text, after the recurrence's
// table of reciprocals: one 2 MiB page fault less than in their own page.
constexpr std::size_t kSpare = 2 * kChunk + kTextBytes / sizeof(std::uint32_t);

// The smaller square root of a in [1, P), if a is a square (Tonelli and Shanks). P - 1 = 119 2^23,
// and 3 generates the multiplicative group.
std::optional<std::uint32_t> square_root(std::uint32_t a) {
    using sparse::multiply, sparse::power;
    constexpr std::uint32_t kOdd = 119;
    if (power(a, (kP - 1) / 2) != 1) return std::nullopt;
    // x^2 = a t; z has order 2^bits; the order of t divides 2^(bits - 1).
    std::uint32_t x = power(a, (kOdd + 1) / 2), t = power(a, kOdd), z = power(3, kOdd);
    for (int bits = 23; t != 1;) {
        int order = 0;  // t has order 2^order, order < bits
        for (std::uint32_t s = t; s != 1; s = multiply(s, s)) ++order;
        std::uint32_t b = z;  // order 2^(order + 1)
        for (int i = order + 1; i < bits; ++i) b = multiply(b, b);
        x = multiply(x, b), z = multiply(b, b), t = multiply(t, z), bits = order;
    }
    return std::min(x, kP - x);
}

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

// count values "0 " through text (bytes bytes); a newline replaces the last space if last.
void write_zeros(io::Writer& out, char* text, std::size_t bytes, std::size_t count, bool last) {
    const std::size_t step = std::min(count, bytes / 2);
    for (std::size_t i = 0; i < step; ++i) text[2 * i] = '0', text[2 * i + 1] = ' ';
    for (std::size_t i = 0; i < count; i += step) {
        const std::size_t m = std::min(step, count - i);
        if (last && i + m == count) text[2 * m - 1] = '\n';
        out.write(std::string_view(text, 2 * m));
    }
}

// s[0, n) of the recurrence printed between before and after zeros; n >= 1.
void print(io::Writer& out, sparse::Holonomic& recurrence, std::size_t before, std::size_t n, std::size_t after) {
    // Coefficients in a ring of history + kChunk words (in the recurrence's spare words) if the
    // taps reach back at most a chunk, else in one array after kPadding zeros. The text follows,
    // 16-byte aligned.
    const auto round_up = [](std::size_t x) { return (x + kGroup - 1) / kGroup * kGroup; };
    const std::size_t history = recurrence.history();
    const bool ring = history <= kChunk;
    const std::size_t padding = ring ? history : sparse::Holonomic::kPadding;
    const std::size_t words = round_up(padding + (ring ? kChunk : round_up(n)));
    std::uint32_t* const area = ring ? recurrence.spare() : allocate(words + kTextBytes / sizeof(std::uint32_t));
    char* const text = reinterpret_cast<char*>(area + words);
    write_zeros(out, text, kTextBytes, before, false);
    std::uint32_t* s = area + padding;
    for (std::size_t i = 0; i < n; i += kChunk) {
        const std::size_t m = std::min(kChunk, n - i);
        recurrence.next(s, m);
        char* const end = format(s, m, text);
        if (i + m == n && after == 0) end[-1] = '\n';
        out.write(std::string_view(text, std::size_t(end - text)));
        if (ring) std::copy(s + kChunk - history, s + kChunk, area);
        else s += m;
    }
    write_zeros(out, text, kTextBytes, after, true);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), count = in.read<std::uint32_t>();
    std::array<sparse::Term, 10> terms;
    for (std::size_t i = 0; i < count; ++i) terms[i].index = in.read<std::uint32_t>(), terms[i].value = in.read<std::uint32_t>();
    io::Writer out;
    if (count == 0) {  // f = 0 = 0^2
        char text[2 * kChunk];
        return write_zeros(out, text, sizeof text, n, true);
    }
    const std::size_t k = terms[0].index;
    const std::optional<std::uint32_t> root = k % 2 ? std::nullopt : square_root(terms[0].value);
    if (!root) return out.write(std::string_view("-1\n"));

    // f = a x^k (1 + h), sqrt(f) = x^(k/2) c (1 + h)^(1/2): taps (d, (3/2) d c_d, -c_d).
    const std::size_t size = n - k;  // coefficients of s that g^2 mod x^n depends on
    const std::uint32_t inverse_a = sparse::inverse(terms[0].value);
    constexpr std::uint32_t kThreeHalves = (kP + 3) / 2;
    std::array<sparse::Tap, 10> taps;
    std::size_t used = 0;
    for (std::size_t i = 1; i < count; ++i) {
        const std::uint32_t d = std::uint32_t(terms[i].index - k);
        if (d >= size) break;  // never reaches a printed coefficient
        const std::uint32_t c = sparse::multiply(terms[i].value, inverse_a);
        taps[used++] = {d, sparse::multiply(sparse::multiply(kThreeHalves, d), c), kP - c};
    }
    sparse::Holonomic recurrence(std::span<const sparse::Tap>(taps.data(), used), *root, size, kSpare);
    print(out, recurrence, k / 2, size, k / 2);
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
