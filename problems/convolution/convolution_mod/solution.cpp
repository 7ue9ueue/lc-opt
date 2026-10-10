// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1, output in fixed-width fields
// (fields.hpp). Factors of at most half the length (all large tests) use ntt::Product
// (lib/ntt/product.hpp), other sizes ntt::Convolution. Inputs of 9-digit or 1-digit tokens take a
// fixed-width parser.
#include <unistd.h>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/ntt/product.hpp"
#include "fields.hpp"

namespace {


// Input fast paths for inputs whose tokens all have 9 digits (13 of the 16 large tests) or 1 digit
// (all_same_00), each followed by one separator: token i then starts at a fixed stride, so blocks
// need no separator search. Each block is checked by its separator mask; the first block that
// fails, and everything after it, goes to io::read_bulk.
namespace fixed_width {

constexpr char Z = char(0x80);  // shuffle control: zero byte

__m256i load(const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); }

// Bit i set iff byte i is a separator (at most ' ').
std::uint32_t separators(__m256i bytes) {
    return std::uint32_t(_mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_set1_epi8(' ' + 1), bytes)));
}

__m256i digits(__m256i bytes, const char (&order)[32]) {
    return _mm256_shuffle_epi8(_mm256_subs_epu8(bytes, _mm256_set1_epi8('0')), load(order));
}

// A 32-byte load at token t of a 9-digit block holds t at byte 0 and t + 2 at byte 20 (byte 4 of
// the high lane); its separator mask is kNineSeparators.
constexpr std::uint32_t kNineSeparators = 1u << 9 | 1u << 19 | 1u << 29;
alignas(32) constexpr char kNineOrder[32] = {1, 2, 3, 4, 5, 6, 7, 8, Z, 0, Z, Z, Z, Z, Z, Z,      // [d1..d8, 0, d0]
                                             5, 6, 7, 8, 9, 10, 11, 12, Z, 4, Z, Z, Z, Z, Z, Z};  // the same at byte 4
alignas(32) constexpr std::int8_t kTens[32] = {10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0, 0, 0,
                                               10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0, 0, 0, 0, 0};

// Per lane, dwords [d1..d4, d5..d8, d0, 0] of the tokens at bytes 0 and 20.
__m256i groups(__m256i bytes) {
    const __m256i pairs = _mm256_maddubs_epi16(digits(bytes, kNineOrder), load(kTens));
    return _mm256_madd_epi16(pairs, _mm256_setr_epi16(100, 1, 100, 1, 1, 0, 0, 0, 100, 1, 100, 1, 1, 0, 0, 0));
}

// Token values in the even dwords, [t - u - | v - w -], for groups x = [t | v] and y = [u | w].
__m256i values(__m256i x, __m256i y) {
    const __m256i halves = _mm256_madd_epi16(_mm256_packus_epi32(x, y),
                                             _mm256_setr_epi16(10000, 1, 1, 0, 10000, 1, 1, 0, 10000, 1, 1, 0, 10000, 1, 1, 0));
    return _mm256_add_epi32(halves, _mm256_mul_epu32(_mm256_srli_epi64(halves, 32), _mm256_set1_epi32(100000000)));
}

// The 8 tokens at p (80 bytes) into dst, stored in any case; true if all have 9 digits and one
// separator. Reads p[0, 82).
bool nine_digits(const char* p, std::uint32_t* dst) {
    const __m256i l0 = load(p), l1 = load(p + 10), l2 = load(p + 40), l3 = load(p + 50);
    const std::uint32_t wrong = (separators(l0) ^ kNineSeparators) | (separators(l1) ^ kNineSeparators) |
                                (separators(l2) ^ kNineSeparators) | ((separators(l3) ^ kNineSeparators) & 0x3FFFFFFF);
    const __m256i low = values(groups(l0), groups(l1)), high = values(groups(l2), groups(l3));
    const __m256i mixed = _mm256_blend_epi32(low, _mm256_slli_epi64(high, 32), 0xAA);  // tokens 0 4 1 5 | 2 6 3 7
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), _mm256_permutevar8x32_epi32(mixed, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
    return !wrong;
}

alignas(32) constexpr char kOneOrder[32] = {0, 2, 4, 6, 8, 10, 12, 14, Z, Z, Z, Z, Z, Z, Z, Z,
                                            0, 2, 4, 6, 8, 10, 12, 14, Z, Z, Z, Z, Z, Z, Z, Z};

// As nine_digits for the 16 tokens at p (32 bytes) of one digit each. Reads p[0, 32).
bool one_digit(const char* p, std::uint32_t* dst) {
    const __m256i bytes = load(p), d = digits(bytes, kOneOrder);
    auto* out = reinterpret_cast<__m256i*>(dst);
    _mm256_storeu_si256(out, _mm256_cvtepu8_epi32(_mm256_castsi256_si128(d)));
    _mm256_storeu_si256(out + 1, _mm256_cvtepu8_epi32(_mm256_extracti128_si256(d, 1)));
    return separators(bytes) == 0xAAAAAAAA;
}

// Tokens parsed from p, at most count, in blocks of Block tokens of Stride bytes each.
template <std::size_t Stride, std::size_t Block, bool (*Parse)(const char*, std::uint32_t*)>
std::size_t parse(const char* p, std::uint32_t* dst, std::size_t count) {
    std::size_t done = 0;
    while (done + Block <= count && Parse(p + Stride * done, dst + done)) done += Block;
    return done;
}

}  // namespace fixed_width

// count values into dst: the fixed-width fast paths, then io::read_bulk for the rest.
void read_values(io::Reader& in, std::uint32_t* dst, std::size_t count) {
    const char* p = in.scan().cur;
    while (static_cast<unsigned char>(*p) <= ' ') ++p;
    std::size_t stride = 10, done = fixed_width::parse<10, 8, fixed_width::nine_digits>(p, dst, count);
    if (!done) stride = 2, done = fixed_width::parse<2, 16, fixed_width::one_digit>(p, dst, count);
    if (done) {  // continue the Reader after the parsed tokens, as io::detail::read_transposed does
        const char* const stop = p + stride * done;
        const auto* block = reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(stop) & ~std::uintptr_t(63));
        in.resume({stop, block, io::detail::block_separators(block) & ~std::uint64_t(0) << (stop - block)});
    }
    io::read_bulk(in, dst + done, count - done);
}

// fields::kTextBytes for the output. Product's sit after its tables: at 2^20 they share the
// tables' huge page (9.25 of 10 MiB used), so they cost no page faults.
char* text(ntt::Product& product) { return static_cast<char*>(product.extra()); }

char* text(ntt::Convolution&) {
    alignas(64) static char buffer[fields::kTextBytes];
    return buffer;
}

template <class Multiplier>
void convolve(io::Reader& in, Multiplier& product, std::size_t n, std::size_t m) {
    read_values(in, product.a(), n);
    read_values(in, product.b(), m);
    const std::uint32_t* c = product.multiply();
    io::Writer out;
    fields::write(out, c, n + m - 1, text(product));
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    if (ntt::Product::fits(n, m)) {
        ntt::Product product(n, m, fields::kTextBytes);
        return convolve(in, product, n, m);
    }
    ntt::Convolution convolution(n, m);
    convolve(in, convolution, n, m);
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here): 0.15 ms less per run. _exit skips their teardown too.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
