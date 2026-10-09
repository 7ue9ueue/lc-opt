// Fast integer and text I/O on file descriptors (stdin and stdout by default).
// Linux, x86-64 with AVX2. Design and measurements: lib/io/notes.md.
//
//   io::Reader in;                            // takes all of stdin
//   io::Writer out;                           // flushes in its destructor
//   const auto n = in.read<std::uint32_t>();
//   std::vector<std::uint32_t> a(n);
//   in.read(a.data(), n);                     // bulk; AVX2 for uint32_t
//   out.write("n = ", n, '\n');               // any mix of integers, chars, strings
//   out.write_array(a.data(), n, ' ');        // bulk; AVX2 for uint32_t
//   out.write('\n');
//
// Input: tokens separated by whitespace (any bytes <= ' '). Integer tokens are an optional '-'
// (signed types only) followed by decimal digits, and fit the requested type. Reading past the
// last token is undefined.
#pragma once

#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace io {
namespace detail {

// Zero bytes after the input. Loads start at or before the end of the input and reach at most
// 63 bytes past it.
inline constexpr std::size_t kPadding = 64;
inline constexpr std::size_t kPage = 4096;

// Input as mapped or read: data at start, kPadding zero bytes after its end.
struct Input {
    char* base;           // 64-byte aligned
    std::size_t mapped;   // bytes mapped, or 0 for a heap buffer
    const char* start;
};

inline Input map_input(int fd, std::size_t size) {
    const off_t offset = ::lseek(fd, 0, SEEK_CUR);
    if (offset < 0 || std::size_t(offset) >= size) return {};
    const std::size_t file_bytes = (size + kPage - 1) & ~(kPage - 1);
    // Bytes past the end of the file up to its last page boundary read as zero. If fewer than
    // kPadding remain, reserve one more page, anonymous and zero, after the file.
    const std::size_t mapped = file_bytes - size >= kPadding ? file_bytes : file_bytes + kPage;
    void* base;
    if (mapped == file_bytes) {
        base = ::mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE, fd, 0);
    } else {
        base = ::mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base != MAP_FAILED && ::mmap(base, file_bytes, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED) {
            ::munmap(base, mapped);
            base = MAP_FAILED;
        }
    }
    if (base == MAP_FAILED) return {};
    return {static_cast<char*>(base), mapped, static_cast<char*>(base) + offset};
}

inline char* allocate(std::size_t capacity) {
    auto* p = static_cast<char*>(std::aligned_alloc(64, capacity + kPadding));
    if (!p) std::abort();
    return p;
}

inline Input read_input(int fd) {
    std::size_t size = 0, capacity = std::size_t(1) << 16;
    char* base = allocate(capacity);
    for (;;) {
        if (size == capacity) {
            char* bigger = allocate(capacity *= 2);
            std::memcpy(bigger, base, size);
            std::free(base);
            base = bigger;
        }
        const ssize_t got = ::read(fd, base + size, capacity - size);
        if (got > 0) size += std::size_t(got);
        else if (got == 0) break;
        else if (errno != EINTR) std::abort();
    }
    std::memset(base + size, 0, kPadding);
    return {base, 0, base};
}

// Regular files are mapped, other inputs read until end of file.
inline Input open_input(int fd) {
    struct stat st;
    if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode))
        if (const Input input = map_input(fd, std::size_t(st.st_size)); input.base) return input;
    return read_input(fd);
}

inline void release(const Input& input) {
    if (input.mapped) ::munmap(input.base, input.mapped);
    else std::free(input.base);
}

// Bit i set iff byte i is whitespace (<= ' ').
inline unsigned separators(__m128i bytes) {
    return unsigned(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_set1_epi8(' ' + 1), bytes)));
}

inline std::uint32_t separators(__m256i bytes) {
    return std::uint32_t(_mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_set1_epi8(' ' + 1), bytes)));
}

// Separators in the 64-byte aligned block.
inline std::uint64_t block_separators(const char* block) {
    const std::uint64_t low = separators(_mm256_load_si256(reinterpret_cast<const __m256i*>(block)));
    const std::uint64_t high = separators(_mm256_load_si256(reinterpret_cast<const __m256i*>(block + 32)));
    return low | high << 32;
}

// row[n + 1] moves the first n bytes (1 <= n <= 16) of a 16-byte window to its end and zeroes
// the rest. Rows for other lengths (-1, 0, 17..32) are all zero with a negative last byte, which
// marks the token as invalid.
struct AlignTable {
    alignas(16) std::int8_t row[34][16];
};

inline constexpr AlignTable kAlign = [] {
    AlignTable t{};
    for (int n = -1; n <= 32; ++n)
        for (int j = 0; j < 16; ++j)
            t.row[n + 1][j] = n >= 1 && n <= 16 && j >= 16 - n ? std::int8_t(j - (16 - n)) : std::int8_t(-128);
    return t;
}();

// Shuffle control for a token of length index - 1.
inline __m128i align_row(unsigned index) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(kAlign.row[index]));
}

// Right-aligned digits (byte values 0..9) -> 4-digit groups, one per dword.
inline __m128i digit_groups(__m128i digits) {
    return _mm_madd_epi16(_mm_maddubs_epi16(digits, _mm_set1_epi16(0x010A)), _mm_set1_epi32(0x00010064));
}

inline __m256i digit_groups(__m256i digits) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(digits, _mm256_set1_epi16(0x010A)), _mm256_set1_epi32(0x00010064));
}

// Value of the first n digits (0 <= n <= 16) of the window. Result < 10^16.
inline std::uint64_t parse16(__m128i window, unsigned n) {
    const __m128i digits = _mm_shuffle_epi8(_mm_subs_epu8(window, _mm_set1_epi8('0')), align_row(n + 1));
    const __m128i groups = digit_groups(digits);
    const __m128i halves = _mm_madd_epi16(_mm_packus_epi32(groups, groups), _mm_set1_epi32(0x00012710));
    const auto both = std::uint64_t(_mm_cvtsi128_si64(halves));  // 8-digit halves, high one first
    return (both & 0xFFFFFFFF) * 100000000 + (both >> 32);
}

// Value of the digits [s, s + n), n <= 20.
inline std::uint64_t parse20(const char* s, unsigned n) {
    // The last min(n, 16) digits in the high lane, the 0..4 before them in the low lane.
    const unsigned head = n > 16 ? n - 16 : 0;
    const __m256i windows =
        _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(s + head), reinterpret_cast<const __m128i*>(s));
    const __m256i rows = _mm256_set_m128i(align_row(n - head + 1), align_row(head + 1));
    const __m256i groups = digit_groups(_mm256_shuffle_epi8(_mm256_subs_epu8(windows, _mm256_set1_epi8('0')), rows));
    const __m256i halves = _mm256_madd_epi16(_mm256_packus_epi32(groups, groups), _mm256_set1_epi32(0x00012710));
    const std::uint64_t head_value = std::uint64_t(_mm256_extract_epi64(halves, 0)) >> 32;
    const auto tail = std::uint64_t(_mm256_extract_epi64(halves, 2));
    return (head_value * 100000000 + (tail & 0xFFFFFFFF)) * 100000000 + (tail >> 32);
}

// kQuad[x] for x < 10^4: its 4 digits with leading zeros, first digit in the lowest byte.
alignas(64) inline constexpr auto kQuad = [] {
    std::array<std::uint32_t, 10000> t{};
    for (std::uint32_t x = 0; x < t.size(); ++x)
        t[x] = ('0' + x / 1000) | ('0' + x / 100 % 10) << 8 | ('0' + x / 10 % 10) << 16 | ('0' + x % 10) << 24;
    return t;
}();

// Digits of x (x >= 1) = (x + kDigitCount32[floor(log2(x))]) >> 32: the low word of an entry
// carries into the high word exactly when x reaches the next power of ten.
inline constexpr auto kDigitCount32 = [] {
    std::array<std::uint64_t, 32> t{};
    for (int i = 0; i < 32; ++i) {
        const std::uint64_t low = std::uint64_t(1) << i, high = 2 * low - 1;
        std::uint64_t digits = 0, power = 1;
        while (power <= low) power *= 10, ++digits;
        t[i] = (digits << 32) + (power <= high ? (std::uint64_t(1) << 32) - power : 0);
    }
    return t;
}();

inline unsigned digit_count(std::uint32_t x) {
    return unsigned((x + kDigitCount32[31 - std::countl_zero(x | 1)]) >> 32);
}

// Digits of x = kDigitCount64[0][b] + (x > kDigitCount64[1][b]), where b = bit width of x | 1.
inline constexpr auto kDigitCount64 = [] {
    std::array<std::array<std::uint64_t, 65>, 2> t{};
    for (int b = 1; b <= 64; ++b) {
        const unsigned __int128 low = (unsigned __int128)1 << (b - 1), high = 2 * low - 1;
        std::uint64_t digits = 1;
        unsigned __int128 power = 10;
        while (power <= low) power *= 10, ++digits;
        t[0][b] = digits;
        t[1][b] = power <= high ? std::uint64_t(power - 1) : ~std::uint64_t(0);
    }
    return t;
}();

inline unsigned digit_count(std::uint64_t x) {
    const int b = std::bit_width(x | 1);
    return unsigned(kDigitCount64[0][b] + (x > kDigitCount64[1][b]));
}

// Bytes kShiftWindow[k..k + 16) shuffle a vector down by k bytes.
alignas(32) inline constexpr std::int8_t kShiftWindow[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                             11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

inline __m128i shift_down(__m128i v, unsigned k) {
    return _mm_shuffle_epi8(v, _mm_loadu_si128(reinterpret_cast<const __m128i*>(kShiftWindow + k)));
}

// Writes x in decimal at p and returns the end. Stores 16 bytes.
inline char* format(char* p, std::uint32_t x) {
    // Twelve digits with leading zeros, as three groups of four, then shifted down past the zeros.
    const std::uint32_t top = x / 100000000, rest = x % 100000000;
    const std::uint32_t high = rest / 10000, low = rest % 10000;
    const __m128i digits = _mm_insert_epi32(
        _mm_cvtsi64_si128(std::int64_t(kQuad[top] | std::uint64_t(kQuad[high]) << 32)), int(kQuad[low]), 2);
    const unsigned n = digit_count(x);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), shift_down(digits, 12 - n));
    return p + n;
}

// x < 10^16 as 16 digits with leading zeros.
inline __m128i sixteen_digits(std::uint64_t x) {
    const auto high = std::uint32_t(x / 100000000), low = std::uint32_t(x % 100000000);
    return _mm_set_epi64x(std::int64_t(kQuad[low / 10000] | std::uint64_t(kQuad[low % 10000]) << 32),
                          std::int64_t(kQuad[high / 10000] | std::uint64_t(kQuad[high % 10000]) << 32));
}

// Writes x in decimal at p and returns the end. Stores up to 20 bytes.
inline char* format(char* p, std::uint64_t x) {
    // Twenty digits with leading zeros: a head of four, then sixteen.
    const auto top = std::uint32_t(x / 10000000000000000);
    const unsigned n = digit_count(x);
    const unsigned head = n > 16 ? n - 16 : 0;
    const auto head_digits = std::uint32_t(std::uint64_t(kQuad[top]) >> (32 - 8 * head));
    std::memcpy(p, &head_digits, 4);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p + head),
                     shift_down(sixteen_digits(x % 10000000000000000), 16 - (n - head)));
    return p + n;
}

// floor(x / d) for every dword x, as (x * Magic) >> Shift. The test checks each use exhaustively.
template <std::uint32_t Magic, int Shift>
inline __m256i divide(__m256i x) {
    const __m256i magic = _mm256_set1_epi64x(Magic);
    const __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(x, magic), Shift);
    const __m256i odd = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), magic), Shift);
    return _mm256_blend_epi32(even, _mm256_slli_epi64(odd, 32), 0xAA);
}

// Each dword g < 10^4 -> its 4 ASCII digits, first digit in the lowest byte.
inline __m256i quad_digits(__m256i g) {
    const __m256i high = _mm256_srli_epi16(_mm256_mulhi_epu16(g, _mm256_set1_epi16(5243)), 3);  // g / 100
    const __m256i low = _mm256_sub_epi16(g, _mm256_mullo_epi16(high, _mm256_set1_epi16(100)));
    const __m256i pairs = _mm256_or_si256(high, _mm256_slli_epi32(low, 16));  // 16-bit halves < 100
    const __m256i tens = _mm256_mulhi_epu16(pairs, _mm256_set1_epi16(6554));  // pair / 10
    const __m256i units = _mm256_sub_epi16(pairs, _mm256_mullo_epi16(tens, _mm256_set1_epi16(10)));
    return _mm256_or_si256(_mm256_or_si256(tens, _mm256_slli_epi16(units, 8)), _mm256_set1_epi8('0'));
}

// Digit counts of the eight dwords of v, one per byte in order.
inline std::uint64_t digit_counts(__m256i v) {
    // Unsigned v >= 10^k via a signed compare after flipping the sign bits; summed as a tree.
    const __m256i x = _mm256_xor_si256(v, _mm256_set1_epi32(int(0x80000000)));
    const auto at_least = [&](std::uint32_t power) {
        return _mm256_cmpgt_epi32(x, _mm256_set1_epi32(int((power - 1) ^ 0x80000000)));
    };
    const __m256i a = _mm256_add_epi32(_mm256_add_epi32(at_least(10), at_least(100)),
                                       _mm256_add_epi32(at_least(1000), at_least(10000)));
    const __m256i b = _mm256_add_epi32(_mm256_add_epi32(at_least(100000), at_least(1000000)),
                                       _mm256_add_epi32(at_least(10000000), at_least(100000000)));
    const __m256i n = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_set1_epi32(1), at_least(1000000000)),
                                       _mm256_add_epi32(a, b));
    const __m256i bytes = _mm256_packus_epi16(_mm256_packus_epi32(n, n), n);  // per lane: 4 counts first
    return std::uint64_t(_mm256_extract_epi32(bytes, 0)) | std::uint64_t(_mm256_extract_epi32(bytes, 4)) << 32;
}

// Stores the two 16-byte lanes of w, values of 12 digits with leading zeros and a separator, as
// their last n0 and n1 digits and the separator. Returns the end; stores 16 bytes per lane.
inline char* put_pair(char* p, __m256i w, unsigned n0, unsigned n1) {
    const __m256i shifted = _mm256_shuffle_epi8(w, _mm256_loadu2_m128i(
        reinterpret_cast<const __m128i*>(kShiftWindow + 12 - n1), reinterpret_cast<const __m128i*>(kShiftWindow + 12 - n0)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_castsi256_si128(shifted));
    p += n0 + 1;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_extracti128_si256(shifted, 1));
    return p + n1 + 1;
}

// Writes the eight dwords of v in decimal, each followed by separator, and returns the end.
// Stores up to 15 bytes past the end.
inline char* format8(char* p, __m256i v, char separator) {
    // Digit counts first, from v: the store addresses then do not wait for the digits.
    const std::uint64_t n = digit_counts(v);
    // Lanes [v0 v2 v4 v6 | v1 v3 v5 v7], so the transpose below pairs consecutive values.
    v = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    const __m256i top = divide<1441151881, 57>(v);  // v / 10^8
    const __m256i rest = _mm256_sub_epi32(v, _mm256_mullo_epi32(top, _mm256_set1_epi32(100000000)));
    const __m256i high = divide<3518437209, 45>(rest);  // rest / 10^4
    const __m256i low = _mm256_sub_epi32(rest, _mm256_madd_epi16(high, _mm256_set1_epi32(10000)));
    const __m256i t = quad_digits(top), h = quad_digits(high), l = quad_digits(low);
    const __m256i s = _mm256_set1_epi32(static_cast<unsigned char>(separator));
    // Transpose: each 16-byte lane becomes one value as dwords [top, high, low, separator].
    const __m256i th_low = _mm256_unpacklo_epi32(t, h), th_high = _mm256_unpackhi_epi32(t, h);
    const __m256i ls_low = _mm256_unpacklo_epi32(l, s), ls_high = _mm256_unpackhi_epi32(l, s);
    const auto count = [n](int i) { return unsigned(n >> 8 * i & 0xFF); };
    p = put_pair(p, _mm256_unpacklo_epi64(th_low, ls_low), count(0), count(1));
    p = put_pair(p, _mm256_unpackhi_epi64(th_low, ls_low), count(2), count(3));
    p = put_pair(p, _mm256_unpacklo_epi64(th_high, ls_high), count(4), count(5));
    return put_pair(p, _mm256_unpackhi_epi64(th_high, ls_high), count(6), count(7));
}

// Parses uint32 tokens from the token at p into dst, at most count and all but fewer than
// kMinTokens of them. Returns where it stopped and the tokens parsed. Each chunk of input is cut
// into four streams at token boundaries. The streams advance in lockstep, two tokens per step, so
// four independent pointer chains overlap.
class BulkParser {
public:
    static constexpr std::size_t kMinTokens = 1024;

    struct Result {
        const char* end;
        std::size_t parsed;
    };

    static Result parse(const char* p, std::uint32_t* dst, std::size_t count) {
        std::uint32_t* const first = dst;
        while (count >= kMinTokens) {
            // With tokens of at most 16 bytes a chunk spans at most chunk + 33 bytes, so it holds
            // at most chunk / 2 + 17 < count tokens: all of them belong to this array.
            const std::size_t chunk = std::min(kChunk, 2 * (count - 32)) & ~std::size_t(3);
            const std::size_t stream = chunk / 4;
            const char* s[4] = {p, after_separator(p + stream), after_separator(p + 2 * stream),
                                after_separator(p + 3 * stream)};
            const char* const end[4] = {s[1], s[2], s[3], after_separator(p + chunk)};
            std::uint32_t* const out[4] = {dst, scratch_[0], scratch_[1], scratch_[2]};
            __m256i invalid = _mm256_setzero_si256();
            const std::size_t done = lockstep(s, end, out, invalid);
            std::size_t n[4];
            bool overrun = false;
            for (int k = 0; k < 4; ++k) {
                n[k] = done;
                while (s[k] < end[k] && n[k] < kCapacity) out[k][n[k]++] = one_token(s[k], invalid);
                overrun |= s[k] != end[k];
            }
            if (overrun || (std::uint32_t(_mm256_movemask_epi8(invalid)) & 0x80008000)) [[unlikely]] {
                // Irregular whitespace or tokens: this chunk one token at a time.
                const std::size_t tokens = std::min(count_tokens(p, end[3]), count);
                p = parse_slowly(p, dst, tokens);
                dst += tokens;
                count -= tokens;
                continue;
            }
            std::uint32_t* next = dst + n[0];
            for (int k = 1; k < 4; ++k) {
                std::memcpy(next, out[k], n[k] * sizeof(std::uint32_t));
                next += n[k];
            }
            dst = next;
            count -= n[0] + n[1] + n[2] + n[3];
            p = skip_whitespace(end[3]);
        }
        return {p, std::size_t(dst - first)};
    }

private:
    static constexpr std::size_t kChunk = std::size_t(1) << 17;
    // A step advances a stream by 2..33 bytes and yields two values, so a stream of at most
    // kChunk / 4 + 33 bytes yields at most kChunk / 4 + 66 values.
    static constexpr std::size_t kCapacity = kChunk / 4 + 66;
    alignas(64) static inline std::uint32_t scratch_[3][kCapacity];

    // Steps all streams while each has 33 bytes left; returns the values per stream. Works on
    // local copies of the stream state, which stay in registers.
    static std::size_t lockstep(const char* (&streams)[4], const char* const (&end)[4],
                                std::uint32_t* const (&out)[4], __m256i& flags) {
        const char* s[4] = {streams[0], streams[1], streams[2], streams[3]};
        __m256i invalid = flags;
        std::size_t done = 0;
        for (;;) {
            std::ptrdiff_t left = end[0] - s[0];
            for (int k = 1; k < 4; ++k) left = std::min(left, end[k] - s[k]);
            if (left < 33) break;
            for (std::ptrdiff_t steps = left / 33; steps; --steps, done += 2) {
                const __m256i g0 = two_tokens(s[0], invalid), g1 = two_tokens(s[1], invalid);
                const __m256i g2 = two_tokens(s[2], invalid), g3 = two_tokens(s[3], invalid);
                const __m256i k = _mm256_set1_epi32(0x00012710);
                // 8-digit halves: [s0 high, s0 low, s1 high, s1 low | second tokens likewise].
                const __m256 h01 = _mm256_castsi256_ps(_mm256_madd_epi16(_mm256_packus_epi32(g0, g1), k));
                const __m256 h23 = _mm256_castsi256_ps(_mm256_madd_epi16(_mm256_packus_epi32(g2, g3), k));
                const __m256i high = _mm256_castps_si256(_mm256_shuffle_ps(h01, h23, 0x88));
                const __m256i low = _mm256_castps_si256(_mm256_shuffle_ps(h01, h23, 0xDD));
                const __m256i v = _mm256_add_epi32(_mm256_mullo_epi32(high, _mm256_set1_epi32(100000000)), low);
                // [s0 a, s0 b, s1 a, s1 b | s2 a, s2 b, s3 a, s3 b]
                const __m256i q = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                const __m128i q01 = _mm256_castsi256_si128(q), q23 = _mm256_extracti128_si256(q, 1);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(out[0] + done), q01);
                _mm_storeh_pd(reinterpret_cast<double*>(out[1] + done), _mm_castsi128_pd(q01));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(out[2] + done), q23);
                _mm_storeh_pd(reinterpret_cast<double*>(out[3] + done), _mm_castsi128_pd(q23));
            }
        }
        for (int k = 0; k < 4; ++k) streams[k] = s[k];
        flags = invalid;
        return done;
    }

    // Two tokens at s, as 4-digit groups [first | second]. Lengths outside 1..16 (repeated
    // whitespace, long tokens) set the sign bit of byte 15 or 31 of invalid.
    [[gnu::always_inline]] static __m256i two_tokens(const char*& s, __m256i& invalid) {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
        const std::uint32_t sep = separators(bytes);
        const unsigned first = unsigned(std::countr_zero(sep));             // length of the first token
        const unsigned second = unsigned(std::countr_zero(sep & (sep - 1)));  // 32 if none
        const __m256i windows =
            _mm256_inserti128_si256(bytes, _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + first + 1)), 1);
        const __m256i rows = _mm256_set_m128i(align_row(second - first), align_row(first + 1));
        invalid = _mm256_or_si256(invalid, rows);
        s += second + 1;
        return digit_groups(_mm256_shuffle_epi8(_mm256_subs_epu8(windows, _mm256_set1_epi8('0')), rows));
    }

    static std::uint32_t one_token(const char*& s, __m256i& invalid) {
        const __m128i window = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
        const unsigned n = unsigned(std::countr_zero(separators(window) | 0x10000));
        invalid = _mm256_or_si256(invalid, _mm256_castsi128_si256(align_row(n + 1)));
        s += n + 1;
        return std::uint32_t(parse16(window, n));
    }

    // The byte after the first separator at or after q.
    static const char* after_separator(const char* q) {
        for (;; q += 32)
            if (const std::uint32_t sep = separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q))))
                return q + std::countr_zero(sep) + 1;
    }

    static const char* skip_whitespace(const char* p) {
        while (static_cast<unsigned char>(*p) <= ' ') ++p;
        return p;
    }

    // Tokens in [p, end), where p starts a token.
    static std::size_t count_tokens(const char* p, const char* end) {
        std::size_t tokens = 1;
        for (const char* q = p + 1; q < end; ++q)
            tokens += static_cast<unsigned char>(*q) > ' ' && static_cast<unsigned char>(q[-1]) <= ' ';
        return tokens;
    }

    static const char* parse_slowly(const char* p, std::uint32_t* dst, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            p = skip_whitespace(p);
            const __m128i window = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const unsigned n = unsigned(std::countr_zero(separators(window) | 0x10000));
            dst[i] = std::uint32_t(parse16(window, n));
            p += n + 1;
        }
        return p;
    }
};

}  // namespace detail

// Token boundaries come from separator bitmasks of aligned 64-byte blocks: finding the next token
// clears one bit, so consecutive reads do not wait on each other's parsing.
class Reader {
public:
    // Takes everything that remains on fd (stdin by default).
    explicit Reader(int fd = 0) : input_(detail::open_input(fd)) { seek(input_.start); }
    ~Reader() { detail::release(input_); }

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    // Next integer token, or the next non-whitespace character for char.
    template <class T>
        requires std::integral<T> && (!std::same_as<T, bool>)
    T read() {
        if constexpr (std::same_as<T, char>) {
            skip_whitespace();
            return *cur_++;
        } else {
            const auto [start, length] = next_token();
            if constexpr (std::is_unsigned_v<T>) {
                return T(parse<sizeof(T) <= 4>(start, length));
            } else {
                const bool negative = *start == '-';
                const auto magnitude = parse<sizeof(T) <= 4>(start + negative, length - negative);
                return T(negative ? 0 - magnitude : magnitude);
            }
        }
    }

    // count tokens into dst. uint32_t uses the AVX2 bulk parser; other types read one by one.
    template <class T>
    void read(T* dst, std::size_t count) {
        if constexpr (std::same_as<T, std::uint32_t>) {
            skip_whitespace();
            const auto [end, parsed] = detail::BulkParser::parse(cur_, dst, count);
            seek(end);
            for (std::size_t i = parsed; i < count; ++i) dst[i] = read<T>();
        } else {
            for (std::size_t i = 0; i < count; ++i) dst[i] = read<T>();
        }
    }

    // Next token; valid while the Reader lives.
    std::string_view word() {
        const auto [start, length] = next_token();
        return {start, length};
    }

private:
    struct Token {
        const char* start;
        unsigned length;
    };

    // Continue reading at p.
    void seek(const char* p) {
        cur_ = p;
        block_ = reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t(63));
        separators_ = detail::block_separators(block_) & ~std::uint64_t(0) << (p - block_);
    }

    // Position of the next separator; at or after cur_.
    const char* next_separator() {
        while (!separators_) [[unlikely]] {
            block_ += 64;
            separators_ = detail::block_separators(block_);
        }
        return block_ + std::countr_zero(separators_);
    }

    void skip_whitespace() {
        while (next_separator() == cur_) {
            separators_ &= separators_ - 1;
            ++cur_;
        }
    }

    Token next_token() {
        for (;;) {
            const char* end = next_separator();
            separators_ &= separators_ - 1;
            const char* start = cur_;
            cur_ = end + 1;
            if (end != start) [[likely]] return {start, unsigned(end - start)};
        }
    }

    // Value of the token [s, s + n). Narrow: at most 10 digits; otherwise at most 20.
    template <bool Narrow>
    static auto parse(const char* s, unsigned n) {
        if constexpr (Narrow)
            return std::uint32_t(detail::parse16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s)), n));
        else
            return detail::parse20(s, n);
    }

    detail::Input input_;
    const char* cur_;             // next unread byte
    const char* block_;           // 64-byte block holding the next separator
    std::uint64_t separators_;    // separators in block_ at or after cur_
};

// Buffered output. write() takes any mix of integers, chars and strings:
//   out.write(x, ' ', y, '\n');
class Writer {
public:
    explicit Writer(int fd = 1) : fd_(fd) {}
    ~Writer() { flush(); }

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    template <class... Ts>
    void write(const Ts&... values) {
        const std::size_t needed = (std::size_t(0) + ... + max_size(values));
        if (needed > kCapacity) [[unlikely]] {
            (write_alone(values), ...);
            return;
        }
        if (std::size_t(buffer_ + kCapacity - cur_) < needed) [[unlikely]] flush();
        char* p = cur_;
        ((p = put(p, values)), ...);
        cur_ = p;
    }

    // values[0..count) joined by separator. uint32_t uses the AVX2 formatter.
    template <class T>
    void write_array(const T* values, std::size_t count, char separator) {
        if (!count) return;
        std::size_t i = 0;
        if constexpr (std::same_as<T, std::uint32_t>) {
            constexpr std::size_t kBatch = 8 * 11 + 16;  // eight values and separators, plus overhang
            char* p = cur_;
            for (; i + 8 <= count; i += 8) {
                if (std::size_t(buffer_ + kCapacity - p) < kBatch) [[unlikely]] {
                    cur_ = p;
                    flush();
                    p = cur_;
                }
                p = detail::format8(p, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + i)), separator);
            }
            cur_ = p;
        }
        for (; i < count; ++i) write(values[i], separator);
        --cur_;  // no separator after the last value
    }

    void flush() {
        write_all(fd_, buffer_, std::size_t(cur_ - buffer_));
        cur_ = buffer_;
    }

private:
    static constexpr std::size_t kCapacity = std::size_t(1) << 16;

    template <class T>
    static constexpr bool kInteger = std::integral<T> && !std::same_as<T, char> && !std::same_as<T, bool>;

    // Bytes put() may store, which can exceed what it writes.
    template <class T>
    static std::size_t max_size(const T& value) {
        if constexpr (std::same_as<T, char>) return 1;
        else if constexpr (kInteger<T>) return 1 + 20;  // a sign, then format() stores up to 20 bytes
        else return std::string_view(value).size();
    }

    template <class T>
    static char* put(char* p, const T& value) {
        if constexpr (std::same_as<T, char>) {
            *p = value;
            return p + 1;
        } else if constexpr (kInteger<T>) {
            using U = std::conditional_t<sizeof(T) <= 4, std::uint32_t, std::uint64_t>;
            if constexpr (std::is_signed_v<T>) {
                *p = '-';
                p += value < 0;
                return detail::format(p, value < 0 ? U(0 - U(value)) : U(value));
            } else {
                return detail::format(p, U(value));
            }
        } else {
            const std::string_view s(value);
            std::memcpy(p, s.data(), s.size());
            return p + s.size();
        }
    }

    // A value longer than the buffer: flush, then write it directly.
    template <class T>
    void write_alone(const T& value) {
        if constexpr (std::same_as<T, char> || kInteger<T>) {
            write(value);
        } else {
            flush();
            const std::string_view s(value);
            write_all(fd_, s.data(), s.size());
        }
    }

    static void write_all(int fd, const char* data, std::size_t size) {
        while (size) {
            const ssize_t put = ::write(fd, data, size);
            if (put > 0) data += put, size -= std::size_t(put);
            else if (put < 0 && errno == EINTR) continue;
            else std::abort();
        }
    }

    int fd_;
    char* cur_ = buffer_;
    alignas(64) char buffer_[kCapacity];
};

}  // namespace io
