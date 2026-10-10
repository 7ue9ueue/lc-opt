// c[k] = min over i + j = k of a[i] + b[j], a and b convex: c[0] = a[0] + b[0], and c's slopes are
// the slopes of a and b merged in ascending order, so c[k] is c[0] plus the k smallest slopes.
// Output blocks of columns::kBlock values are computed and written in turn. In a block, kChains
// independent chains each cover a range of k, interleaved so their latencies overlap. A chain
// starts from an argmin (i, k - i) of c[k], found by binary search, and then moves eight values
// at a time: the merge from an argmin passes through an argmin of each later c[k + p], so
// c[k + 1..k + 8] are minima over a window of 9 splits next to (i, k - i) (AVX2 add and min).
#include <array>
#include <bit>
#include <climits>

#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "lib/run/early.hpp"
#include "columns.hpp"

namespace {

using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t kChains = 4;
// Value of a and b outside their ends: above any value, and two of them add up without wrapping.
// So window terms out of range never win, and real values (sums <= 2e9) compare as u32.
constexpr u32 kEnd = INT32_MAX;
constexpr std::size_t kPad = 96;   // elements kEnd after a and b; chains read < 64 past
constexpr std::size_t kFront = 8;  // elements kEnd before b; windows read 3 before

// Slopes are differences mod 2^32 read as i32: values are <= 1e9, so real slopes fit.
i32 slope(const u32* p) { return i32(p[1] - p[0]); }

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

// [c[k + 1], ..., c[k + 8]] for an argmin (x - a, y - b) of c[k]: lane p - 1 is the minimum of
// x[q] + y[p - q] over q in [0, 8]. Each term is a[i'] + b[k + p - i'] for some i', or holds
// kEnd; the merge from (x, y) passes through an argmin of c[k + p] with q <= p. Terms with q in
// [5, 8] matter only for p >= 5, the high half.
__m256i minima(const u32* x, const u32* y) {
    const auto low = [x, y](int q) {
        return _mm256_add_epi32(_mm256_set1_epi32(i32(x[q])), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + 1 - q)));
    };
    const auto high = [x, y](int q) {
        return _mm_add_epi32(_mm_set1_epi32(i32(x[q])), _mm_loadu_si128(reinterpret_cast<const __m128i*>(y + 5 - q)));
    };
    const auto min = [](__m256i u, __m256i v) { return _mm256_min_epu32(u, v); };
    const __m128i high_min = _mm_min_epu32(_mm_min_epu32(high(5), high(6)), _mm_min_epu32(high(7), high(8)));
    const __m256i low_min = min(min(min(low(0), low(1)), min(low(2), low(3))), low(4));
    return min(low_min, _mm256_inserti128_si256(_mm256_set1_epi32(-1), high_min, 1));
}

// c[0, size) = values k0 + [0, size) for 1 <= size <= columns::kBlock, k0 + size <= n + m - 1;
// c[size, size + 8 * kChains + 1) receives garbage. Chain s writes c[t + 1, t + length + 1) from
// its start k0 + t, t = s * length (clamped to the block). Each step moves the chain from an
// argmin (x, y) of c[k] to one of c[k + 8]: (x + q, y + 8 - q) for the first q in [0, 8) with
// x[q] + y[8 - q] = c[k + 8], else q = 8.
void block(const u32* a, std::size_t n, const u32* b, std::size_t m, std::size_t k0, std::size_t size, u32* c) {
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0), last = _mm256_set1_epi32(7);
    const std::size_t length = ((size - 1 + kChains - 1) / kChains + 7) / 8 * 8;
    const u32* pa[kChains];
    const u32* pb[kChains];
    for (std::size_t s = 0; s < kChains; ++s) {
        const Chain first = start(a, n, b, m, k0 + std::min(s * length, size - 1));
        pa[s] = first.a, pb[s] = first.b;
    }
    c[0] = *pa[0] + *pb[0];
    for (std::size_t step = 0; step < length; step += 8) {
#pragma GCC unroll 16
        for (std::size_t s = 0; s < kChains; ++s) {
            const u32* const x = pa[s];
            const u32* const y = pb[s];
            const __m256i values = minima(x, y);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + s * length + step + 1), values);
            const __m256i ahead = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + 1));
            const __m256i diagonal = _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)),
                                                      _mm256_permutevar8x32_epi32(ahead, reverse));
            const __m256i on_path = _mm256_cmpeq_epi32(diagonal, _mm256_permutevar8x32_epi32(values, last));
            const unsigned q = unsigned(std::countr_zero(unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(on_path))) | 0x100u));
            pa[s] = x + q, pb[s] = y + 8 - q;
        }
    }
}

// Separator bits of 8 tokens of length s - 1, each followed by one separator: bits s - 1, 2s - 1,
// ..., 8s - 1 of a 96-bit mask, and the bits [0, 8s) they must match.
struct Stride {
    std::uint64_t low_mask, low_bits;
    std::uint32_t high_mask, high_bits;
};

constexpr auto kStrides = [] {
    std::array<Stride, 12> t{};
    for (unsigned s = 2; s <= 11; ++s) {
        unsigned __int128 mask = 0, bits = 0;
        for (unsigned i = 0; i < 8 * s; ++i) mask |= (unsigned __int128)1 << i;
        for (unsigned k = 1; k <= 8; ++k) bits |= (unsigned __int128)1 << (k * s - 1);
        t[s] = {std::uint64_t(mask), std::uint64_t(bits), std::uint32_t(mask >> 64), std::uint32_t(bits >> 64)};
    }
    return t;
}();

// Digit groups (lib/io) of the tokens of length s - 1 at low and high, in the two lanes.
__m256i two_tokens(const char* low, const char* high, __m256i row) {
    const __m256i window = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(high), reinterpret_cast<const __m128i*>(low));
    return io::detail::digit_groups(_mm256_shuffle_epi8(_mm256_subs_epu8(window, _mm256_set1_epi8('0')), row));
}

// The token at or after p (after whitespace); p moves past its separator.
u32 one_token(const char*& p) {
    while (static_cast<unsigned char>(*p) <= ' ') ++p;
    const __m128i window = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const unsigned length = unsigned(std::countr_zero(io::detail::separators(window) | 0x10000));
    p += length + 1;
    return u32(io::detail::parse16(window, length));
}

// Separator bits of the 96 bytes at p.
struct Separators {
    std::uint64_t low;
    std::uint32_t high;
};

Separators separators96(const char* p) {
    const auto at = [p](int i) { return io::detail::separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i))); };
    return {at(0) | std::uint64_t(at(32)) << 32, at(64)};
}

bool matches(Separators x, const Stride& t) {
    return (x.low & t.low_mask) == t.low_bits && (x.high & t.high_mask) == t.high_bits;
}

// Values of the 8 tokens at p, of length s - 1 each, one separator apart.
__m256i eight_tokens(const char* p, std::size_t s, __m256i row) {
    // Group j holds tokens j and j + 4, so the values come out in order.
    const __m256i g0 = two_tokens(p, p + 4 * s, row), g1 = two_tokens(p + s, p + 5 * s, row);
    const __m256i g2 = two_tokens(p + 2 * s, p + 6 * s, row), g3 = two_tokens(p + 3 * s, p + 7 * s, row);
    const __m256i k = _mm256_set1_epi32(0x00012710);  // 8-digit halves: high group * 10^4 + low
    const __m256 h01 = _mm256_castsi256_ps(_mm256_madd_epi16(_mm256_packus_epi32(g0, g1), k));
    const __m256 h23 = _mm256_castsi256_ps(_mm256_madd_epi16(_mm256_packus_epi32(g2, g3), k));
    const __m256i upper = _mm256_castps_si256(_mm256_shuffle_ps(h01, h23, 0x88));
    const __m256i lower = _mm256_castps_si256(_mm256_shuffle_ps(h01, h23, 0xDD));
    return _mm256_add_epi32(_mm256_mullo_epi32(upper, _mm256_set1_epi32(100000000)), lower);
}

// count values from position into dst; position ends past the last one's separator. Tokens have
// at most 10 digits; the 64 bytes after the input read as zeros. Fast path: runs of tokens of one
// length, each followed by one separator (most tests: 9 digits throughout, or long runs of one
// length), 8 at a time. Within a run p advances by a constant, so steps do not wait on each other.
void read_values(const char*& position, u32* dst, std::size_t count) {
    const char* p = position;  // through the reference, GCC stored and reloaded p every step
    std::size_t i = 0;
    // 64 tokens left span >= 127 bytes, so loads (< 96 bytes from p) stay in the input.
    while (i + 64 <= count) {
        const Separators first = separators96(p);
        const std::size_t s = std::size_t(std::countr_zero(first.low)) + 1;  // token length + 1
        if (s >= 2 && s <= 11 && matches(first, kStrides[s])) {
            const __m256i row = _mm256_broadcastsi128_si256(io::detail::align_row(s));
            do {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), eight_tokens(p, s, row));
                p += 8 * s, i += 8;
            } while (i + 64 <= count && matches(separators96(p), kStrides[s]));
        }
        if (i + 64 <= count) dst[i++] = one_token(p);
    }
    for (; i < count; ++i) dst[i] = one_token(p);
    position = p;
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>(), m = in.read<u32>(), count = n + m - 1;
    constexpr std::size_t kValues = (columns::kBlock + 8 * kChains + 1 + 15) / 16 * 16;  // block, garbage
    const std::size_t text_words = columns::kTextBytes / sizeof(u32);
    u32* const memory = mem::huge<u32>(text_words + kValues + (n + kPad) + (kFront + m + kPad));
    char* const text = reinterpret_cast<char*>(memory);
    u32* const c = memory + text_words;
    u32* const a = c + kValues;
    u32* const b = a + n + kPad + kFront;
    const char* p = in.scan().cur;
    read_values(p, a, n);
    read_values(p, b, m);
    std::fill_n(a + n, kPad + kFront, kEnd);  // after a, then before b
    std::fill_n(b + m, kPad, kEnd);

    io::Writer out;
    for (std::size_t k0 = 0; k0 < count; k0 += columns::kBlock) {
        const std::size_t size = std::min(columns::kBlock, count - k0);
        block(a, n, b, m, k0, size, c);
        // c is convex, so the largest value of a block is at one of its ends.
        columns::write_block(out, c, size, std::max(c[0], c[size - 1]), k0 + size == count, text);
    }
}

}  // namespace

RUN_EARLY(solve)
