// Bulk read of uint32 arrays whose tokens come in long runs of one length (fixed-width or sorted
// inputs), on top of io::Reader:
//
//   runs::read(in, a, n);  // same values as io::read_bulk(in, a, n)
//
// Within a run of tokens of length L, each followed by one separator, token i starts at i (L + 1)
// bytes: 8 tokens at a time are checked against a 96-bit separator pattern and parsed without a
// separator search, so steps do not wait on each other. Elsewhere tokens go one at a time; once
// such misses pass 64 plus 1/64 of the tokens read (short runs), the rest goes to io::read_bulk.
// Same scheme as ../min_plus_convolution_convex_convex (its read_values).
#pragma once

#include <array>
#include <bit>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"

namespace runs {
namespace detail {

// Separator bits of 8 tokens of length s - 1, each followed by one separator: bits s - 1, 2s - 1,
// ..., 8s - 1 of a 96-bit mask, and the bits [0, 8s) they must match.
struct Stride {
    std::uint64_t low_mask, low_bits;
    std::uint32_t high_mask, high_bits;
};

inline constexpr auto kStrides = [] {
    std::array<Stride, 12> t{};
    for (unsigned s = 2; s <= 11; ++s) {
        unsigned __int128 mask = 0, bits = 0;
        for (unsigned i = 0; i < 8 * s; ++i) mask |= (unsigned __int128)1 << i;
        for (unsigned k = 1; k <= 8; ++k) bits |= (unsigned __int128)1 << (k * s - 1);
        t[s] = {std::uint64_t(mask), std::uint64_t(bits), std::uint32_t(mask >> 64), std::uint32_t(bits >> 64)};
    }
    return t;
}();

// Separator bits of the 96 bytes at p.
struct Separators {
    std::uint64_t low;
    std::uint32_t high;
};

inline Separators separators96(const char* p) {
    const auto at = [p](int i) {
        return io::detail::separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)));
    };
    return {at(0) | std::uint64_t(at(32)) << 32, at(64)};
}

inline bool matches(Separators x, const Stride& t) {
    return (x.low & t.low_mask) == t.low_bits && (x.high & t.high_mask) == t.high_bits;
}

// Digit groups (lib/io) of the tokens of length s - 1 at low and high, in the two lanes.
inline __m256i two_tokens(const char* low, const char* high, __m256i row) {
    const __m256i window =
        _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(high), reinterpret_cast<const __m128i*>(low));
    return io::detail::digit_groups(_mm256_shuffle_epi8(_mm256_subs_epu8(window, _mm256_set1_epi8('0')), row));
}

// Values of the 8 tokens at p, of length s - 1 each, one separator apart.
inline __m256i eight_tokens(const char* p, std::size_t s, __m256i row) {
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

// The token at or after p (after whitespace); p moves past its separator.
inline std::uint32_t one_token(const char*& p) {
    while (static_cast<unsigned char>(*p) <= ' ') ++p;
    const __m128i window = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const unsigned length = unsigned(std::countr_zero(io::detail::separators(window) | 0x10000));
    p += length + 1;
    return std::uint32_t(io::detail::parse16(window, length));
}

}  // namespace detail

// count values into dst: the same values as io::read_bulk(in, dst, count). Tokens have at most
// 10 digits.
inline void read(io::Reader& in, std::uint32_t* dst, std::size_t count) {
    using namespace detail;
    const char* p = in.scan().cur;
    std::size_t i = 0, misses = 0;
    // 64 tokens left span >= 127 bytes, so loads (< 96 bytes from p) stay in the input.
    while (i + 64 <= count && misses <= 64 + i / 64) {
        while (static_cast<unsigned char>(*p) <= ' ') ++p;
        const Separators first = separators96(p);
        const std::size_t s = std::size_t(std::countr_zero(first.low)) + 1;  // token length + 1
        if (s >= 2 && s <= 11 && matches(first, kStrides[s])) {
            const __m256i row = _mm256_broadcastsi128_si256(io::detail::align_row(s));
            do {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), eight_tokens(p, s, row));
                p += 8 * s, i += 8;
            } while (i + 64 <= count && matches(separators96(p), kStrides[s]));
        } else {
            ++misses;
        }
        if (i + 64 <= count) dst[i++] = one_token(p);
    }
    // Continue the Reader at p, as io::read_fixed does.
    const auto* block = reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t(63));
    in.resume({p, block, io::detail::block_separators(block) & ~std::uint64_t(0) << (p - block)});
    if (i < count) io::read_bulk(in, dst + i, count - i);
}

}  // namespace runs
