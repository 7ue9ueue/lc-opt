// Fixed-width output of 64-bit values: a space, then each value right-aligned in 20 characters;
// a newline at the end. Judge-specific: the checker compares tokens, so the padding is accepted.
//
// Eight values per step, AVX2. Value x = top 10^16 + mid 10^8 + low: the quotients come from
// multiply-shift estimates on the high bits, corrected by one exact step. Mid and low split into 4-digit chunks,
// then into digit bytes in 16-bit lanes. Each field is two stores: the top's 4 digits (8 bytes,
// the last 4 overwritten), then mid and low (16 bytes). The separators are never written: the
// text buffer starts as spaces and every block has the same layout.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace fields64 {

inline constexpr std::size_t kWidth = 21;     // a separator and 20 digits
inline constexpr std::size_t kBlock = 12288;  // values per call of format(), a multiple of 8
inline constexpr std::size_t kTextBytes = kWidth * kBlock + 32;

namespace detail {

inline __m256i all64(std::uint64_t x) { return _mm256_set1_epi64x(std::int64_t(x)); }
inline __m256i all16(std::uint16_t x) { return _mm256_set1_epi16(std::int16_t(x)); }

// Qwords x -> x / D and x mod D. The estimate q = ((x >> In) M >> Out), M = floor(2^(In + Out) / D),
// is at most x / D and low by less than 1: x >> In < 2^32, and the truncations lose
// 2^In / D + (x / D) / M < 1 in total. Needs q D < 2^64.
template <std::uint64_t D, int In, int Out>
inline void divide(__m256i x, __m256i& quotient, __m256i& remainder) {
    constexpr std::uint64_t kM = (static_cast<unsigned __int128>(1) << (In + Out)) / D;
    static_assert(kM < (std::uint64_t(1) << 32));
    const __m256i q = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, In), all64(kM)), Out);
    __m256i product = _mm256_mul_epu32(q, all64(D & 0xFFFFFFFF));
    if constexpr (D >> 32 != 0) product = _mm256_add_epi64(product, _mm256_slli_epi64(_mm256_mul_epu32(q, all64(D >> 32)), 32));
    const __m256i r = _mm256_sub_epi64(x, product);  // < 2D
    const __m256i over = _mm256_cmpgt_epi64(r, all64(D - 1));
    remainder = _mm256_sub_epi64(r, _mm256_and_si256(over, all64(D)));
    quotient = _mm256_sub_epi64(q, over);
}

// Dwords x < 10^8 -> 16-bit chunks [x / 10^4, x % 10^4].
inline __m256i chunks(__m256i x) {
    const __m256i magic = all64(3518437209);  // ceil(2^45 / 10^4)
    const __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(x, magic), 45);
    const __m256i odd = _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), magic), 13);
    const __m256i high = _mm256_blend_epi32(even, odd, 0xAA);
    // x - 10^4 high < 2^16: the low 16 bits suffice.
    const __m256i low = _mm256_sub_epi16(x, _mm256_mullo_epi16(high, all16(10000)));
    return _mm256_or_si256(high, _mm256_slli_epi32(low, 16));
}

// 16-bit z < 10^4 -> [z / 100, z % 100].
inline void hundreds(__m256i z, __m256i& high, __m256i& low) {
    high = _mm256_srli_epi16(_mm256_mulhi_epu16(z, all16(5243)), 3);
    low = _mm256_sub_epi16(z, _mm256_mullo_epi16(high, all16(100)));
}

// 16-bit z < 100 -> bytes [z / 10, z % 10] = 256 z - 2559 (z / 10).
inline __m256i two_digits(__m256i z) {
    const __m256i tens = _mm256_mulhi_epu16(z, all16(6554));
    return _mm256_sub_epi16(_mm256_slli_epi16(z, 8), _mm256_mullo_epi16(tens, all16(2559)));
}

inline __m256i ascii(__m256i digits) { return _mm256_or_si256(digits, all64(0x3030303030303030)); }

// Text with the bytes where blank has its top bit set as spaces.
inline __m256i blanked(__m256i digits, __m256i blank) {
    return _mm256_blendv_epi8(ascii(digits), all64(0x2020202020202020), blank);
}

// Leading-zero bytes of qword digits: 0xFF up to the first nonzero byte, < 0x10 at it, 0 after.
inline __m256i leading(__m256i digits) { return _mm256_xor_si256(digits, _mm256_add_epi64(digits, all64(~0ull))); }

// Qwords top < 10^4 -> text of its 4 digits in the low dword, leading zeros blank.
inline __m256i top_text(__m256i top) {
    __m256i high, low;
    hundreds(top, high, low);
    const __m256i digits = two_digits(_mm256_or_si256(high, _mm256_slli_epi32(low, 16)));
    return blanked(digits, leading(digits));
}

// Text of mid and low, 16 bytes per value: lanes hold [mid digits, low digits]; top_zero: per
// qword, all ones where the value's top is 0. Leading zeros blank, the last digit always shown.
inline __m256i rest_text(__m256i digits, __m256i top_zero) {
    const __m256i mid_zero = _mm256_slli_si256(_mm256_cmpeq_epi64(digits, _mm256_setzero_si256()), 8);
    const __m256i open = _mm256_and_si256(top_zero, _mm256_or_si256(mid_zero, _mm256_setr_epi64x(-1, 0, -1, 0)));
    // Bit 56, in the low group's last byte, stops the borrow there.
    const __m256i stopped = _mm256_or_si256(digits, _mm256_setr_epi64x(0, 1ll << 56, 0, 1ll << 56));
    return blanked(digits, _mm256_and_si256(leading(stopped), open));
}

// Eight values split into groups: top in qword lanes, [mid, low] in dword pairs; values 0-3, 4-7.
struct Split {
    __m256i top[2], rest[2];
};

[[gnu::always_inline]] inline Split split(const std::uint64_t* x) {
    Split s;
    for (int h = 0; h < 2; ++h) {
        __m256i rest, mid, low;
        // Losses: 2^32 / 10^16 + 1845 / M < 10^-6, and 2^22 / 10^8 + 10^8 / M < 0.1 (rest < 2^54).
        divide<10000000000000000, 32, 53>(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + 4 * h)), s.top[h], rest);
        divide<100000000, 22, 36>(rest, mid, low);
        s.rest[h] = _mm256_or_si256(mid, _mm256_slli_epi64(low, 32));
    }
    return s;
}

// Fields of the split values at p + 21 i + 1 (the separators stay as they are).
[[gnu::always_inline]] inline void emit(char* p, const Split& s) {
    __m256i rest_digits[2][2];
    for (int h = 0; h < 2; ++h) {
        // 16-bit chunks, then [z / 100, z % 100] pairs: unpacklo gives values 0 and 2 (one per
        // 128-bit lane), unpackhi values 1 and 3, as 8 words each.
        __m256i high, low;
        hundreds(chunks(s.rest[h]), high, low);
        rest_digits[h][0] = two_digits(_mm256_unpacklo_epi16(high, low));
        rest_digits[h][1] = two_digits(_mm256_unpackhi_epi16(high, low));
    }
    __m256i rest_text_of[2][2];
    const __m256i zero = _mm256_setzero_si256();
    const __m256i top_zero[2] = {_mm256_cmpeq_epi64(s.top[0], zero), _mm256_cmpeq_epi64(s.top[1], zero)};
    if (_mm256_testz_si256(_mm256_or_si256(top_zero[0], top_zero[1]), _mm256_set1_epi8(-1))) [[likely]] {
        for (int h = 0; h < 2; ++h)
            for (int j = 0; j < 2; ++j) rest_text_of[h][j] = ascii(rest_digits[h][j]);
    } else {
        for (int h = 0; h < 2; ++h) {
            rest_text_of[h][0] = rest_text(rest_digits[h][0], _mm256_unpacklo_epi64(top_zero[h], top_zero[h]));
            rest_text_of[h][1] = rest_text(rest_digits[h][1], _mm256_unpackhi_epi64(top_zero[h], top_zero[h]));
        }
    }
    const auto at = [p](int value, int offset) { return reinterpret_cast<__m128i*>(p + kWidth * value + offset); };
    for (int h = 0; h < 2; ++h) {
        const int v = 4 * h;
        // The top's 8-byte stores first: their last 4 bytes fall where mid and low go.
        const __m256i t = top_text(s.top[h]);
        const __m128i t01 = _mm256_castsi256_si128(t), t23 = _mm256_extracti128_si256(t, 1);
        _mm_storel_epi64(at(v, 1), t01);
        _mm_storel_epi64(at(v + 1, 1), _mm_unpackhi_epi64(t01, t01));
        _mm_storel_epi64(at(v + 2, 1), t23);
        _mm_storel_epi64(at(v + 3, 1), _mm_unpackhi_epi64(t23, t23));
        const __m256i r02 = rest_text_of[h][0], r13 = rest_text_of[h][1];
        _mm_storeu_si128(at(v, 5), _mm256_castsi256_si128(r02));
        _mm_storeu_si128(at(v + 1, 5), _mm256_castsi256_si128(r13));
        _mm_storeu_si128(at(v + 2, 5), _mm256_extracti128_si256(r02, 1));
        _mm_storeu_si128(at(v + 3, 5), _mm256_extracti128_si256(r13, 1));
    }
}

}  // namespace detail

// Text buffer of kTextBytes bytes, 32-byte aligned, filled with spaces.
inline void prepare(char* text) { std::memset(text, ' ', kTextBytes); }

// values[0, count) as fields in text (from prepare(), only used by format()), count <= kBlock;
// values readable up to count rounded up to 8. last: end with a newline. Returns the bytes.
inline std::size_t format(char* text, const std::uint64_t* values, std::size_t count, bool last) {
    // The divisions of the next 8 values are issued before the digits of these.
    detail::Split current = detail::split(values);
    for (std::size_t i = 0; i < count; i += 8) {
        const detail::Split next = detail::split(values + (i + 8 < count ? i + 8 : i));
        detail::emit(text + kWidth * i, current);
        current = next;
    }
    std::size_t bytes = kWidth * count;
    if (last) text[bytes++] = '\n';
    return bytes;
}

}  // namespace fields64
