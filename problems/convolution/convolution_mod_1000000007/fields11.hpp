// Fixed-width output of residues < 1000000007: each value right-aligned in 10 characters, then a
// space; the last separator is a newline. Judge-specific: the checker compares tokens, so the
// padding is accepted.
//
// Per value v: w = v / 100 as 8 digits in a qword (leading zeros blank), and a tail dword with
// the last two digits and the separator. 16 values per step; each 256-bit lane holds 8 values,
// whose 88 bytes of text are built as six 16-byte chunks by pshufb. The divisions of the next
// step are issued before the digits of this one.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>

#include "lib/io/io.hpp"

namespace fields11 {

namespace detail {

struct alignas(32) Lanes {
    std::uint32_t lane[8];
};

constexpr Lanes all(std::uint32_t x) { return {{x, x, x, x, x, x, x, x}}; }
constexpr Lanes all16(std::uint32_t x) { return all(x * 0x10001); }

constexpr int kWidth = 11;               // bytes per value
constexpr int kLaneBytes = 8 * kWidth;   // text of one lane's 8 values
constexpr int kChunks = 6;               // 16-byte chunks per lane; the last holds 8 bytes
constexpr int kSources = 6;              // digit qword pairs d01 d23 d45 d67, tails of values 0-3 and 4-7

// Shuffle control of chunk c from source s, the same in both lanes; 0x80 (zero) where the byte
// comes from another source. Value j of a lane: digits in source j / 2 at byte 8 (j % 2), tail in
// source 4 + j / 4 at byte 4 (j % 4).
constexpr Lanes control(int c, int s) {
    Lanes t{};
    for (int i = 0; i < 32; ++i) {
        const int at = 16 * c + i % 16, j = at / kWidth, pos = at % kWidth;
        int byte = 0x80;
        if (at < kLaneBytes) {
            if (pos < 8 && s == j / 2) byte = 8 * (j % 2) + pos;
            if (pos >= 8 && s == 4 + j / 4) byte = 4 * (j % 4) + pos - 8;
        }
        t.lane[i / 4] |= std::uint32_t(byte) << 8 * (i % 4);
    }
    return t;
}

constexpr bool used(int c, int s) {
    const Lanes t = control(c, s);
    for (std::uint32_t w : t.lane)
        if (w != 0x80808080) return true;
    return false;
}

struct Constants {
    Lanes div100 = all(2748779070);    // ceil(2^38 / 100): v / 100 = v * div100 >> 38, any v < 2^32
    Lanes div10e6 = all(2251799814);   // ceil(2^51 / 10^6): v / 10^6 = v * div10e6 >> 51
    Lanes pack = all(1 - 10000 * 65536u);  // [h, w - 10^4 h] in 16-bit halves = 2^16 w + h pack
    Lanes hundred32 = all(100), ten = all(10), blank_tens = all(0x10);
    Lanes by103 = all16(103);              // z / 10 = z * 103 >> 10 for z < 100
    Lanes tail = all(' ' << 16 | '0' << 8 | '0');
    Lanes by100 = all16(5243), hundred = all16(100);  // x / 100 = (x * 5243 >> 16) >> 3 for x < 10^4
    Lanes by10 = all16(6554);  // z / 10 = z * 6554 >> 16 for z < 100
    Lanes tens_out = all16(2559);  // bytes [z / 10, z % 10] = 256 z - 2559 (z / 10)
    Lanes ones = all(~0u), zero = all(0x30303030), space = all(0x20202020);
    Lanes chunk[kChunks][kSources];

    constexpr Constants() {
        for (int c = 0; c < kChunks; ++c)
            for (int s = 0; s < kSources; ++s) chunk[c][s] = control(c, s);
    }
};

inline constexpr Constants kConstants{};

[[gnu::always_inline]] inline __m256i load(const Lanes& l) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(&l));
}

// Eight values v < 2^31 after the divisions: halves holds [w / 10^4, w % 10^4] in 16-bit halves
// for w = v / 100; tail holds the last two digits and the separator as text.
struct Divided {
    __m256i halves, tail;
};

[[gnu::always_inline]] inline Divided divide(__m256i v, const Constants& k) {
    const __m256i odd = _mm256_srli_epi64(v, 32);
    const __m256i w = _mm256_blend_epi32(_mm256_srli_epi64(_mm256_mul_epu32(v, load(k.div100)), 38),
                                         _mm256_srli_epi64(_mm256_mul_epu32(odd, load(k.div100)), 6), 0xAA);
    const __m256i h = _mm256_blend_epi32(_mm256_srli_epi64(_mm256_mul_epu32(v, load(k.div10e6)), 51),
                                         _mm256_srli_epi64(_mm256_mul_epu32(odd, load(k.div10e6)), 19), 0xAA);
    const __m256i halves = _mm256_add_epi32(_mm256_slli_epi32(w, 16), _mm256_mullo_epi32(h, load(k.pack)));
    const __m256i r = _mm256_sub_epi32(v, _mm256_mullo_epi32(w, load(k.hundred32)));  // < 100
    const __m256i tens = _mm256_srli_epi16(_mm256_mullo_epi16(r, load(k.by103)), 10);
    const __m256i last2 = _mm256_sub_epi16(_mm256_slli_epi32(r, 8), _mm256_mullo_epi16(tens, load(k.tens_out)));
    // v < 10: the tens digit is blank ('0' - 0x10 = ' ').
    const __m256i blank = _mm256_and_si256(_mm256_cmpgt_epi32(load(k.ten), v), load(k.blank_tens));
    return {halves, _mm256_sub_epi32(_mm256_add_epi32(last2, load(k.tail)), blank)};
}

// 16-bit z < 100 -> bytes [z / 10, z % 10].
[[gnu::always_inline]] inline __m256i two_digits(__m256i z, const Constants& k) {
    const __m256i tens = _mm256_mulhi_epu16(z, load(k.by10));
    return _mm256_sub_epi16(_mm256_slli_epi16(z, 8), _mm256_mullo_epi16(tens, load(k.tens_out)));
}

// Qwords of 8 digits, most significant first -> text; leading zeros become spaces. x ^ (x - 1)
// sets the bits up to the lowest set one: 0xFF in leading zero bytes, < 0x10 in the first
// nonzero digit, 0 above. blendv takes the space where a byte's top bit is set.
[[gnu::always_inline]] inline __m256i text(__m256i digits, const Constants& k) {
    const __m256i upto = _mm256_xor_si256(digits, _mm256_add_epi64(digits, load(k.ones)));
    return _mm256_blendv_epi8(_mm256_or_si256(digits, load(k.zero)), load(k.space), upto);
}

// Text of the 8 digits of w, two values per lane: values [0, 1 | 4, 5] of d in lo, [2, 3 | 6, 7]
// in hi. Per value, 16-bit [w / 10^6, w / 10^4 % 100, w / 100 % 100, w % 100] before two_digits.
[[gnu::always_inline]] inline void digits(const Divided& d, __m256i& lo, __m256i& hi, const Constants& k) {
    const __m256i hundreds = _mm256_srli_epi16(_mm256_mulhi_epu16(d.halves, load(k.by100)), 3);
    const __m256i rest = _mm256_sub_epi16(d.halves, _mm256_mullo_epi16(hundreds, load(k.hundred)));
    lo = text(two_digits(_mm256_unpacklo_epi16(hundreds, rest), k), k);
    hi = text(two_digits(_mm256_unpackhi_epi16(hundreds, rest), k), k);
}

template <int C>
[[gnu::always_inline]] inline __m256i chunk(const __m256i (&src)[kSources], const Constants& k) {
    __m256i r = _mm256_setzero_si256();
    [&]<int... S>(std::integer_sequence<int, S...>) {
        ((used(C, S) ? r = _mm256_or_si256(r, _mm256_shuffle_epi8(src[S], load(k.chunk[C][S]))) : r), ...);
    }(std::make_integer_sequence<int, kSources>{});
    return r;
}

// 176 bytes at p (16-byte aligned) for values x0..x15, divided as lo = [x0..x3 | x8..x11] and
// hi = [x4..x7 | x12..x15]; writes 8 bytes past.
[[gnu::always_inline]] inline void store(char* p, const Divided& lo, const Divided& hi, const Constants& k) {
    __m256i src[kSources];  // per lane: values 0 1, 2 3, 4 5, 6 7 (digits), 0-3, 4-7 (tails)
    digits(lo, src[0], src[1], k);
    digits(hi, src[2], src[3], k);
    src[4] = lo.tail, src[5] = hi.tail;
    const __m256i c[kChunks] = {chunk<0>(src, k), chunk<1>(src, k), chunk<2>(src, k),
                                chunk<3>(src, k), chunk<4>(src, k), chunk<5>(src, k)};
    // Chunk i whole at p + 72 + 16i: its high lane lands in place (lane 1 text starts at 88), its
    // low lane on bytes that the next lower chunk or a low-lane store fixes afterwards.
    for (int i = kChunks - 1; i >= 0; --i)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + kLaneBytes - 16 + 16 * i), c[i]);
    for (int i = 0; i < kChunks - 1; ++i)
        _mm_store_si128(reinterpret_cast<__m128i*>(p + 16 * i), _mm256_castsi256_si128(c[i]));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p + 16 * (kChunks - 1)), _mm256_castsi256_si128(c[kChunks - 1]));
}

// Values x[0, 16) in the lane order of store(), divided.
[[gnu::always_inline]] inline void divide16(const std::uint32_t* x, Divided& lo, Divided& hi, const Constants& k) {
    const auto at = [x](int i) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)); };
    const auto both = [x](int i) { return _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i))); };
    lo = divide(_mm256_blend_epi32(at(0), both(8), 0xF0), k);
    hi = divide(_mm256_blend_epi32(at(8), both(4), 0x0F), k);
}

// Text of x[0, count), count a positive multiple of 16, at p (16-byte aligned); writes 8 bytes past.
inline void format(char* p, const std::uint32_t* x, std::size_t count, const Constants& k) {
    Divided dl, dh;
    divide16(x, dl, dh, k);
    for (std::size_t i = 16; i < count; i += 16, p += 16 * kWidth) {
        Divided nl, nh;
        divide16(x + i, nl, nh, k);
        store(p, dl, dh, k);
        dl = nl, dh = nh;
    }
    store(p, dl, dh, k);
}

}  // namespace detail

inline constexpr std::size_t kBlock = 25600;  // values per write(2), a multiple of 16
inline constexpr std::size_t kTextBytes = detail::kWidth * kBlock + 16;

// values[0, count), count >= 1, as fixed-width fields; text: kTextBytes bytes, 16-byte aligned.
inline void write(io::Writer& out, const std::uint32_t* values, std::size_t count, char* text, bool last) {
    const detail::Constants& k = detail::kConstants;
    constexpr std::size_t w = detail::kWidth;
    const std::size_t full = count / 16 * 16;
    if (full) detail::format(text, values, full, k);
    if (full < count) {
        alignas(32) std::uint32_t rest[16] = {};
        alignas(16) char tail[16 * w + 16];
        std::memcpy(rest, values + full, (count - full) * sizeof(std::uint32_t));
        detail::format(tail, rest, 16, k);
        std::memcpy(text + w * full, tail, w * (count - full));
    }
    if (last) text[w * count - 1] = '\n';
    out.write(std::string_view(text, w * count));
}

}  // namespace fields11
