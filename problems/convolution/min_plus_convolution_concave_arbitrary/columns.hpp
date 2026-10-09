// Fixed-width output of values < 2^31: each value right-aligned in W - 1 characters, then a space;
// the last separator is a newline. W = 10 for blocks whose values are all below 10^9, else 11.
// Judge-specific: the checker compares tokens, so the padding is accepted.
//
// Per value v: w = v / 10^(W - 9) as 8 digits, most significant first, in a qword (leading zeros
// blank); the tail holds the last W - 9 digits and the separator in a dword. 16 values per step,
// written as W 16-byte chunks built by pshufb from these sources. The divisions of the next step
// are issued before the digits of this one, so the two dependency chains overlap.
// Same scheme as ../convolution_mod/fields.hpp (W = 10 only); here the shuffle controls are
// generated for either width.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include "lib/io/io.hpp"

namespace columns {

namespace detail {

using u32 = std::uint32_t;

struct alignas(32) Lanes {
    u32 lane[8];
};

constexpr Lanes all(u32 x) { return {{x, x, x, x, x, x, x, x}}; }
constexpr Lanes all16(u32 x) { return all(x * 0x10001); }

[[gnu::always_inline]] inline __m256i load(const Lanes& l) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(&l));
}

// Sources of a step's text. Digit qwords: kD01 holds values [0, 1 | 8, 9], kD23 [2, 3 | 10, 11],
// kD45 [4, 5 | 12, 13], kD67 [6, 7 | 14, 15]. Tail dwords: kLowTail [0..3 | 8..11],
// kHighTail [4..7 | 12..15]. kSwapped: kD01 with its lanes exchanged (W = 11 only, for the
// digits of value 8, which end the low lane's last chunk).
enum Source { kD01, kD23, kD45, kD67, kLowTail, kHighTail, kSwapped, kSources };

constexpr int kZero = 0x80;  // shuffle control: zero byte

// Chunks of the 16W text bytes of a step: chunks [0, low) from the low lane (values 0-7, and
// value 8's digits when W = 11), chunks [low, W) from the high lane. Vector k holds chunk k in
// its low lane and chunk low + k in its high lane.
template <int W>
struct Layout {
    static constexpr int kLow = (8 * W + 15) / 16;
    static constexpr int kHigh = W - kLow;
    Lanes control[kLow][kSources] = {};
    bool used[kLow][kSources] = {};

    // Text byte at offset 16 * chunk + i, read in lane `lane`.
    constexpr void place(int k, int lane, int chunk, int i) {
        const int at = 16 * chunk + i, value = at / W, offset = at % W, j = value % 8;
        int source, byte;
        if (value / 8 != lane) {  // value 8's digits in the low lane
            source = kSwapped, byte = offset;
        } else if (offset < 8) {
            source = kD01 + j / 2, byte = 8 * (j % 2) + offset;
        } else {
            source = kLowTail + j / 4, byte = 4 * (j % 4) + offset - 8;
        }
        used[k][source] = true;
        const int b = 16 * lane + i;
        u32& word = control[k][source].lane[b / 4];
        word = (word & ~(0xFFu << 8 * (b % 4))) | u32(byte) << 8 * (b % 4);
    }

    constexpr Layout() {
        for (int k = 0; k < kLow; ++k)
            for (int s = 0; s < kSources; ++s)
                for (int b = 0; b < 32; ++b) control[k][s].lane[b / 4] |= u32(kZero) << 8 * (b % 4);
        for (int k = 0; k < kLow; ++k)
            for (int i = 0; i < 16; ++i) {
                place(k, 0, k, i);
                if (k < kHigh) place(k, 1, kLow + k, i);
            }
    }
};

template <int W>
inline constexpr Layout<W> kLayout{};

// Divisions: v / 10^(W - 9) = v * kDigits.multiplier >> kDigits.shift, exact for v < 2^31,
// and likewise v / 10^(W - 4) by kHigher.
struct Magic {
    u32 multiplier;
    int shift;
};

template <int W>
struct Constants;

template <>
struct Constants<10> {
    static constexpr Magic kDigits{429496730, 32};    // ceil(2^32 / 10), v < 2^30
    static constexpr Magic kHigher{1407374884, 47};   // ceil(2^47 / 10^5)
};

template <>
struct Constants<11> {
    static constexpr Magic kDigits{2748779070, 38};   // ceil(2^38 / 100), v < 2^32
    static constexpr Magic kHigher{2251799814, 51};   // ceil(2^51 / 10^6), v < 7 * 10^9
};

struct Common {
    Lanes pack = all(1 - 10000 * 65536u);  // [h, w - 10^4 h] in 16-bit halves = 2^16 w + h pack
    Lanes by100 = all16(5243), hundred = all16(100);  // x / 100 = (x * 5243 >> 16) >> 3 for x < 10^4
    Lanes by10 = all16(6554);       // z / 10 = z * 6554 >> 16 for z < 100
    Lanes tens_out = all16(2559);   // bytes [z / 10, z % 10] = 256 z - 2559 (z / 10)
    Lanes ones = all(~0u), zero = all(0x30303030), space = all(0x20202020);
    Lanes ten = all(10), hundred32 = all(100);
    Lanes tail10 = all(' ' << 8 | '0');               // units digit, space
    Lanes tail11 = all(' ' << 16 | '0' << 8 | '0');   // tens digit, units digit, space
    Lanes blank = all(0x10);                          // '0' - 0x10 = ' '
    Lanes digits_multiplier[2] = {all(Constants<10>::kDigits.multiplier), all(Constants<11>::kDigits.multiplier)};
    Lanes higher_multiplier[2] = {all(Constants<10>::kHigher.multiplier), all(Constants<11>::kHigher.multiplier)};
};

inline constexpr Common kCommon{};

// v / magic for eight dwords v; odd = v >> 32 per qword.
template <Magic M>
[[gnu::always_inline]] inline __m256i divide(__m256i v, __m256i odd, __m256i multiplier) {
    const __m256i even_q = _mm256_srli_epi64(_mm256_mul_epu32(v, multiplier), M.shift);
    const __m256i odd_q = _mm256_mul_epu32(odd, multiplier);
    return _mm256_blend_epi32(even_q, M.shift == 32 ? odd_q : _mm256_srli_epi64(odd_q, M.shift - 32), 0xAA);
}

// Eight values after the divisions: halves holds [w / 10^4, w % 10^4] in 16-bit halves;
// tail holds the last W - 9 digits and the separator as text.
struct Divided {
    __m256i halves, tail;
};

template <int W>
[[gnu::always_inline]] inline Divided divide(__m256i v, const Common& k) {
    const __m256i odd = _mm256_srli_epi64(v, 32);
    const __m256i w = divide<Constants<W>::kDigits>(v, odd, load(k.digits_multiplier[W - 10]));
    const __m256i h = divide<Constants<W>::kHigher>(v, odd, load(k.higher_multiplier[W - 10]));
    const __m256i halves = _mm256_add_epi32(_mm256_slli_epi32(w, 16), _mm256_mullo_epi32(h, load(k.pack)));
    if constexpr (W == 10) {
        return {halves, _mm256_sub_epi32(_mm256_add_epi32(v, load(k.tail10)), _mm256_mullo_epi32(w, load(k.ten)))};
    } else {
        // r = v % 100 < 100 in the low 16 bits; tail bytes [r / 10, r % 10, ' ', 0] as text,
        // the tens digit blank for v < 10 (then w = 0 and the digits are blank too).
        const __m256i r = _mm256_sub_epi32(v, _mm256_mullo_epi32(w, load(k.hundred32)));
        const __m256i tens = _mm256_mulhi_epu16(r, load(k.by10));
        const __m256i pair = _mm256_sub_epi16(_mm256_slli_epi16(r, 8), _mm256_mullo_epi16(tens, load(k.tens_out)));
        const __m256i small = _mm256_and_si256(_mm256_cmpgt_epi32(load(k.ten), v), load(k.blank));
        return {halves, _mm256_sub_epi32(_mm256_add_epi32(pair, load(k.tail11)), small)};
    }
}

// 16-bit z < 100 -> bytes [z / 10, z % 10].
[[gnu::always_inline]] inline __m256i two_digits(__m256i z, const Common& k) {
    const __m256i tens = _mm256_mulhi_epu16(z, load(k.by10));
    return _mm256_sub_epi16(_mm256_slli_epi16(z, 8), _mm256_mullo_epi16(tens, load(k.tens_out)));
}

// Qwords of 8 digits, most significant first -> text; leading zeros become spaces. x ^ (x - 1)
// sets the bits up to the lowest set one: 0xFF in leading zero bytes, < 0x10 in the first
// nonzero digit, 0 above. blendv takes the space where a byte's top bit is set.
[[gnu::always_inline]] inline __m256i text(__m256i digits, const Common& k) {
    const __m256i upto = _mm256_xor_si256(digits, _mm256_add_epi64(digits, load(k.ones)));
    return _mm256_blendv_epi8(_mm256_or_si256(digits, load(k.zero)), load(k.space), upto);
}

// Text of the 8 digits of w, two values per lane: values [0, 1 | 4, 5] of d in lo, [2, 3 | 6, 7]
// in hi. Per value, 16-bit [w / 10^6, w / 10^4 % 100, w / 100 % 100, w % 100] before two_digits.
[[gnu::always_inline]] inline void digits(const Divided& d, __m256i& lo, __m256i& hi, const Common& k) {
    const __m256i hundreds = _mm256_srli_epi16(_mm256_mulhi_epu16(d.halves, load(k.by100)), 3);
    const __m256i rest = _mm256_sub_epi16(d.halves, _mm256_mullo_epi16(hundreds, load(k.hundred)));
    lo = text(two_digits(_mm256_unpacklo_epi16(hundreds, rest), k), k);
    hi = text(two_digits(_mm256_unpackhi_epi16(hundreds, rest), k), k);
}

// Vector k of the step: the OR of one shuffle per source it uses.
template <int W, int K, std::size_t... S>
[[gnu::always_inline]] inline __m256i chunk(const __m256i (&source)[kSources], std::index_sequence<S...>) {
    constexpr const Layout<W>& layout = kLayout<W>;
    __m256i c = _mm256_setzero_si256();
    const auto add = [&](std::size_t s) {
        c = _mm256_or_si256(c, _mm256_shuffle_epi8(source[s], load(layout.control[K][s])));
    };
    ((layout.used[K][S] ? add(S) : void()), ...);
    return c;
}

// 16W bytes at p (16-byte aligned) for values x0..x15, divided as lo = [x0..x3 | x8..x11] and
// hi = [x4..x7 | x12..x15].
template <int W>
[[gnu::always_inline]] inline void store(char* p, const Divided& lo, const Divided& hi, const Common& k) {
    constexpr int kLow = Layout<W>::kLow, kHigh = Layout<W>::kHigh;
    __m256i source[kSources];
    digits(lo, source[kD01], source[kD23], k);
    digits(hi, source[kD45], source[kD67], k);
    source[kLowTail] = lo.tail, source[kHighTail] = hi.tail;
    source[kSwapped] = W == 11 ? _mm256_permute2x128_si256(source[kD01], source[kD01], 0x01) : source[kD01];
    __m256i c[kLow];
    [&]<std::size_t... K>(std::index_sequence<K...>) {
        ((c[K] = chunk<W, K>(source, std::make_index_sequence<kSources>())), ...);
    }(std::make_integer_sequence<std::size_t, kLow>());
    // Vector k whole at chunk kLow + k - 1: its high lane lands in place, its low lane on the slot
    // of the previous vector's high lane (or of chunk kLow - 1), which a later store fixes.
    for (int i = kHigh - 1; i >= 0; --i)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 16 * (kLow + i - 1)), c[i]);
    for (int i = 0; i < kLow; ++i)
        _mm_store_si128(reinterpret_cast<__m128i*>(p + 16 * i), _mm256_castsi256_si128(c[i]));
}

// Values x[0, 16) in the lane order of store(), divided.
template <int W>
[[gnu::always_inline]] inline void divide16(const u32* x, Divided& lo, Divided& hi, const Common& k) {
    const auto at = [x](int i) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)); };
    const auto both = [x](int i) { return _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i))); };
    lo = divide<W>(_mm256_blend_epi32(at(0), both(8), 0xF0), k);
    hi = divide<W>(_mm256_blend_epi32(at(8), both(4), 0x0F), k);
}

// Text of x[0, count), count a positive multiple of 16, at p (16-byte aligned).
template <int W>
void format(char* p, const u32* x, std::size_t count) {
    const Common& k = kCommon;
    Divided dl, dh;
    divide16<W>(x, dl, dh, k);
    for (std::size_t i = 16; i < count; i += 16, p += 16 * W) {
        Divided nl, nh;
        divide16<W>(x + i, nl, nh, k);
        store<W>(p, dl, dh, k);
        dl = nl, dh = nh;
    }
    store<W>(p, dl, dh, k);
}

inline u32 maximum(const u32* x, std::size_t count) {
    __m256i m = _mm256_setzero_si256();
    for (std::size_t i = 0; i < count; i += 8)
        m = _mm256_max_epu32(m, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)));
    alignas(32) u32 lane[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane), m);
    return *std::max_element(lane, lane + 8);
}

}  // namespace detail

inline constexpr std::size_t kBlock = 25600;  // values per write(2), a multiple of 16
inline constexpr std::size_t kTextBytes = 11 * kBlock;

// values[0, count), count >= 1, each < 2^31; values[count, count rounded up to 16) must be
// readable and zero. text: kTextBytes bytes, 16-byte aligned. Blocks are longer than the
// Writer's buffer, so it hands each to write(2) directly.
inline void write(io::Writer& out, const std::uint32_t* values, std::size_t count, char* text) {
    for (std::size_t i = 0; i < count; i += kBlock) {
        const std::size_t n = std::min(kBlock, count - i), padded = (n + 15) / 16 * 16;
        const std::size_t width = detail::maximum(values + i, padded) < 1'000'000'000 ? 10 : 11;
        if (width == 10) detail::format<10>(text, values + i, padded);
        else detail::format<11>(text, values + i, padded);
        if (i + n == count) text[width * n - 1] = '\n';
        out.write(std::string_view(text, width * n));
    }
}

}  // namespace columns
