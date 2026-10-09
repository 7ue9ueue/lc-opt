// Fixed-width output of values < 10^9 (convolution_mod's fields.hpp): each value right-aligned
// in 9 characters, then a space; the last separator is a newline. Judge-specific: the checker
// compares tokens, so the padding is accepted.
//
// 16 values per step, written as ten 16-byte chunks. The divisions of the next step are issued
// before the digits of this one, so the two dependency chains overlap.
#pragma once

#include "lib/io/io.hpp"

namespace fields10 {

namespace detail {

struct alignas(32) Lanes {
    std::uint32_t lane[8];
};

constexpr Lanes all(std::uint32_t x) { return {{x, x, x, x, x, x, x, x}}; }
constexpr Lanes all16(std::uint32_t x) { return all(x * 0x10001); }

constexpr int Z = 0x80;  // shuffle control: zero byte

// The same 16-byte shuffle control in both lanes.
constexpr Lanes shuffle(const int (&b)[16]) {
    Lanes t{};
    for (int i = 0; i < 32; ++i) t.lane[i / 4] |= std::uint32_t(b[i % 16] & 0xFF) << 8 * (i % 4);
    return t;
}

struct Constants {
    Lanes div10 = all(429496730);     // ceil(2^32 / 10): v / 10 = v * div10 >> 32 for v < 2^30
    Lanes div10e5 = all(1407374884);  // ceil(2^47 / 10^5): v / 10^5 = v * div10e5 >> 47
    Lanes pack = all(1 - 10000 * 65536u);  // [h, w - 10^4 h] in 16-bit halves = 2^16 w + h pack
    Lanes ten = all(10);
    Lanes tail = all(' ' << 8 | '0');
    Lanes by100 = all16(5243), hundred = all16(100);  // x / 100 = (x * 5243 >> 16) >> 3 for x < 10^4
    Lanes by10 = all16(6554);  // z / 10 = z * 6554 >> 16 for z < 100
    Lanes tens_out = all16(2559);  // bytes [z / 10, z % 10] = 256 z - 2559 (z / 10)
    Lanes ones = all(~0u), zero = all(0x30303030), space = all(0x20202020);
    // Chunk c of a step: text bytes [16c, 16c + 16) of values 0-7 in the low lane, of values 8-15
    // in the high lane. Sources, each lane: digits of two values (8 bytes each), or four tails
    // (2 bytes in each dword). Comments: field bytes of the low lane.
    Lanes chunk0[2] = {shuffle({0, 1, 2, 3, 4, 5, 6, 7, Z, Z, 8, 9, 10, 11, 12, 13}),  // D0 D1
                       shuffle({Z, Z, Z, Z, Z, Z, Z, Z, 0, 1, Z, Z, Z, Z, Z, Z})};     // T0
    Lanes chunk1[3] = {shuffle({14, 15, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z}),    // D1
                       shuffle({Z, Z, 4, 5, Z, Z, Z, Z, Z, Z, Z, Z, 8, 9, Z, Z}),      // T1 T2
                       shuffle({Z, Z, Z, Z, 0, 1, 2, 3, 4, 5, 6, 7, Z, Z, 8, 9})};     // D2 D3
    Lanes chunk2[3] = {shuffle({10, 11, 12, 13, 14, 15, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z}),  // D3
                       shuffle({Z, Z, Z, Z, Z, Z, 12, 13, Z, Z, Z, Z, Z, Z, Z, Z}),      // T3
                       shuffle({Z, Z, Z, Z, Z, Z, Z, Z, 0, 1, 2, 3, 4, 5, 6, 7})};       // D4
    Lanes chunk3[3] = {shuffle({0, 1, Z, Z, Z, Z, Z, Z, Z, Z, 4, 5, Z, Z, Z, Z}),        // T4 T5
                       shuffle({Z, Z, 8, 9, 10, 11, 12, 13, 14, 15, Z, Z, Z, Z, Z, Z}),  // D5
                       shuffle({Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 0, 1, 2, 3})};       // D6
    Lanes chunk4[2] = {shuffle({4, 5, 6, 7, Z, Z, 8, 9, 10, 11, 12, 13, 14, 15, Z, Z}),  // D6 D7
                       shuffle({Z, Z, Z, Z, 8, 9, Z, Z, Z, Z, Z, Z, Z, Z, 12, 13})};     // T6 T7
};

inline constexpr Constants kConstants{};

[[gnu::always_inline]] inline __m256i load(const Lanes& l) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(&l));
}

[[gnu::always_inline]] inline __m256i shuffle(__m256i x, const Lanes& control) {
    return _mm256_shuffle_epi8(x, load(control));
}

// Eight values v < 10^9 (dwords) after the divisions by 10 and 10^5: halves holds
// [w / 10^4, w % 10^4] in 16-bit halves for w = v / 10; tail holds the units digit and the
// separator as text.
struct Divided {
    __m256i halves, tail;
};

[[gnu::always_inline]] inline Divided divide(__m256i v, __m256i odd, const Constants& k) {
    const __m256i w = _mm256_blend_epi32(_mm256_srli_epi64(_mm256_mul_epu32(v, load(k.div10)), 32),
                                         _mm256_mul_epu32(odd, load(k.div10)), 0xAA);
    const __m256i h = _mm256_blend_epi32(_mm256_srli_epi64(_mm256_mul_epu32(v, load(k.div10e5)), 47),
                                         _mm256_srli_epi64(_mm256_mul_epu32(odd, load(k.div10e5)), 15), 0xAA);
    const __m256i halves = _mm256_add_epi32(_mm256_slli_epi32(w, 16), _mm256_mullo_epi32(h, load(k.pack)));
    const __m256i tail = _mm256_sub_epi32(_mm256_add_epi32(v, load(k.tail)), _mm256_mullo_epi32(w, load(k.ten)));
    return {halves, tail};
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

// 160 bytes at p (16-byte aligned) for values x0..x15, divided as lo = [x0..x3 | x8..x11] and
// hi = [x4..x7 | x12..x15].
[[gnu::always_inline]] inline void store(char* p, const Divided& lo, const Divided& hi, const Constants& k) {
    __m256i d01, d23, d45, d67;  // values x0 x1 | x8 x9, x2 x3 | x10 x11, and so on
    const auto either = [](__m256i a, __m256i b) { return _mm256_or_si256(a, b); };
    digits(lo, d01, d23, k);
    digits(hi, d45, d67, k);
    const __m256i c0 = either(shuffle(d01, k.chunk0[0]), shuffle(lo.tail, k.chunk0[1]));
    const __m256i c1 = either(either(shuffle(d01, k.chunk1[0]), shuffle(lo.tail, k.chunk1[1])), shuffle(d23, k.chunk1[2]));
    const __m256i c2 = either(either(shuffle(d23, k.chunk2[0]), shuffle(lo.tail, k.chunk2[1])), shuffle(d45, k.chunk2[2]));
    const __m256i c3 = either(either(shuffle(hi.tail, k.chunk3[0]), shuffle(d45, k.chunk3[1])), shuffle(d67, k.chunk3[2]));
    const __m256i c4 = either(shuffle(d67, k.chunk4[0]), shuffle(hi.tail, k.chunk4[1]));
    // Chunk c whole at p + 64 + 16c: its high lane lands in place, its low lane on the slot of the
    // previous chunk's high lane (or of chunk 4's low lane), which a later store fixes.
    const auto whole = [p](int c, __m256i chunk) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 64 + 16 * c), chunk); };
    const auto low = [p](int c, __m256i chunk) { _mm_store_si128(reinterpret_cast<__m128i*>(p + 16 * c), _mm256_castsi256_si128(chunk)); };
    whole(4, c4), whole(3, c3), whole(2, c2), whole(1, c1), whole(0, c0);
    low(0, c0), low(1, c1), low(2, c2), low(3, c3), low(4, c4);
}

// Values x[0, 16) in the lane order of store(), divided.
[[gnu::always_inline]] inline void divide16(const std::uint32_t* x, Divided& lo, Divided& hi, const Constants& k) {
    const auto at = [x](int i) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i)); };
    const auto both = [x](int i) { return _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i))); };
    const __m256i l = _mm256_blend_epi32(at(0), both(8), 0xF0), h = _mm256_blend_epi32(at(8), both(4), 0x0F);
    lo = divide(l, _mm256_srli_epi64(l, 32), k);
    hi = divide(h, _mm256_srli_epi64(h, 32), k);
}

// Text of x[0, count), count a positive multiple of 16, at p (16-byte aligned).
inline void format(char* p, const std::uint32_t* x, std::size_t count, const Constants& k) {
    Divided dl, dh;
    divide16(x, dl, dh, k);
    for (std::size_t i = 16; i < count; i += 16, p += 160) {
        Divided nl, nh;
        divide16(x + i, nl, nh, k);  // before the stores: 12% faster
        store(p, dl, dh, k);
        dl = nl, dh = nh;
    }
    store(p, dl, dh, k);
}

}  // namespace detail

inline constexpr std::size_t kBlock = 25600;  // values per write(2), a multiple of 16
inline constexpr std::size_t kTextBytes = 10 * kBlock;

// values[0, count), 1 <= count <= kBlock, as fixed-width fields; the last separator is a newline
// if last. text: kTextBytes bytes, 16-byte aligned. Longer than the Writer's buffer, the text
// goes to write(2) directly.
inline void write(io::Writer& out, const std::uint32_t* values, std::size_t count, char* text, bool last) {
    const detail::Constants& k = detail::kConstants;
    const std::size_t full = count / 16 * 16;
    if (full) detail::format(text, values, full, k);
    if (full < count) {
        alignas(32) std::uint32_t rest[16] = {};
        alignas(16) char tail[160];
        std::memcpy(rest, values + full, (count - full) * sizeof(std::uint32_t));
        detail::format(tail, rest, 16, k);
        std::memcpy(text + 10 * full, tail, 10 * (count - full));
    }
    if (last) text[10 * count - 1] = '\n';
    out.write(std::string_view(text, 10 * count));
}

}  // namespace fields10
