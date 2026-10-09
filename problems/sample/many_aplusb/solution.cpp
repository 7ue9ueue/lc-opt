// A + B in decimal: both numbers right-aligned in 32 bytes as digits 0..9, added bytewise, carries
// resolved with bitmasks. No binary values at all.
#include "lib/io/io.hpp"

#include <algorithm>

namespace {

// kKeep[n + 1..n + 33) keeps the last n + 1 bytes of 32.
alignas(64) constexpr std::uint8_t kKeep[64] = {
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// A token of at most 30 digits as byte values in bytes 30 - n..30; byte 31 (its separator) and the
// bytes before it become 0.
__m256i digits(std::string_view token) {
    const char* end = token.data() + token.size();  // the separator
    const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(end - 31));
    const __m256i keep = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kKeep + token.size() + 1));
    return _mm256_and_si256(_mm256_subs_epu8(bytes, _mm256_set1_epi8('0')), keep);
}

// Writes a + b and a newline at p; returns the end. Sums have at most 20 digits.
char* write_sum(char* p, __m256i a, __m256i b) {
    const __m256i sum = _mm256_add_epi8(a, b);  // bytes 0..18
    // Bit i of the masks is byte 31 - i, so carries move toward higher bits as in an addition.
    const __m256i reversed = _mm256_permute4x64_epi64(
        _mm256_shuffle_epi8(sum, _mm256_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                                  15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)),
        0x4E);
    const auto generate = std::uint32_t(_mm256_movemask_epi8(_mm256_cmpgt_epi8(reversed, _mm256_set1_epi8(9))));
    const auto propagate = std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(reversed, _mm256_set1_epi8(9))));
    const std::uint32_t carries = ((generate << 1) + propagate) ^ propagate;  // carry into each digit
    // Byte j takes bit 31 - j of carries.
    const __m256i spread = _mm256_shuffle_epi8(
        _mm256_set1_epi32(int(carries)), _mm256_setr_epi8(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
                                                          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0));
    const __m256i bit = _mm256_set1_epi64x(std::int64_t(0x0102040810204080));
    const __m256i carry = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit), bit);  // -1 where a carry enters
    __m256i digit = _mm256_sub_epi8(sum, carry);
    digit = _mm256_sub_epi8(digit, _mm256_and_si256(_mm256_cmpgt_epi8(digit, _mm256_set1_epi8(9)), _mm256_set1_epi8(10)));
    // The text is bytes first..31: the digits without leading zeros (at least the units at byte
    // 30), then the newline at byte 31.
    const auto nonzero = ~std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(digit, _mm256_setzero_si256())));
    const auto first = unsigned(std::countr_zero(nonzero | 1u << 30));  // 11 <= first <= 30
    const __m256i text = _mm256_add_epi8(digit, _mm256_setr_epi8('0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                                                  '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                                                  '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                                                  '0', '\n'));
    const unsigned head = first < 16 ? 16 - first : 0;  // bytes from the low half
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), io::detail::shift_down(_mm256_castsi256_si128(text), 16 - head));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p + head),
                     io::detail::shift_down(_mm256_extracti128_si256(text, 1), first + head - 16));
    return p + 32 - first;
}

}  // namespace

int main() {
    io::Reader in;
    io::Writer out;
    constexpr std::uint32_t kLines = 1024;  // per buffer reservation; a line takes at most 21 bytes
    for (auto t = in.read<std::uint32_t>(); t;) {
        const std::uint32_t lines = std::min(t, kLines);
        t -= lines;
        out.write_with(kLines * 21 + 16, [&](char* p) {
            for (std::uint32_t i = 0; i < lines; ++i) {
                const __m256i a = digits(in.word());
                const __m256i b = digits(in.word());
                p = write_sum(p, a, b);
            }
            return p;
        });
    }
}
