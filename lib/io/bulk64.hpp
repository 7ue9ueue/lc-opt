// Bulk read of uint64 arrays with AVX2, on top of io::Reader:
//
//   io::Reader in;
//   io::read_bulk(in, a.data(), n);  // same values as in.read(a.data(), n), faster
//
// Kept apart from io.hpp so that the code of the programs that do not need it stays unchanged.
// Design and measurements: lib/io/notes.md.
#pragma once

#include "lib/io/io.hpp"

namespace io {
namespace detail {

// The 32 bytes at kDigitMask[n] (n <= 64), subtracted with unsigned saturation from the 32 bytes
// that end at a token of length n, give its digit values and zero the bytes before it.
alignas(64) inline constexpr auto kDigitMask = [] {
    std::array<std::uint8_t, 96> t{};
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = i < 32 ? 0xFF : '0';
    return t;
}();

// Parses uint64 tokens of at most 20 digits from the token at p into dst: at most count, which
// must be the tokens that follow. Each chunk of input is cut into four streams at token
// boundaries; the streams advance in lockstep, two tokens per step. Stops when fewer than
// kMinTokens tokens remain. Returns where it stopped and the tokens parsed.
class BulkParser64 {
public:
    static constexpr std::size_t kMinTokens = 1024;

    struct Result {
        const char* end;
        std::size_t parsed;
    };

    static Result parse(const char* p, std::uint64_t* dst, std::size_t count) {
        std::uint64_t* const first = dst;
        while (count >= kMinTokens) {
            // The count tokens span at least 2 * count - 1 bytes. A chunk spans at most chunk + 21
            // bytes, so it holds at most chunk / 2 + 11 < count tokens, and steps load at most 64
            // bytes past a stream position: all within those bytes.
            const std::size_t chunk = std::min(kChunk, 2 * (count - 64)) & ~std::size_t(3);
            const std::size_t stream = chunk / 4;
            const char* s[4] = {p, after_separator(p + stream), after_separator(p + 2 * stream),
                                after_separator(p + 3 * stream)};
            const char* const end[4] = {s[1], s[2], s[3], after_separator(p + chunk)};
            std::uint64_t* const out[4] = {dst, scratch_[0], scratch_[1], scratch_[2]};
            std::uint64_t lengths = 0;
            const std::size_t done = lockstep(s, end, out, lengths);
            std::size_t n[4];
            bool overrun = false;
            for (int k = 0; k < 4; ++k) {
                n[k] = done;
                while (s[k] < end[k] && n[k] < kCapacity) out[k][n[k]++] = one_token(s[k], lengths);
                overrun |= s[k] != end[k];
            }
            if (overrun || lengths >= 32) [[unlikely]] {
                // Irregular whitespace: this chunk one token at a time.
                const std::size_t tokens = std::min(count_tokens(p, end[3]), count);
                p = parse_slowly(p, dst, tokens);
                dst += tokens;
                count -= tokens;
                continue;
            }
            std::uint64_t* next = dst + n[0];
            for (int k = 1; k < 4; ++k) {
                std::memcpy(next, out[k], n[k] * sizeof(std::uint64_t));
                next += n[k];
            }
            dst = next;
            count -= n[0] + n[1] + n[2] + n[3];
            p = skip_whitespace(end[3]);
        }
        return {p, std::size_t(dst - first)};
    }

    static const char* skip_whitespace(const char* p) {
        while (static_cast<unsigned char>(*p) <= ' ') ++p;
        return p;
    }

private:
    static constexpr std::size_t kChunk = std::size_t(1) << 17;
    // A step advances a stream by at least 2 bytes and yields two values.
    static constexpr std::size_t kCapacity = kChunk / 4 + 64;
    alignas(64) static inline std::uint64_t scratch_[3][kCapacity];

    // Steps all streams while each has 42 bytes left (two tokens of 21 bytes) and the outputs have
    // room; returns the values per stream. lengths collects (length - 1) of each token by OR: 32
    // or more flags an empty token. Works on local copies of the stream state.
    static std::size_t lockstep(const char* (&streams)[4], const char* const (&end)[4],
                                std::uint64_t* const (&outputs)[4], std::uint64_t& flags) {
        const char* s[4] = {streams[0], streams[1], streams[2], streams[3]};
        std::uint64_t* const out[4] = {outputs[0], outputs[1], outputs[2], outputs[3]};
        std::uint64_t lengths = flags;
        std::size_t done = 0;
        for (;;) {
            std::ptrdiff_t left = end[0] - s[0];
            for (int k = 1; k < 4; ++k) left = std::min(left, end[k] - s[k]);
            const std::ptrdiff_t steps = std::min(left / 42, std::ptrdiff_t(kCapacity - done) / 2);
            if (steps == 0) break;
            for (std::ptrdiff_t step = 0; step < steps; ++step, done += 2) {
                const __m256i l0 = two_tokens(s[0], lengths), l1 = two_tokens(s[1], lengths);
                const __m256i l2 = two_tokens(s[2], lengths), l3 = two_tokens(s[3], lengths);
                const __m256i v01 = values(l0, l1), v23 = values(l2, l3);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out[0] + done), _mm256_castsi256_si128(v01));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out[1] + done), _mm256_extracti128_si256(v01, 1));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out[2] + done), _mm256_castsi256_si128(v23));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out[3] + done), _mm256_extracti128_si256(v23, 1));
            }
        }
        for (int k = 0; k < 4; ++k) streams[k] = s[k];
        flags = lengths;
        return done;
    }

    // 4-digit groups of the n <= 64 bytes that end at end: dwords 3..7 for 20 digits, 0 before.
    [[gnu::always_inline]] static __m256i groups_ending(const char* end, std::size_t n) {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(end - 32));
        const __m256i mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kDigitMask.data() + n));
        return digit_groups(_mm256_subs_epu8(bytes, mask));
    }

    // Two tokens a, b at s as 8-digit limbs [0, top a, 0, top b | mid a, low a, mid b, low b],
    // value = top * 10^16 + mid * 10^8 + low. Bit 63 of the separator mask keeps both lengths
    // below 64, so the loads stay within 64 bytes of s.
    [[gnu::always_inline]] static __m256i two_tokens(const char*& s, std::uint64_t& lengths) {
        const std::uint64_t low = separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s)));
        const std::uint64_t high = separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 32)));
        const std::uint64_t sep = low | (high | 0x80000000) << 32;
        const auto first = std::size_t(std::countr_zero(sep));              // length of a
        const auto second = std::size_t(std::countr_zero(sep & (sep - 1)));  // 64 if none
        const std::size_t length = second - first - 1;                       // of b
        lengths |= (first - 1) | (length - 1);
        const __m256i a = groups_ending(s + first, first), b = groups_ending(s + second, length);
        s += second + 1;
        return _mm256_madd_epi16(_mm256_packus_epi32(a, b), _mm256_set1_epi32(0x00012710));
    }

    // Limbs of two token pairs j, k -> values [j a, j b | k a, k b].
    static __m256i values(__m256i j, __m256i k) {
        constexpr std::uint64_t kTen16 = 10000000000000000;  // top * kTen16 in two 32-bit halves, mod 2^64
        const __m256i tops = _mm256_permute2x128_si256(j, k, 0x20);  // top in each high dword
        const __m256i rest = _mm256_permute2x128_si256(j, k, 0x31);  // [mid, low] per qword
        const __m256i top_low = _mm256_mul_epu32(_mm256_srli_epi64(tops, 32), _mm256_set1_epi64x(kTen16 & 0xFFFFFFFF));
        const __m256i top_high = _mm256_mullo_epi32(tops, _mm256_set1_epi64x(std::int64_t(kTen16 >> 32 << 32)));
        const __m256i mid = _mm256_mul_epu32(rest, _mm256_set1_epi64x(100000000));
        return _mm256_add_epi64(_mm256_add_epi64(top_low, top_high), _mm256_add_epi64(mid, _mm256_srli_epi64(rest, 32)));
    }

    static std::uint64_t one_token(const char*& s, std::uint64_t& lengths) {
        const std::uint32_t sep = separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s)));
        const auto n = unsigned(std::countr_zero(sep | 0x80000000));
        lengths |= std::uint64_t(n) - 1;
        s += n + 1;
        return parse_ending24(s - 1, n);
    }

    // The byte after the first separator at or after q.
    static const char* after_separator(const char* q) {
        for (;; q += 32)
            if (const std::uint32_t sep = separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q))))
                return q + std::countr_zero(sep) + 1;
    }

    // Tokens in [p, end), where p starts a token.
    static std::size_t count_tokens(const char* p, const char* end) {
        std::size_t tokens = 1;
        for (const char* q = p + 1; q < end; ++q)
            tokens += static_cast<unsigned char>(*q) > ' ' && static_cast<unsigned char>(q[-1]) <= ' ';
        return tokens;
    }

    static const char* parse_slowly(const char* p, std::uint64_t* dst, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            p = skip_whitespace(p);
            const std::uint32_t sep = separators(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
            const auto n = unsigned(std::countr_zero(sep | 0x80000000));
            dst[i] = parse_ending24(p + n, n);
            p += n + 1;
        }
        return p;
    }
};

}  // namespace detail

// count uint64 tokens into dst, as in.read(dst, count) but with the AVX2 bulk parser.
inline void read_bulk(Reader& in, std::uint64_t* dst, std::size_t count) {
    std::size_t parsed = 0;
    if (count >= detail::BulkParser64::kMinTokens) {
        const char* const start = detail::BulkParser64::skip_whitespace(in.scan().cur);
        const auto [stop, done] = detail::BulkParser64::parse(start, dst, count);
        // Continue the Reader at stop: its separator mask from there on.
        const auto* block = reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(stop) & ~std::uintptr_t(63));
        in.resume({stop, block, detail::block_separators(block) & ~std::uint64_t(0) << (stop - block)});
        parsed = done;
    }
    for (std::size_t i = parsed; i < count; ++i) dst[i] = in.read<std::uint64_t>();
}

}  // namespace io
