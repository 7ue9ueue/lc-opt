// io::read_bulk (lib/io/bulk32.hpp) with a progress callback, so that a caller can consume the
// values while they are still in the cache, without splitting the read into calls:
//
//   progress::read(in, dst, count, [&](std::size_t parsed) { ... dst[0, parsed) are final ... });
//
// Each read_bulk call ends with shrinking chunks and up to 1023 tokens parsed one at a time: in
// 2^16-token calls that costs 0.25 ms per 2^21 tokens on Zen 3 (notes.md, round 4). This is a
// copy of BulkParser32 that reports after each chunk of input (256 KiB here, 128 KiB in lib/io).
// Unlike read_bulk it runs on every CPU: on Intel (lc-intel) it beat Reader::read in 2^16-token
// pieces too.
#pragma once

#include "lib/io/io.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace progress {

namespace detail {

using io::detail::align_row;
using io::detail::digit_groups;
using io::detail::parse16;
using io::detail::separators;

// BulkParser32 of lib/io/bulk32.hpp with chunks of 2^18 bytes; parse() calls done(parsed) after
// each chunk.
class Parser {
public:
    static constexpr std::size_t kMinTokens = 1024;

    struct Result {
        const char* end;
        std::size_t parsed;
    };

    template <class Done>
    static Result parse(const char* p, std::uint32_t* dst, std::size_t count, Done& done) {
        std::uint32_t* const first = dst;
        while (count >= kMinTokens) {
            // The count tokens span at least 2 * count - 1 bytes. A chunk spans at most chunk + 17
            // bytes, so it holds at most chunk / 2 + 9 < count tokens, and loads stay below
            // p + chunk + 33: all within those bytes.
            const std::size_t chunk = std::min(kChunk, 2 * (count - 32)) & ~std::size_t(3);
            const std::size_t stream = chunk / 4;
            const char* s[4] = {p, after_separator(p + stream), after_separator(p + 2 * stream),
                                after_separator(p + 3 * stream)};
            const char* const end[4] = {s[1], s[2], s[3], after_separator(p + chunk)};
            __m256i invalid = _mm256_setzero_si256();
            const std::size_t steps = lockstep(s, end, invalid);
            std::size_t tail[4];
            bool overrun = false;
            for (int k = 0; k < 4; ++k) {
                tail[k] = 0;
                while (s[k] < end[k] && tail[k] < kTail) tails_[k][tail[k]++] = one_token(s[k], invalid);
                overrun |= s[k] != end[k];
            }
            if (overrun || (std::uint32_t(_mm256_movemask_epi8(invalid)) & 0x80008000)) [[unlikely]] {
                // Irregular whitespace or tokens: this chunk one token at a time.
                const std::size_t tokens = std::min(count_tokens(p, end[3]), count);
                p = parse_slowly(p, dst, tokens);
                dst += tokens;
                count -= tokens;
                done(std::size_t(dst - first));
                continue;
            }
            std::uint32_t* out[4] = {dst};
            for (int k = 0; k < 3; ++k) out[k + 1] = out[k] + 2 * steps + tail[k];
            transpose(steps, out);
            for (int k = 0; k < 4; ++k) std::memcpy(out[k] + 2 * steps, tails_[k], tail[k] * sizeof(std::uint32_t));
            const auto parsed = std::size_t(out[3] + 2 * steps + tail[3] - dst);
            dst += parsed;
            count -= parsed;
            p = skip_whitespace(end[3]);
            done(std::size_t(dst - first));
        }
        return {p, std::size_t(dst - first)};
    }

    static const char* skip_whitespace(const char* p) {
        while (static_cast<unsigned char>(*p) <= ' ') ++p;
        return p;
    }

private:
    static constexpr std::size_t kChunk = std::size_t(1) << 18;
    // A step of valid tokens advances a stream by at least 4 bytes; lockstep stops at kSteps.
    static constexpr std::size_t kSteps = kChunk / 16 + 16;
    // A token takes at least 2 bytes of a stream.
    static constexpr std::size_t kTail = kChunk / 8 + 32;
    alignas(64) static inline std::uint32_t steps_[8 * kSteps];
    alignas(64) static inline std::uint32_t tails_[4][kTail];

    // Steps all streams while each has 33 bytes left; returns the steps. Step j stores its values
    // at steps_ + 8j. Works on local copies of the stream state, which stay in registers.
    static std::size_t lockstep(const char* (&streams)[4], const char* const (&end)[4], __m256i& flags) {
        const char* s[4] = {streams[0], streams[1], streams[2], streams[3]};
        __m256i invalid = flags;
        std::size_t done = 0;
        for (;;) {
            std::ptrdiff_t left = end[0] - s[0];
            for (int k = 1; k < 4; ++k) left = std::min(left, end[k] - s[k]);
            const std::ptrdiff_t steps = std::min(left / 33, std::ptrdiff_t(kSteps - done));
            if (steps <= 0) break;
            for (std::uint32_t* v = steps_ + 8 * done; v != steps_ + 8 * (done + std::size_t(steps)); v += 8) {
                const __m256i g0 = two_tokens(s[0], invalid), g1 = two_tokens(s[1], invalid);
                const __m256i g2 = two_tokens(s[2], invalid), g3 = two_tokens(s[3], invalid);
                const __m256i k = _mm256_set1_epi32(0x00012710);
                // 8-digit halves: [s0 high, s0 low, s1 high, s1 low | second tokens likewise].
                const __m256 h01 = _mm256_castsi256_ps(_mm256_madd_epi16(_mm256_packus_epi32(g0, g1), k));
                const __m256 h23 = _mm256_castsi256_ps(_mm256_madd_epi16(_mm256_packus_epi32(g2, g3), k));
                const __m256i high = _mm256_castps_si256(_mm256_shuffle_ps(h01, h23, 0x88));
                const __m256i low = _mm256_castps_si256(_mm256_shuffle_ps(h01, h23, 0xDD));
                _mm256_store_si256(reinterpret_cast<__m256i*>(v),
                                   _mm256_add_epi32(_mm256_mullo_epi32(high, _mm256_set1_epi32(100000000)), low));
            }
            done += std::size_t(steps);
        }
        for (int k = 0; k < 4; ++k) streams[k] = s[k];
        flags = invalid;
        return done;
    }

    // The 2 * steps values of stream k from steps_ to out[k]: a 4x4 transpose of value pairs.
    static void transpose(std::size_t steps, std::uint32_t* const (&out)[4]) {
        const auto pairs = [](std::size_t j) {  // step j as [a0 b0 a1 b1 | a2 b2 a3 b3]
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(steps_ + 8 * j));
            return _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        };
        std::size_t j = 0;
        for (; j + 4 <= steps; j += 4) {
            const __m256i p0 = pairs(j), p1 = pairs(j + 1), p2 = pairs(j + 2), p3 = pairs(j + 3);
            const __m256i t0 = _mm256_unpacklo_epi64(p0, p1), t1 = _mm256_unpackhi_epi64(p0, p1);
            const __m256i t2 = _mm256_unpacklo_epi64(p2, p3), t3 = _mm256_unpackhi_epi64(p2, p3);
            const auto put = [j](std::uint32_t* to, __m256i v) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(to + 2 * j), v);
            };
            put(out[0], _mm256_permute2x128_si256(t0, t2, 0x20));
            put(out[1], _mm256_permute2x128_si256(t1, t3, 0x20));
            put(out[2], _mm256_permute2x128_si256(t0, t2, 0x31));
            put(out[3], _mm256_permute2x128_si256(t1, t3, 0x31));
        }
        for (; j < steps; ++j)
            for (int k = 0; k < 4; ++k) {
                out[k][2 * j] = steps_[8 * j + k];
                out[k][2 * j + 1] = steps_[8 * j + 4 + k];
            }
    }

    // Two tokens at s, as 4-digit groups [first | second]. Lengths outside 1..16 (repeated
    // whitespace, long tokens) set the sign bit of byte 15 or 31 of invalid.
    [[gnu::always_inline]] static __m256i two_tokens(const char*& s, __m256i& invalid) {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
        const std::uint32_t sep = separators(bytes);
        const auto first = std::size_t(std::countr_zero(sep));              // length of the first token
        const auto second = std::size_t(std::countr_zero(sep & (sep - 1)));  // 32 if none
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

// count uint32 tokens into dst: the same values as io::read_bulk(in, dst, count). Calls
// done(parsed) as values become final, dst[0, parsed) with parsed increasing, last with count.
template <class Done>
void read(io::Reader& in, std::uint32_t* dst, std::size_t count, Done&& done) {
    using detail::Parser;
    std::size_t parsed = 0;
    if (count >= Parser::kMinTokens) {
        const char* const start = Parser::skip_whitespace(in.scan().cur);
        const auto [stop, bulk] = Parser::parse(start, dst, count, done);
        // Continue the Reader at stop: its separator mask from there on.
        const auto* block = reinterpret_cast<const char*>(reinterpret_cast<std::uintptr_t>(stop) & ~std::uintptr_t(63));
        in.resume({stop, block, io::detail::block_separators(block) & ~std::uint64_t(0) << (stop - block)});
        parsed = bulk;
    }
    for (std::size_t i = parsed; i < count; ++i) dst[i] = in.read<std::uint32_t>();
    done(count);
}

}  // namespace progress
