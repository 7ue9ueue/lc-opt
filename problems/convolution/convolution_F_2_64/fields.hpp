// Output of uint64 values separated by whitespace, then a newline. Blocks of mostly large values
// are printed in fixed width by ../convolution_mod_2_64/fields64.hpp: a space, then each value
// right-aligned in 20 characters. Judge-specific: the checker (testlib wcmp) compares tokens, so
// the padding is accepted. Blocks with many short values stay variable-width, where padding would
// cost more in write(2).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "lib/io/io.hpp"
#include "../convolution_mod_2_64/fields64.hpp"

namespace fields {

namespace detail {

// Whether at least 1/8 of values[0, count) are below 2^53 (at most 16 digits).
inline bool mostly_short(const std::uint64_t* values, std::size_t count) {
    __m256i shorts = _mm256_setzero_si256();  // minus the count per lane
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const __m256i top = _mm256_srli_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values + i)), 53);
        shorts = _mm256_add_epi64(shorts, _mm256_cmpeq_epi64(top, _mm256_setzero_si256()));
    }
    alignas(32) std::int64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), shorts);
    std::size_t n = std::size_t(-(lanes[0] + lanes[1] + lanes[2] + lanes[3]));
    for (; i < count; ++i) n += values[i] >> 53 == 0;
    return 8 * n >= count;
}

}  // namespace detail

// values[0, count), count >= 1, readable up to count rounded up to 8. text: fields64::kTextBytes
// bytes, 32-byte aligned.
inline void write(io::Writer& out, const std::uint64_t* values, std::size_t count, char* text) {
    bool prepared = false;
    for (std::size_t i = 0; i < count; i += fields64::kBlock) {
        const std::size_t n = std::min(fields64::kBlock, count - i);
        const bool last = i + n == count;
        if (detail::mostly_short(values + i, n)) {
            out.write(' ');
            out.write_array(values + i, n, ' ');
            if (last) out.write('\n');
            continue;
        }
        if (!prepared) fields64::prepare(text), prepared = true;
        out.write(std::string_view(text, fields64::format(text, values + i, n, last)));
    }
}

}  // namespace fields
