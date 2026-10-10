// 1 / f mod (x^N, y^M), N M <= 500000: Newton iteration in one variable on the Kronecker
// substitution of the other (lib/poly/inverse_2d.hpp), in whichever orientation its cost model
// prefers. Input by lib/io/bulk32.hpp; output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp). One arena holds every array.
#include <immintrin.h>

#include <cstring>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/poly/inverse_2d.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

using u32 = std::uint32_t;

// to[j * to_stride + i] = from[i * from_stride + j] for i < rows, j < cols, in 8 x 8 blocks.
void transpose(const u32* from, std::size_t from_stride, std::size_t rows, std::size_t cols, u32* to,
               std::size_t to_stride) {
    const std::size_t rows8 = rows / 8 * 8, cols8 = cols / 8 * 8;
    for (std::size_t i = 0; i < rows8; i += 8) {
        for (std::size_t j = 0; j < cols8; j += 8) {
            __m256i r[8], s[8];
            for (int k = 0; k < 8; ++k)
                r[k] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(from + (i + k) * from_stride + j));
            for (int k = 0; k < 8; k += 2) {
                s[k] = _mm256_unpacklo_epi32(r[k], r[k + 1]);
                s[k + 1] = _mm256_unpackhi_epi32(r[k], r[k + 1]);
            }
            for (int k = 0; k < 8; k += 4) {
                r[k] = _mm256_unpacklo_epi64(s[k], s[k + 2]);
                r[k + 1] = _mm256_unpackhi_epi64(s[k], s[k + 2]);
                r[k + 2] = _mm256_unpacklo_epi64(s[k + 1], s[k + 3]);
                r[k + 3] = _mm256_unpackhi_epi64(s[k + 1], s[k + 3]);
            }
            // r[0..3]: columns 0, 1, 2, 3 (low lanes) and 4, 5, 6, 7 (high lanes) of rows 0-3; r[4..7]: rows 4-7.
            for (int k = 0; k < 4; ++k) {
                const __m256i lo = _mm256_permute2x128_si256(r[k], r[k + 4], 0x20);
                const __m256i hi = _mm256_permute2x128_si256(r[k], r[k + 4], 0x31);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(to + (j + k) * to_stride + i), lo);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(to + (j + k + 4) * to_stride + i), hi);
            }
        }
        for (std::size_t j = cols8; j < cols; ++j)
            for (std::size_t k = i; k < i + 8; ++k) to[j * to_stride + k] = from[k * from_stride + j];
    }
    for (std::size_t i = rows8; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) to[j * to_stride + i] = from[i * from_stride + j];
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<u32>(), m = in.read<u32>(), count = n * m;
    const poly::Inverse2d by_rows(n, m), by_columns(m, n);
    // Iterate in y (rows of the plan are columns of f) when cheaper; a single row or column is
    // the one-dimensional inverse either way, and then the layouts need no transposition.
    const bool transposed = m == 1 || (n > 1 && by_columns.cost() < by_rows.cost());
    const poly::Inverse2d& plan = transposed ? by_columns : by_rows;
    const bool moves = !transposed || n == 1 || m == 1;  // rows stay rows; else a transposition
    const std::size_t rows = transposed ? m : n, cols = transposed ? n : m, stride = plan.stride();
    const std::size_t split = plan.split();
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(u32);
    using poly::Arena;
    Arena arena(poly::Transform::words(plan.lg()) + Arena::footprint(plan.f_words()) + Arena::footprint(plan.g_words()) +
                Arena::footprint(plan.scratch_words()) + Arena::footprint(kTextWords));
    const poly::Transform transform(arena, plan.lg());
    const std::span<u32> f = arena.take(plan.f_words()), g = arena.take(plan.g_words());
    const std::span<u32> scratch = arena.take(plan.scratch_words());  // >= count words
    u32* const rows_at = f.data() + plan.offset();
    if (moves) {
        // Dense rows, then each row moved to its place, last first (row i moves up by i (S - C)).
        io::read_bulk(in, rows_at, count);
        for (std::size_t i = rows; i-- > 0;) {
            std::memmove(rows_at + i * stride, rows_at + i * cols, cols * sizeof(u32));
            const std::size_t gap = i * stride + cols, end = std::min((i + 1) * stride, count);
            if (gap < end) std::memset(rows_at + gap, 0, (end - gap) * sizeof(u32));
        }
    } else {
        io::read_bulk(in, scratch.data(), count);
        transpose(scratch.data(), m, n, m, rows_at, stride);
    }
    plan.run(transform, f, g, scratch);
    // Rows 0 .. split - 1 of the result are in g, the others in f's buffer at rows_at.
    const u32* result = scratch.data();
    if (moves) {
        u32* const to = split == rows ? g.data() : f.data();
        for (std::size_t i = split; i < rows; ++i) std::memmove(to + i * cols, rows_at + i * stride, cols * sizeof(u32));
        for (std::size_t i = 0; i < split; ++i) std::memmove(to + i * cols, g.data() + i * stride, cols * sizeof(u32));
        result = to;
    } else {
        transpose(g.data(), stride, split, n, scratch.data(), m);
        transpose(rows_at + split * stride, stride, rows - split, n, scratch.data() + split, m);
    }
    io::Writer out;
    fields::write(out, result, count, reinterpret_cast<char*>(arena.take(kTextWords).data()));
}

}  // namespace

RUN_EARLY(solve)
