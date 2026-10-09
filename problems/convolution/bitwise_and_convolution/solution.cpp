// Bitwise AND convolution mod 998244353: c = mu(zeta(a) * zeta(b)), where zeta(f)[i] sums f over
// the supersets of i and mu inverts it. One level per index bit; the levels commute.
//
// Layout: 2^lg values as rows of 2^block_log values, each row kSkew values longer, so that equal
// columns of different rows fall in different cache sets (without the skew, the column pass takes
// 12 times as long). Each row is transformed over its own bits right after it is parsed; one pass
// over columns then does the row-bit levels of a and b, the product and the inverse row-bit levels;
// each row then gets its inverse low levels and is printed.
#include <sys/mman.h>

#include "lib/io/io.hpp"
#include "../fixed_width.hpp"

namespace {

using Vec = __m256i;

constexpr std::uint32_t kP = 998244353;
constexpr std::uint32_t kPInverse = [] {  // P^-1 mod 2^32
    std::uint32_t x = kP;
    for (int i = 0; i < 5; ++i) x *= 2 - kP * x;
    return x;
}();
constexpr std::uint32_t kR = std::uint32_t((std::uint64_t(1) << 32) % kP);
constexpr std::uint32_t kRQuotient = std::uint32_t((std::uint64_t(kR) << 32) / kP);  // Shoup

#ifndef BLOCK_LOG
#define BLOCK_LOG 17  // tests set 6 to get several rows from small inputs
#endif

constexpr int kMinLog = 6;  // shorter inputs are padded with zeros (zero entries add nothing)
constexpr int kBlockLog = BLOCK_LOG;  // row length
constexpr int kMaxRowsLog = 3;        // N <= 20
constexpr int kPieceLog = 12;         // 16 KiB, in L1
constexpr std::size_t kSkew = 16;     // one cache line

inline Vec broadcast(std::uint32_t x) { return _mm256_set1_epi32(int(x)); }

// x + y (zeta) or x - y (mu) mod P, for x, y < P.
template <bool Inverse>
[[gnu::always_inline]] inline Vec step(Vec x, Vec y) {
    if constexpr (Inverse) {
        const Vec d = _mm256_sub_epi32(x, y);
        return _mm256_min_epu32(d, _mm256_add_epi32(d, broadcast(kP)));
    } else {
        const Vec s = _mm256_add_epi32(x, y);
        return _mm256_min_epu32(s, _mm256_sub_epi32(s, broadcast(kP)));
    }
}

// The levels of the K bits of the index into x[0, 2^K).
template <bool Inverse, int K>
[[gnu::always_inline]] inline void butterflies(Vec* x) {
#pragma GCC unroll 8
    for (int bit = 1; bit < (1 << K); bit <<= 1)
#pragma GCC unroll 32
        for (int k = 0; k < (1 << K); ++k)
            if (!(k & bit)) x[k] = step<Inverse>(x[k], x[k | bit]);
}

// x[i] lane j <-> x[j] lane i.
[[gnu::always_inline]] inline void transpose(Vec* x) {
    Vec t[8], u[8];
#pragma GCC unroll 4
    for (int k = 0; k < 8; k += 2) {
        t[k] = _mm256_unpacklo_epi32(x[k], x[k + 1]);
        t[k + 1] = _mm256_unpackhi_epi32(x[k], x[k + 1]);
    }
#pragma GCC unroll 2
    for (int k = 0; k < 8; k += 4) {
        u[k] = _mm256_unpacklo_epi64(t[k], t[k + 2]);
        u[k + 1] = _mm256_unpackhi_epi64(t[k], t[k + 2]);
        u[k + 2] = _mm256_unpacklo_epi64(t[k + 1], t[k + 3]);
        u[k + 3] = _mm256_unpackhi_epi64(t[k + 1], t[k + 3]);
    }
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
        x[k] = _mm256_permute2x128_si256(u[k], u[k + 4], 0x20);
        x[k + 4] = _mm256_permute2x128_si256(u[k], u[k + 4], 0x31);
    }
}

// Levels of K vector-index bits from stride h (vectors) on f[0, count).
template <bool Inverse, int K>
void sweep(Vec* f, std::size_t count, std::size_t h) {
    for (std::size_t g = 0; g < count; g += h << K)
        for (std::size_t j = g; j < g + h; ++j) {
            Vec x[1 << K];
#pragma GCC unroll 8
            for (int k = 0; k < (1 << K); ++k) x[k] = f[j + k * h];
            butterflies<Inverse, K>(x);
#pragma GCC unroll 8
            for (int k = 0; k < (1 << K); ++k) f[j + k * h] = x[k];
        }
}

// Levels of the vector-index bits [from, to) on f[0, count).
template <bool Inverse>
void sweeps(Vec* f, std::size_t count, int from, int to) {
    for (; from + 3 <= to; from += 3) sweep<Inverse, 3>(f, count, std::size_t(1) << from);
    if (to - from == 2) sweep<Inverse, 2>(f, count, std::size_t(1) << from);
    if (to - from == 1) sweep<Inverse, 1>(f, count, std::size_t(1) << from);
}

// The lane bits and the 3 vector-index bits above them, tile by tile (8 vectors): the vector bits,
// a transpose, then the former lane bits. The transform leaves each tile transposed; the inverse
// takes transposed tiles and restores them.
template <bool Inverse>
void tiles(Vec* f, std::size_t count) {
    for (std::size_t g = 0; g < count; g += 8) {
        Vec x[8];
#pragma GCC unroll 8
        for (int k = 0; k < 8; ++k) x[k] = f[g + k];
        butterflies<Inverse, 3>(x);
        transpose(x);
        butterflies<Inverse, 3>(x);
#pragma GCC unroll 8
        for (int k = 0; k < 8; ++k) f[g + k] = x[k];
    }
}

// All levels of a row of 2^bits values, bits >= 6: pieces of 2^kPieceLog values (in L1), then the
// row's upper bits. The inverse goes the other way, so each piece ends in natural order.
template <bool Inverse>
void row_levels(std::uint32_t* row, int bits) {
    Vec* f = reinterpret_cast<Vec*>(row);
    const std::size_t count = std::size_t(1) << (bits - 3);
    const int piece_bits = std::min(bits, kPieceLog) - 3;  // vector-index bits
    const std::size_t piece = std::size_t(1) << piece_bits;
    if constexpr (Inverse) sweeps<true>(f, count, piece_bits, bits - 3);
    for (std::size_t p = 0; p < count; p += piece) {
        tiles<Inverse>(f + p, piece);
        sweeps<Inverse>(f + p, piece, 3, piece_bits);
    }
    if constexpr (!Inverse) sweeps<false>(f, count, piece_bits, bits - 3);
}

// x y 2^-32 mod P in [0, P), for x, y < P: Montgomery, q = lo(x y) P^-1, x y - q P = 2^32 (hi - hi).
[[gnu::always_inline]] inline Vec montgomery(Vec x, Vec y) {
    const Vec even = _mm256_mul_epu32(x, y);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32));
    const Vec q_even = _mm256_mul_epu32(even, broadcast(kPInverse));
    const Vec q_odd = _mm256_mul_epu32(odd, broadcast(kPInverse));
    const Vec m_even = _mm256_mul_epu32(q_even, broadcast(kP));
    const Vec m_odd = _mm256_mul_epu32(q_odd, broadcast(kP));
    const Vec high = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);  // < P
    const Vec m_high = _mm256_blend_epi32(_mm256_srli_epi64(m_even, 32), m_odd, 0xAA);  // < P
    const Vec d = _mm256_sub_epi32(high, m_high);
    return _mm256_min_epu32(d, _mm256_add_epi32(d, broadcast(kP)));
}

// x 2^32 mod P in [0, P), for x < P: Shoup, q = floor(x floor(2^32 R / P) / 2^32).
[[gnu::always_inline]] inline Vec times_r(Vec x) {
    const Vec q_even = _mm256_srli_epi64(_mm256_mul_epu32(x, broadcast(kRQuotient)), 32);
    const Vec q_odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), broadcast(kRQuotient));
    const Vec q = _mm256_blend_epi32(q_even, q_odd, 0xAA);
    const Vec r = _mm256_sub_epi32(_mm256_mullo_epi32(x, broadcast(kR)), _mm256_mullo_epi32(q, broadcast(kP)));
    return _mm256_min_epu32(r, _mm256_sub_epi32(r, broadcast(kP)));  // r < 2P
}

// Column by column: the row-bit levels of a and b, the product into a, the inverse row-bit levels.
template <int RowsLog>
void combine(std::uint32_t* a_rows, const std::uint32_t* b_rows, std::size_t columns, std::size_t stride) {
    constexpr int kRows = 1 << RowsLog;
    Vec* a = reinterpret_cast<Vec*>(a_rows);
    const Vec* b = reinterpret_cast<const Vec*>(b_rows);
    const std::size_t s = stride / 8;
    for (std::size_t j = 0; j < columns / 8; ++j) {
        Vec x[kRows], y[kRows];
#pragma GCC unroll 8
        for (int r = 0; r < kRows; ++r) x[r] = a[r * s + j], y[r] = b[r * s + j];
        butterflies<false, RowsLog>(x);
        butterflies<false, RowsLog>(y);
#pragma GCC unroll 8
        for (int r = 0; r < kRows; ++r) x[r] = times_r(montgomery(x[r], y[r]));
        butterflies<true, RowsLog>(x);
#pragma GCC unroll 8
        for (int r = 0; r < kRows; ++r) a[r * s + j] = x[r];
    }
}

// Zeroed memory for two arrays of words, 2 MiB aligned, in huge pages where the kernel allows.
std::uint32_t* allocate(std::size_t words) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    const std::size_t bytes = (words * sizeof(std::uint32_t) + kHuge - 1) / kHuge * kHuge + kHuge;
    void* region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) std::abort();
    const std::uintptr_t start = (reinterpret_cast<std::uintptr_t>(region) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
    ::madvise(reinterpret_cast<void*>(start), bytes - kHuge, MADV_HUGEPAGE);
#endif
    return reinterpret_cast<std::uint32_t*>(start);
}

}  // namespace

int main() {
    io::Reader in;
    const int n = int(in.read<std::uint32_t>());
    const std::size_t total = std::size_t(1) << n;
    const int lg = std::max(n, kMinLog), block_log = std::min(lg, kBlockLog), rows_log = lg - block_log;
    const std::size_t block = std::size_t(1) << block_log, stride = block + kSkew, rows = std::size_t(1) << rows_log;
    std::uint32_t* const a = allocate(2 * rows * stride);
    std::uint32_t* const b = a + rows * stride;
    for (std::uint32_t* f : {a, b})
        for (std::size_t r = 0; r < rows; ++r) {
            in.read(f + r * stride, std::min(block, total));
            row_levels<false>(f + r * stride, block_log);
        }
    switch (rows_log) {
    case 0: combine<0>(a, b, block, stride); break;
    case 1: combine<1>(a, b, block, stride); break;
    case 2: combine<2>(a, b, block, stride); break;
    default: combine<kMaxRowsLog>(a, b, block, stride); break;
    }
    io::Writer out;
    for (std::size_t r = 0; r < rows; ++r) {
        row_levels<true>(a + r * stride, block_log);
        fixed_width::write(out, a + r * stride, std::min(block, total));
    }
}
