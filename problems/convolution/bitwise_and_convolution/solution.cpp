// Bitwise AND convolution mod 998244353: c = mu(zeta(a) * zeta(b)), where zeta(f)[i] sums f over
// the supersets of i and mu inverts it. One level per index bit; the levels commute.
//
// Layout: 2^lg values as rows of 2^block_log values. Bands of up to 4 rows are contiguous and read
// by one Reader call; each band (of a, then of b) starts kBandSkew values after the previous one
// ends, so equal columns of the rows of a and b spread over 4 groups of cache sets (without a skew,
// the column pass takes 12 times as long). Each band's rows are transformed over their own bits
// right after it is parsed; one pass over columns then does the row-bit levels of a and b, the
// product and the inverse row-bit levels; each row then gets its inverse low levels, and the
// band's values are printed as they become ready.
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "../convolution_mod/fields.hpp"
#include "../text_buffer.hpp"

namespace {

using Vec = __m256i;

constexpr std::uint32_t kP = 998244353;
constexpr std::uint32_t kBarrett = std::uint32_t((std::uint64_t(1) << 61) / kP);

#ifndef BLOCK_LOG
#define BLOCK_LOG 17  // tests set 6 to get several rows from small inputs
#endif

constexpr int kMinLog = 6;  // shorter inputs are padded with zeros (zero entries add nothing)
constexpr int kBlockLog = BLOCK_LOG;  // row length
constexpr int kMaxRowsLog = 3;        // N <= 20
constexpr int kBandRowsLog = 2;       // 2 MiB per Reader call at N = 20
constexpr int kPieceLog = 12;         // 16 KiB, in L1
constexpr std::size_t kBandSkew = 256;  // 16 cache lines: 4 bands spread over 64 L1 sets

// Offset of row r in an array: rows of a band are contiguous, bands are kBandSkew apart.
constexpr std::size_t row_offset(std::size_t r, std::size_t block) {
    return r * block + (r >> kBandRowsLog) * kBandSkew;
}

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

// The 4x4 transposes of x[0, 4) and of x[4, 8) inside each 128-bit lane.
[[gnu::always_inline]] inline void transpose_lanes(Vec* x) {
    Vec t[8];
#pragma GCC unroll 4
    for (int k = 0; k < 8; k += 2) {
        t[k] = _mm256_unpacklo_epi32(x[k], x[k + 1]);
        t[k + 1] = _mm256_unpackhi_epi32(x[k], x[k + 1]);
    }
#pragma GCC unroll 2
    for (int k = 0; k < 8; k += 4) {
        x[k] = _mm256_unpacklo_epi64(t[k], t[k + 2]);
        x[k + 1] = _mm256_unpackhi_epi64(t[k], t[k + 2]);
        x[k + 2] = _mm256_unpacklo_epi64(t[k + 1], t[k + 3]);
        x[k + 3] = _mm256_unpackhi_epi64(t[k + 1], t[k + 3]);
    }
}

// x[i] lane j = f[j] lane i. The loads swap the two off-diagonal 4x4 blocks with 128-bit broadcasts
// and blends, which run on any of Zen 3's 4 vector pipes (shuffles run on 2); then each 128-bit
// lane is transposed.
[[gnu::always_inline]] inline void load_transposed(const Vec* f, Vec* x) {
    const auto* half = reinterpret_cast<const __m128i*>(f);  // half[2k], half[2k + 1]: f[k] low, high
#pragma GCC unroll 4
    for (int k = 0; k < 4; ++k) {
        x[k] = _mm256_blend_epi32(f[k], _mm256_broadcastsi128_si256(half[2 * k + 8]), 0xF0);
        x[k + 4] = _mm256_blend_epi32(_mm256_broadcastsi128_si256(half[2 * k + 1]), f[k + 4], 0xF0);
    }
    transpose_lanes(x);
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
            // x[2^K - 1] has every bit set: no level changes it.
#pragma GCC unroll 8
            for (int k = 0; k + 1 < (1 << K); ++k) f[j + k * h] = x[k];
        }
}

// Levels of the vector-index bits [from, to) on f[0, count).
template <bool Inverse>
void sweeps(Vec* f, std::size_t count, int from, int to) {
    for (; from + 3 <= to; from += 3) sweep<Inverse, 3>(f, count, std::size_t(1) << from);
    if (to - from == 2) sweep<Inverse, 2>(f, count, std::size_t(1) << from);
    if (to - from == 1) sweep<Inverse, 1>(f, count, std::size_t(1) << from);
}

// The lane bits and the 3 vector-index bits above them in each tile of 8 vectors: a sweep does the
// vector bits, then each tile is transposed and its former lane bits done. The transform leaves
// each tile transposed; the inverse takes transposed tiles and restores them. On Zen 3, one pass
// doing both took 32 cycles per tile in L1; these two take 27.5. The transform also prefetches
// `next` (count vectors): its first pass is light, and its pieces come from L3 right after parsing.
template <bool Inverse>
void tiles(Vec* f, std::size_t count, const Vec* next) {
    sweep<Inverse, 3>(f, count, 1);
    for (std::size_t g = 0; g < count; g += 8) {
        if constexpr (!Inverse)
#pragma GCC unroll 4
            for (int line = 0; line < 4; ++line)
                _mm_prefetch(reinterpret_cast<const char*>(next + g) + 64 * line, _MM_HINT_T0);
        Vec x[8];
        load_transposed(f + g, x);
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
        tiles<Inverse>(f + p, piece, f + p + piece);  // past the last piece: prefetches never fault
        sweeps<Inverse>(f + p, piece, 3, piece_bits);
    }
    if constexpr (!Inverse) sweeps<false>(f, count, piece_bits, bits - 3);
}

// x y mod P in [0, P), for x, y < P: Barrett. t = floor(x y / 2^29) < 2^31 and
// q = floor(t floor(2^61 / P) / 2^32) is floor(x y / P) or one less, so x y - q P < 2P < 2^32:
// the low words of x y and q P give it.
[[gnu::always_inline]] inline Vec multiply(Vec x, Vec y) {
    const Vec even = _mm256_mul_epu32(x, y);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32));
    const Vec q_even = _mm256_mul_epu32(_mm256_srli_epi64(even, 29), broadcast(kBarrett));
    const Vec q_odd = _mm256_mul_epu32(_mm256_srli_epi64(odd, 29), broadcast(kBarrett));
    const Vec q = _mm256_blend_epi32(_mm256_srli_epi64(q_even, 32), q_odd, 0xAA);
    const Vec r = _mm256_sub_epi32(_mm256_mullo_epi32(x, y), _mm256_mullo_epi32(q, broadcast(kP)));
    return _mm256_min_epu32(r, _mm256_sub_epi32(r, broadcast(kP)));
}

// Column by column: the row-bit levels of a and b, the product into a, the inverse row-bit levels.
template <int RowsLog>
void combine(std::uint32_t* a_rows, const std::uint32_t* b_rows, std::size_t block) {
    constexpr int kRows = 1 << RowsLog;
    Vec* a = reinterpret_cast<Vec*>(a_rows);
    const Vec* b = reinterpret_cast<const Vec*>(b_rows);
    std::size_t row[kRows];  // in vectors
    for (int r = 0; r < kRows; ++r) row[r] = row_offset(r, block) / 8;
    for (std::size_t j = 0; j < block / 8; ++j) {
        Vec x[kRows], y[kRows];
#pragma GCC unroll 8
        for (int r = 0; r < kRows; ++r) x[r] = a[row[r] + j], y[r] = b[row[r] + j];
        butterflies<false, RowsLog>(x);
        butterflies<false, RowsLog>(y);
#pragma GCC unroll 8
        for (int r = 0; r < kRows; ++r) x[r] = multiply(x[r], y[r]);
        butterflies<true, RowsLog>(x);
#pragma GCC unroll 8
        for (int r = 0; r < kRows; ++r) a[row[r] + j] = x[r];
    }
}

// Zeroed memory for words values: whole 2 MiB pages (huge where the kernel allows), and a remainder
// under 1 MiB in small pages just below them. At N = 20, 8 MiB + 4 KiB: 4 huge pages and 1 small.
std::uint32_t* allocate(std::size_t words) {
    constexpr std::size_t kHuge = std::size_t(1) << 21, kPage = 4096;
    const std::size_t bytes = words * sizeof(std::uint32_t);
    std::size_t huge = bytes / kHuge * kHuge, small = (bytes - huge + kPage - 1) / kPage * kPage;
    if (small >= kHuge / 2) huge += kHuge, small = 0;
    void* region = ::mmap(nullptr, small + huge + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) std::abort();
    const std::uintptr_t start = (reinterpret_cast<std::uintptr_t>(region) + small + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
    ::madvise(reinterpret_cast<void*>(start), huge, MADV_HUGEPAGE);
#endif
    return reinterpret_cast<std::uint32_t*>(start - small);
}

// Marks a mapped input as read once, so the kernel skips marking each page accessed when the
// Reader unmaps it (0.04 ms per 20 MB). The mapping starts at the page of the first token.
void advise_sequential(const io::Reader& in) {
    struct stat st;
    if (::fstat(0, &st) != 0 || !S_ISREG(st.st_mode) || std::size_t(st.st_size) <= io::detail::kMapAbove) return;
    const auto start = reinterpret_cast<std::uintptr_t>(in.scan().cur) & ~std::uintptr_t(4095);
    ::madvise(reinterpret_cast<void*>(start), std::size_t(st.st_size), MADV_SEQUENTIAL);
}

// The inverse low levels of each row of a band, then its values as fixed-width text: whole blocks
// as rows finish, the rest at the band's end, so every write(2) goes straight from text.
void print_band(io::Writer& out, std::uint32_t* band, std::size_t rows, std::size_t block, std::size_t count,
                int block_log, char* text) {
    std::size_t printed = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        row_levels<true>(band + r * block, block_log);
        const std::size_t ready = std::min((r + 1) * block, count);
        const std::size_t now = r + 1 < rows ? (ready - printed) / fields::kBlock * fields::kBlock : ready - printed;
        if (now) fields::write(out, band + printed, now, text);
        printed += now;
    }
}

void solve() {
    io::Reader in;
    const int n = int(in.read<std::uint32_t>());
    advise_sequential(in);
    const std::size_t total = std::size_t(1) << n;
    const int lg = std::max(n, kMinLog), block_log = std::min(lg, kBlockLog), rows_log = lg - block_log;
    const std::size_t block = std::size_t(1) << block_log, rows = std::size_t(1) << rows_log;
    const std::size_t band_rows = std::min(rows, std::size_t(1) << kBandRowsLog);
    const std::size_t size = row_offset(rows - 1, block) + block + kBandSkew;  // an array, then a skew
    std::uint32_t* const a = allocate(2 * size);
    std::uint32_t* const b = a + size;
    for (std::uint32_t* f : {a, b})
        for (std::size_t r = 0; r < rows; r += band_rows) {
            io::read_bulk(in, f + row_offset(r, block), std::min(band_rows * block, total));
            for (std::size_t k = r; k < r + band_rows; ++k) row_levels<false>(f + row_offset(k, block), block_log);
        }
    switch (rows_log) {
    case 0: combine<0>(a, b, block); break;
    case 1: combine<1>(a, b, block); break;
    case 2: combine<2>(a, b, block); break;
    default: combine<kMaxRowsLog>(a, b, block); break;
    }
    io::Writer out;
    char* const text = text_buffer<fields::kTextBytes>(b, size * sizeof(std::uint32_t));  // b is dead
    for (std::size_t r = 0; r < rows; r += band_rows)
        print_band(out, a + row_offset(r, block), band_rows, block, std::min(band_rows * block, total), block_log, text);
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here), and _exit skips their teardown.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
