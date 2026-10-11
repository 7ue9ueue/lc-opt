// Codeforces 472G Design Tutorial: Increase the Constraints. Binary strings a, b of length
// n, m <= 2*10^5 and q <= 4*10^5 queries (p1, p2, len): the Hamming distance of a[p1, p1 + len)
// and b[p2, p2 + len). 7 s, 256 MB.
//
// The editorial's sqrt decomposition with FFT: cut a into blocks of B bits. One cyclic convolution
// of length L = 2^ceil(log2 m) per block (bits as +1/-1, so the correlation is B - 2 * distance)
// gives the block's distance to every window of b, stored as a row. A query adds one entry per
// whole block and counts its two ends (< B bits each) by XOR and popcount.
// O((n / B) L log L + q (n / B + B / 64)), memory O((n / B) m). Accepted solutions are mostly a
// bitset brute force, O(q len / 64); the FFT route is the heavy one here.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <span>
#include <string_view>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/poly.hpp"

namespace {

// Hamming distances between bit ranges of a and b, 64 bits per step.
class BitPairs {
public:
    BitPairs(std::string_view a, std::string_view b) : a_(a.size() / 64 + 1), width_(b.size() / 64 + 2) {
        for (std::size_t i = 0; i < a.size(); ++i) a_[i / 64] |= std::uint64_t(a[i] == '1') << (i % 64);
        // b' = 64 zero bits, then b, then zeros; copy t holds bits [64 j + t, 64 j + t + 64) of b' as word j.
        std::vector<std::uint64_t> padded(width_ + 1);
        for (std::size_t i = 0; i < b.size(); ++i) padded[i / 64 + 1] |= std::uint64_t(b[i] == '1') << (i % 64);
        copies_.resize(64 * width_);
        for (std::size_t j = 0; j < width_; ++j) copies_[j] = padded[j];
        for (std::size_t t = 1; t < 64; ++t)
            for (std::size_t j = 0; j < width_; ++j)
                copies_[t * width_ + j] = (padded[j] >> t) | (padded[j + 1] << (64 - t));
    }

    // Hamming distance of a[p, p + len) and b[r, r + len).
    std::uint32_t distance(std::size_t p, std::size_t r, std::size_t len) const {
        if (len == 0) return 0;
        const std::size_t count = (p + len - 1) / 64 - p / 64, start = p / 64 * 64 + r - p + 64;  // start >= 1
        const std::uint64_t* x = a_.data() + p / 64;
        const std::uint64_t* y = copies_.data() + start % 64 * width_ + start / 64;
        const std::uint64_t first = ~std::uint64_t(0) << (p % 64), last = ~std::uint64_t(0) >> (63 - (p + len - 1) % 64);
        if (count == 0) return popcount((x[0] ^ y[0]) & first & last);
        return popcount((x[0] ^ y[0]) & first) + xor_popcount(x + 1, y + 1, count - 1) +
               popcount((x[count] ^ y[count]) & last);
    }

private:
    static std::uint32_t popcount(std::uint64_t x) { return std::uint32_t(__builtin_popcountll(x)); }

    // Bits set in x[k] ^ y[k] over k < count; 4 words per step by a nibble table.
    static std::uint32_t xor_popcount(const std::uint64_t* x, const std::uint64_t* y, std::size_t count) {
        const __m256i table = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,  //
                                               0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        const __m256i nibble = _mm256_set1_epi8(0x0f), zero = _mm256_setzero_si256();
        __m256i total = zero;  // four 64-bit sums
        std::size_t k = 0;
        for (; k + 4 <= count; k += 4) {
            const __m256i v = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k)),
                                               _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + k)));
            const __m256i bytes = _mm256_add_epi8(_mm256_shuffle_epi8(table, _mm256_and_si256(v, nibble)),
                                                  _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble)));
            total = _mm256_add_epi64(total, _mm256_sad_epu8(bytes, zero));
        }
        const __m128i half = _mm_add_epi64(_mm256_castsi256_si128(total), _mm256_extracti128_si256(total, 1));
        std::uint32_t sum = std::uint32_t(_mm_cvtsi128_si64(half) + _mm_extract_epi64(half, 1));
        for (; k < count; ++k) sum += popcount(x[k] ^ y[k]);
        return sum;
    }

    std::vector<std::uint64_t> a_;
    std::size_t width_;
    std::vector<std::uint64_t> copies_;
};

// Block size: balances the transforms, about n m / B butterflies, against q B / 64 popcount
// words. At least n m / 2^24 (rows within 64 MiB), at most 2^18 (>= n).
std::size_t block_size(std::size_t n, std::size_t m, std::size_t q) {
    constexpr double kCost = 16;  // a popcount word over a transform butterfly, measured
    const std::size_t balanced = std::size_t(std::sqrt(double(n) * double(m) / double(q + 1) * kCost));
    return std::clamp<std::size_t>(balanced, std::max<std::size_t>(1, n * m >> 24), std::size_t(1) << 18);
}

// out[i] = previous[i] + (B - c_i) / 2 for correlations c_i given mod P, |c_i| <= B < P / 2.
void add_distances(std::span<const std::uint32_t> c, std::uint32_t block, const std::uint32_t* previous, std::uint32_t* out) {
    // B - c_i + P reduced mod P is B - c_i in [0, 2B]: min(v, v - P) unsigned.
    std::size_t i = 0;
    const __m256i b_plus_p = _mm256_set1_epi32(int(block + easy::kMod)), p = _mm256_set1_epi32(int(easy::kMod));
    for (; i + 8 <= c.size(); i += 8) {
        const __m256i v = _mm256_sub_epi32(b_plus_p, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c.data() + i)));
        const __m256i distance = _mm256_srli_epi32(_mm256_min_epu32(v, _mm256_sub_epi32(v, p)), 1);
        const __m256i sum = _mm256_add_epi32(distance, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), sum);
    }
    for (; i < c.size(); ++i) {
        const std::uint32_t v = block + easy::kMod - c[i];
        out[i] = previous[i] + std::min(v, v - easy::kMod) / 2;
    }
}

// Prefix sums along diagonals: prefix[(k + 1)(m + 1 + B) + d] is the sum over blocks k' <= k
// with k'B + d >= 0 of the distance of a[k'B, k'B + B) and b[k'B + d, k'B + d + B), for the n / B
// whole blocks k of a and -B <= kB + d <= m - B (0 if kB + d < 0). Row k + 1 (m + 1 entries) holds
// B zeros, then j = kB + d = 0 .. m - B; row 0 is zero.
// Overlap-save: b in chunks of length C = kChunk 2^ceil(log2 B), each giving S = C - B + 1
// windows; a block's transform meets each chunk's in one inverse transform.
std::vector<std::uint32_t> diagonal_prefix(std::string_view a, std::string_view b, std::size_t block) {
    constexpr std::size_t kChunk = 4;
    const std::size_t n = a.size(), m = b.size(), blocks = n / block, width = m - block + 1, stride = m + 1;
    const int lg = std::max(poly::Transform::kMinLog, int(std::bit_width(kChunk * std::bit_ceil(block) - 1)));
    const std::size_t len = std::size_t(1) << lg, step = len - block + 1, chunks = (width + step - 1) / step;
    poly::Arena arena(poly::Transform::words(lg) + poly::Arena::footprint(n) + poly::Arena::footprint(m) +
                      (chunks + 2) * poly::Arena::footprint(len));
    const poly::Transform t(arena, lg);
    const auto sign = [](char c) { return c == '1' ? 1u : easy::kMod - 1; };
    const auto reversed = arena.take(n), signs = arena.take(m), block_hat = arena.take(len), product = arena.take(len);
    for (std::size_t i = 0; i < n; ++i) reversed[i] = sign(a[n - 1 - i]);
    for (std::size_t i = 0; i < m; ++i) signs[i] = sign(b[i]);
    std::vector<std::span<std::uint32_t>> chunk_hat(chunks);
    for (std::size_t s = 0; s < chunks; ++s) {
        chunk_hat[s] = arena.take(len);
        t.forward(signs.subspan(s * step, std::min(len, m - s * step)), 0, chunk_hat[s]);
    }
    std::vector<std::uint32_t> prefix((blocks + 1) * stride);
    for (std::size_t k = 0; k < blocks; ++k) {
        t.forward(reversed.subspan(n - (k + 1) * block, block), 0, block_hat);
        std::uint32_t* const row = prefix.data() + (k + 1) * stride + block;  // row[j]: j = kB + d
        for (std::size_t s = 0; s < chunks; ++s) {
            // product[B - 1 + i] = sum over l < B of a_(kB + l) b_(sS + i + l), i < S: wraparound stays below B - 1.
            t.inverse_product(block_hat, chunk_hat[s], product);
            std::uint32_t* const out = row + s * step;
            add_distances(product.subspan(block - 1, std::min(step, width - s * step)), std::uint32_t(block),
                          out - stride - block, out);  // (k - 1, j - B) is (k, j)'s predecessor on the diagonal
        }
    }
    return prefix;
}

struct Query {
    std::uint32_t p1, p2, len;
};

// Query indices sorted by diagonal p2 - p1, so that prefix sums are read in order (counting sort).
std::vector<std::uint32_t> by_diagonal(std::span<const Query> queries, std::size_t n, std::size_t m) {
    std::vector<std::uint32_t> start(n + m + 1), order(queries.size());
    for (const Query& x : queries) ++start[x.p2 + n - x.p1 + 1];
    for (std::size_t d = 1; d <= n + m; ++d) start[d] += start[d - 1];
    for (std::size_t i = 0; i < queries.size(); ++i) order[start[queries[i].p2 + n - queries[i].p1]++] = std::uint32_t(i);
    return order;
}

}  // namespace

int main() {
    easy::Reader in;
    const std::string_view a = in.token(), b = in.token();
    const std::size_t n = a.size(), m = b.size();
    std::vector<Query> queries(in.read<std::uint32_t>());
    for (Query& x : queries) x.p1 = in.read<std::uint32_t>(), x.p2 = in.read<std::uint32_t>(), x.len = in.read<std::uint32_t>();

    const std::size_t block = block_size(n, m, queries.size());
    const std::vector<std::uint32_t> prefix =
        block <= std::min(n, m) ? diagonal_prefix(a, b, block) : std::vector<std::uint32_t>();
    const std::size_t diagonal_step = m + 1 + block;
    const BitPairs bits(a, b);
    std::vector<std::uint32_t> answer(queries.size());
    for (const std::uint32_t i : by_diagonal(queries, n, m)) {
        const std::size_t p1 = queries[i].p1, p2 = queries[i].p2, len = queries[i].len;
        const std::size_t first = (p1 + block - 1) / block, last = (p1 + len) / block;  // whole blocks [first, last)
        if (first >= last) {
            answer[i] = bits.distance(p1, p2, len);
            continue;
        }
        // Blocks k in [first, last) pair with b at kB + p2 - p1 >= -B.
        const std::uint32_t* diagonal = prefix.data() + std::ptrdiff_t(p2) - std::ptrdiff_t(p1);
        const std::uint32_t sum = diagonal[last * diagonal_step] - diagonal[first * diagonal_step];
        const std::size_t lo = first * block, hi = last * block;
        answer[i] = sum + bits.distance(p1, p2, lo - p1) + bits.distance(hi, p2 + (hi - p1), p1 + len - hi);
    }
    easy::Writer out;
    for (const std::uint32_t d : answer) out.write(d, '\n');
}
