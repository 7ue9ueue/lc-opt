// Baseline for solution.cpp: the same algorithm (blocks of a, overlap-save products, prefix sums
// along diagonals, ends by XOR and popcount) with a textbook NTT in place of lib/easy.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <span>
#include <string_view>
#include <vector>

#include <cstdio>
#include <string>

#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

namespace {

constexpr std::uint32_t kMod = 998244353;
using Poly = std::vector<std::uint32_t>;
using Span = std::span<const std::uint32_t>;

std::uint32_t power(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1;
    for (; e; e >>= 1, a = a * a % kMod)
        if (e & 1) r = r * a % kMod;
    return std::uint32_t(r);
}

// Textbook NTT: bit reversal, then radix-2 levels with a precomputed root table, products % P.
void transform(Poly& f, bool inverse) {
    const std::size_t n = f.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(f[i], f[j]);
    }
    static Poly roots{0, 1};  // roots[h + j] = w_(2h)^j for j < h
    while (roots.size() < n) {
        const std::size_t h = roots.size();
        roots.resize(2 * h);
        const std::uint64_t w = power(3, (kMod - 1) / (2 * h));
        for (std::size_t j = h; j < 2 * h; ++j) roots[j] = j % 2 ? std::uint32_t(roots[j / 2] * w % kMod) : roots[j / 2];
    }
    for (std::size_t h = 1; h < n; h *= 2)
        for (std::size_t i = 0; i < n; i += 2 * h)
            for (std::size_t j = 0; j < h; ++j) {
                const std::uint32_t u = f[i + j], v = std::uint32_t(std::uint64_t(f[i + j + h]) * roots[h + j] % kMod);
                f[i + j] = u + v >= kMod ? u + v - kMod : u + v;
                f[i + j + h] = u >= v ? u - v : u + kMod - v;
            }
    if (inverse) {
        std::reverse(f.begin() + 1, f.end());
        const std::uint64_t scale = power(n, kMod - 2);
        for (auto& x : f) x = std::uint32_t(x * scale % kMod);
    }
}

Poly multiply(Span a, Span b) {
    const std::size_t len = std::bit_ceil(a.size() + b.size() - 1);
    Poly x(len), y(len);
    std::copy(a.begin(), a.end(), x.begin());
    std::copy(b.begin(), b.end(), y.begin());
    transform(x, false);
    transform(y, false);
    for (std::size_t i = 0; i < len; ++i) x[i] = std::uint32_t(std::uint64_t(x[i]) * y[i] % kMod);
    transform(x, true);
    x.resize(a.size() + b.size() - 1);
    return x;
}


// Hamming distances between bit ranges of a and b, 256 bits per step.
class BitPairs {
public:
    // a_ holds bit i of a at bit i % 64 of word i / 64; b_ the same for 64 zero bits, then b.
    BitPairs(std::string_view a, std::string_view b) : a_(a.size() / 64 + 1), b_(b.size() / 64 + 3) {
        for (std::size_t i = 0; i < a.size(); ++i) a_[i / 64] |= std::uint64_t(a[i] == '1') << (i % 64);
        for (std::size_t i = 0; i < b.size(); ++i) b_[i / 64 + 1] |= std::uint64_t(b[i] == '1') << (i % 64);
    }

    // Hamming distance of a[p, p + len) and b[r, r + len): word k of a from p / 64 on against
    // bits [start + 64 k, start + 64 k + 64) of b_, start >= 1.
    std::uint32_t distance(std::size_t p, std::size_t r, std::size_t len) const {
        if (len == 0) return 0;
        const std::size_t count = (p + len - 1) / 64 - p / 64, start = p / 64 * 64 + r - p + 64;
        const std::uint64_t* x = a_.data() + p / 64;
        const std::uint64_t* y = b_.data() + start / 64;
        const unsigned shift = start % 64;
        const std::uint64_t first = ~std::uint64_t(0) << (p % 64), last = ~std::uint64_t(0) >> (63 - (p + len - 1) % 64);
        if (count == 0) return popcount((x[0] ^ word(y, shift)) & first & last);
        return popcount((x[0] ^ word(y, shift)) & first) + xor_popcount(x + 1, y + 1, shift, count - 1) +
               popcount((x[count] ^ word(y + count, shift)) & last);
    }

private:
    static std::uint32_t popcount(std::uint64_t x) { return std::uint32_t(__builtin_popcountll(x)); }

    // Bits [shift, shift + 64) of y[0], y[1].
    static std::uint64_t word(const std::uint64_t* y, unsigned shift) { return (y[0] >> shift) | (y[1] << 1 << (63 - shift)); }

    // Bits set in x[k] ^ word(y + k, shift) over k < count; nibble table, 4 words per step.
    static std::uint32_t xor_popcount(const std::uint64_t* x, const std::uint64_t* y, unsigned shift, std::size_t count) {
        const __m256i table = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,  //
                                               0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        const __m256i nibble = _mm256_set1_epi8(0x0f), zero = _mm256_setzero_si256();
        const __m128i right = _mm_cvtsi32_si128(int(shift)), left = _mm_cvtsi32_si128(int(64 - shift));  // 64: 0
        const auto load = [](const std::uint64_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        __m256i total = zero;  // four 64-bit sums
        std::size_t k = 0;
        for (; k + 4 <= count; k += 4) {
            const __m256i yk = _mm256_or_si256(_mm256_srl_epi64(load(y + k), right), _mm256_sll_epi64(load(y + k + 1), left));
            const __m256i v = _mm256_xor_si256(load(x + k), yk);
            const __m256i bytes = _mm256_add_epi8(_mm256_shuffle_epi8(table, _mm256_and_si256(v, nibble)),
                                                  _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble)));
            total = _mm256_add_epi64(total, _mm256_sad_epu8(bytes, zero));
        }
        const __m128i half = _mm_add_epi64(_mm256_castsi256_si128(total), _mm256_extracti128_si256(total, 1));
        std::uint32_t sum = std::uint32_t(_mm_cvtsi128_si64(half) + _mm_extract_epi64(half, 1));
        for (; k < count; ++k) sum += popcount(x[k] ^ word(y + k, shift));
        return sum;
    }

    std::vector<std::uint64_t> a_, b_;
};

// Block size: balances the products, about n m / B butterflies, against q B / 64 popcount words.
// At least n m / 2^24 (prefix sums within 64 MiB), at most 2^17.
std::size_t block_size(std::size_t n, std::size_t m, std::size_t q) {
    constexpr double kCost = 16384;  // fastest of 1024 .. 65536 on lc-amd (notes.md)
    const std::size_t balanced = std::size_t(std::sqrt(double(n) * double(m) / double(q + 1) * kCost));
    return std::clamp<std::size_t>(balanced, std::max<std::size_t>(1, n * m >> 24), std::size_t(1) << 17);
}

// out[i] = previous[i] + (B - c_i) / 2 for correlations c_i given mod P, |c_i| <= B < P / 2.
void add_distances(std::span<const std::uint32_t> c, std::uint32_t block, const std::uint32_t* previous, std::uint32_t* out) {
    // B - c_i + P reduced mod P is B - c_i in [0, 2B]: min(v, v - P) unsigned.
    const __m256i b_plus_p = _mm256_set1_epi32(int(block + kMod)), p = _mm256_set1_epi32(int(kMod));
    std::size_t i = 0;
    for (; i + 8 <= c.size(); i += 8) {
        const __m256i v = _mm256_sub_epi32(b_plus_p, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c.data() + i)));
        const __m256i distance = _mm256_srli_epi32(_mm256_min_epu32(v, _mm256_sub_epi32(v, p)), 1);
        const __m256i sum = _mm256_add_epi32(distance, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(previous + i)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), sum);
    }
    for (; i < c.size(); ++i) {
        const std::uint32_t v = block + kMod - c[i];
        out[i] = previous[i] + std::min(v, v - kMod) / 2;
    }
}

// Prefix sums along diagonals: prefix[(k + 1)(m + 1 + B) + d] is the sum over blocks k' <= k
// with k'B + d >= 0 of the distance of a[k'B, k'B + B) and b[k'B + d, k'B + d + B), for the n / B
// whole blocks k of a and -B <= kB + d <= m - B (0 if kB + d < 0). Row k + 1 (m + 1 entries) holds
// B zeros, then j = kB + d = 0 .. m - B; row 0 is zero.
// Overlap-save: a product of the reversed block and b[sS, sS + S + B - 1) has length
// C = kChunk 2^ceil(log2 B) and gives the S = C - 2B + 2 windows from sS on.
std::vector<std::uint32_t> diagonal_prefix(std::string_view a, std::string_view b, std::size_t block) {
    constexpr std::size_t kChunk = 8;
    const std::size_t n = a.size(), m = b.size(), width = m - block + 1, stride = m + 1;
    const std::size_t step = kChunk * std::bit_ceil(block) - 2 * block + 2;
    const auto sign = [](char c) { return c == '1' ? 1u : kMod - 1; };
    Poly signs(m), reversed(block);
    std::transform(b.begin(), b.end(), signs.begin(), sign);
    std::vector<std::uint32_t> prefix((n / block + 1) * stride);
    for (std::size_t k = 0; k < n / block; ++k) {
        std::transform(a.begin() + std::ptrdiff_t(k * block), a.begin() + std::ptrdiff_t((k + 1) * block), reversed.rbegin(), sign);
        std::uint32_t* const row = prefix.data() + (k + 1) * stride + block;  // row[j]: j = kB + d
        for (std::size_t s = 0; s * step < width; ++s) {
            const std::size_t windows = std::min(step, width - s * step);
            const Poly product = multiply(reversed, Span(signs).subspan(s * step, windows + block - 1));
            // product[B - 1 + i] = sum over l < B of a_(kB + l) b_(sS + i + l); (k - 1, j - B) precedes (k, j).
            std::uint32_t* const out = row + s * step;
            add_distances(Span(product).subspan(block - 1, windows), std::uint32_t(block), out - stride - block, out);
        }
    }
    return prefix;
}

}  // namespace

// Whole input in one buffer; tokens split at bytes <= ' '.
class Input {
public:
    Input() {
        char chunk[1 << 16];
        for (std::size_t got; (got = std::fread(chunk, 1, sizeof chunk, stdin)) > 0;) data_.append(chunk, got);
        data_.push_back('\0');
    }
    std::string_view token() {
        while (data_[pos_] && data_[pos_] <= ' ') ++pos_;
        const std::size_t start = pos_;
        while (data_[pos_] > ' ') ++pos_;
        return {data_.data() + start, pos_ - start};
    }
    std::size_t number() {
        std::size_t x = 0;
        for (const char c : token()) x = x * 10 + std::size_t(c - '0');
        return x;
    }

private:
    std::string data_;
    std::size_t pos_ = 0;
};

int main() {
    Input in;
    const std::string_view a = in.token(), b = in.token();
    const std::size_t n = a.size(), m = b.size(), q = in.number();
    const std::size_t block = block_size(n, m, q), diagonal_step = m + 1 + block;
    const std::vector<std::uint32_t> prefix =
        block <= std::min(n, m) ? diagonal_prefix(a, b, block) : std::vector<std::uint32_t>();
    const BitPairs bits(a, b);
    std::string out;
    for (std::size_t i = 0; i < q; ++i) {
        const std::size_t p1 = in.number(), p2 = in.number(), len = in.number();
        const std::size_t first = (p1 + block - 1) / block, last = (p1 + len) / block;  // whole blocks [first, last)
        std::uint32_t d;
        if (first >= last) {
            d = bits.distance(p1, p2, len);
        } else {
            const std::uint32_t* diagonal = prefix.data() + std::ptrdiff_t(p2) - std::ptrdiff_t(p1);
            const std::size_t lo = first * block, hi = last * block;
            d = diagonal[last * diagonal_step] - diagonal[first * diagonal_step] + bits.distance(p1, p2, lo - p1) +
                bits.distance(hi, p2 + (hi - p1), p1 + len - hi);
        }
        out += std::to_string(d);
        out += '\n';
    }
    std::fwrite(out.data(), 1, out.size(), stdout);
}
