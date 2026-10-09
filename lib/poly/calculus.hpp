// Coefficient-wise operations on power series modulo 998244353: derivative and division by
// consecutive integers (integration). x86-64 with AVX2. Log: lib/poly/notes.md.
//
//   poly::derivative(f, d);              // d[i] = (i + 1) f[i + 1]; d may be f (in place)
//   poly::divide_by_index(a, first, q);  // q[i] = a[i] / (first + i)
//   poly::divide_by_index(a, 1, q.subspan(1)), q[0] = 0;  // q = the integral of a
//
// Coefficients in [0, P), spans of any size and alignment.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/poly/transform.hpp"

namespace poly {

namespace detail {

inline Vec load_unaligned(const std::uint32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
inline void store_unaligned(std::uint32_t* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }

// a b / 2^32 mod P in [0, 2P), for a b < 2^32 P (for example a, b < 2P). Each 64-bit sum
// a b + m P stays below 2^33 P.
inline Vec montgomery(Vec a, Vec b) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    Vec even = _mm256_mul_epu32(a, b);
    Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
    even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
    odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
    return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
}

// The integers [k, k + 8) times 2^32 mod P, for multiplying by them with montgomery().
class Indices {
public:
    explicit Indices(std::size_t k) {
        alignas(32) std::uint32_t x[8];
        for (std::uint32_t i = 0; i < 8; ++i) x[i] = std::uint32_t((std::uint64_t(k + i) << 32) % kP);
        value_ = load(x);
    }

    Vec value() const { return value_; }
    void next() { value_ = reduce(add(value_, broadcast(kStep)), kP); }  // k += 8

private:
    static constexpr std::uint32_t kStep = std::uint32_t((std::uint64_t(8) << 32) % kP);
    Vec value_;
};

inline std::uint32_t scalar_inverse(std::uint32_t x) { return ntt::detail::power(x, kP - 2); }

// out[i] = a(i) / (first + i) for i < out.size(), where a(i) is lane i % 8 of a(8 (i / 8)), a
// vector in [0, 2P). a(j) may read past the end of out's range (lanes discarded), but not the
// words of out. Montgomery's batch inversion in 4 interleaved vector chains: out first holds
// prefix products of blocks of 32 integers, lane by lane, each with a factor 2^-32 per step;
// the factors cancel in the backward pass. first >= 1, first + out.size() <= P.
template <class A>
void divide_by_index(std::size_t first, std::span<std::uint32_t> out, const A& a) {
    constexpr std::size_t kLanes = 32;
    const std::size_t n = out.size();
    if (n == 0) return;
    const std::size_t blocks = (n + kLanes - 1) / kLanes, last = kLanes * (blocks - 1);
    const Vec one = broadcast(1), step = broadcast(kLanes);
    // The integers of block b: first + 32 b + lane; lanes past n count as 1.
    Vec x[4];
    for (int v = 0; v < 4; ++v)
        x[v] = _mm256_add_epi32(broadcast(std::uint32_t(first + 8 * v)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    const auto valid = [n](std::size_t at) {  // lanes of the vector at index at that are < n
        const Vec lanes = _mm256_add_epi32(broadcast(std::uint32_t(at)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        return _mm256_cmpgt_epi32(broadcast(std::uint32_t(n)), lanes);  // n < 2^31
    };
    const auto masked = [&](Vec value, std::size_t at) { return at + 8 <= n ? value : _mm256_blendv_epi8(one, value, valid(at)); };

    Vec prefix[4];
    for (int v = 0; v < 4; ++v) prefix[v] = masked(x[v], 8 * v);
    for (std::size_t b = kLanes; b < n; b += kLanes) {
        for (int v = 0; v < 4; ++v) {
            store_unaligned(out.data() + b - kLanes + 8 * v, prefix[v]);
            x[v] = _mm256_add_epi32(x[v], step);
            prefix[v] = montgomery(prefix[v], masked(x[v], b + 8 * v));
        }
    }

    // inverse[lane] = 2^32 / (prefix product of the last block), by one more batch inversion.
    alignas(32) std::uint32_t total[kLanes], inverse[kLanes];
    for (int v = 0; v < 4; ++v) store(total + 8 * v, canonical(prefix[v]));
    std::uint32_t running = 1;
    for (std::size_t l = 0; l < kLanes; ++l) {
        inverse[l] = running;
        running = ntt::detail::multiply_mod(running, total[l]);
    }
    running = ntt::detail::multiply_mod(scalar_inverse(running), kR);
    for (std::size_t l = kLanes; l-- > 0;) {
        inverse[l] = ntt::detail::multiply_mod(inverse[l], running);
        running = ntt::detail::multiply_mod(running, total[l]);
    }

    // Backward: 1 / x = prefix(b - 1) inverse(b) 2^-32, inverse(b - 1) = inverse(b) x 2^-32.
    // inverse carries an extra 2^32, which the product with a(i) removes.
    Vec inv[4];
    for (int v = 0; v < 4; ++v) inv[v] = load(inverse + 8 * v);
    const auto put = [&](std::size_t at, Vec reciprocal) {
        const Vec q = canonical(montgomery(reciprocal, a(at)));
        if (at + 8 <= n) return store_unaligned(out.data() + at, q);
        _mm256_maskstore_epi32(reinterpret_cast<int*>(out.data() + at), valid(at), q);
    };
    for (std::size_t b = last; b > 0; b -= kLanes) {
        for (int v = 0; v < 4; ++v) {
            const std::size_t at = b + 8 * v;
            if (at < n) put(at, montgomery(load_unaligned(out.data() + at - kLanes), inv[v]));
            inv[v] = montgomery(inv[v], masked(x[v], at));
            x[v] = _mm256_sub_epi32(x[v], step);
        }
    }
    for (int v = 0; v < 4; ++v)
        if (8 * std::size_t(v) < n) put(8 * v, inv[v]);
}

}  // namespace detail

// d[i] = (i + 1) f[i + 1] for i < d.size(), with f[j] = 0 for j >= f.size(). d may start at f
// (in place); otherwise the two must not overlap.
inline void derivative(std::span<const std::uint32_t> f, std::span<std::uint32_t> d) {
    using namespace detail;
    const std::size_t n = d.size(), full = std::min(n, f.size() > 0 ? f.size() - 1 : 0) / 8 * 8;
    Indices index(1);
    for (std::size_t i = 0; i < full; i += 8, index.next())
        store_unaligned(d.data() + i, canonical(montgomery(load_unaligned(f.data() + i + 1), index.value())));
    for (std::size_t i = full; i < n; ++i)
        d[i] = i + 1 < f.size() ? ntt::detail::multiply_mod(std::uint32_t(i + 1), f[i + 1]) : 0;
}

// q[i] = a[i] / (first + i) for i < q.size() = a.size(). first >= 1, first + a.size() <= P.
// a and q must not overlap.
inline void divide_by_index(std::span<const std::uint32_t> a, std::size_t first, std::span<std::uint32_t> q) {
    if (a.size() != q.size() || first == 0 || first + a.size() > kModulus) std::abort();
    const std::size_t n = a.size();
    detail::divide_by_index(first, q, [a, n](std::size_t at) {
        if (at + 8 <= n) return detail::load_unaligned(a.data() + at);
        alignas(32) std::uint32_t rest[8] = {};
        std::copy(a.begin() + at, a.end(), rest);
        return detail::load(rest);
    });
}

}  // namespace poly
