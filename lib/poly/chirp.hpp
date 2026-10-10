// Chirps modulo 998244353: x_k = c s^k q^t(k) with t(k) = k (k - 1) / 2, the pointwise factors of
// the chirp z-transform (i j = t(i + j) - t(i) - t(j), so f(a r^i) = r^-t(i) sum_j c_j a^j r^-t(j)
// r^t(i + j)). x86-64 with AVX2. Log: lib/poly/notes.md.
//
//   poly::chirp(c, s, q, out);           // out[k] = x_k for k < out.size()
//   poly::multiply_chirp(c, s, q, f);    // f[k] = f[k] x_k for k < f.size()
//   poly::detail::Chirp terms(c, s, q);  // x_k, 32 at a time: terms[v] for v < 4, then terms.next()
//
// c, s, q and the coefficients in [0, P); spans of any size and alignment.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/calculus.hpp"

namespace poly {

namespace detail {

// Terms x_k = c s^k q^t(k), k = 0, 1, ..., in [0, 2P), 32 per step: x_(k+32) = x_k g_k with
// g_k = s^32 q^(32 k + 496) and g_(k+32) = g_k q^1024. g is kept times 2^32, so a step is one
// montgomery() and one Shoup product per vector. For x_k f / 2^32 by montgomery(f, x_k), pass
// c 2^32 mod P as c.
class Chirp {
public:
    Chirp(std::uint32_t c, std::uint32_t s, std::uint32_t q) : step_(ntt::detail::power(q, 1024)) {
        using ntt::detail::multiply_mod;
        using ntt::detail::power;
        alignas(32) std::uint32_t x[32], g[32];
        const std::uint32_t q32 = power(q, 32);
        std::uint32_t term = c, ratio = s;  // ratio = x_(k+1) / x_k = s q^k
        std::uint32_t g_k = multiply_mod(multiply_mod(power(s, 32), power(q, 496)), kR);
        for (int k = 0; k < 32; ++k) {
            x[k] = term, g[k] = g_k;
            term = multiply_mod(term, ratio), ratio = multiply_mod(ratio, q), g_k = multiply_mod(g_k, q32);
        }
        for (int v = 0; v < 4; ++v) x_[v] = load(x + 8 * v), g_[v] = load(g + 8 * v);
    }

    // Vector v of the current 32 terms: x_(32 j + 8 v + l) in lane l after j steps.
    Vec operator[](int v) const { return x_[v]; }

    void next() {
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) {
            x_[v] = montgomery(x_[v], g_[v]);
            g_[v] = times(g_[v], step_);
        }
    }

private:
    Vec x_[4], g_[4];
    Factor step_;
};

// out[k] = put(k, x_k) for k < out.size(), put(k, x) a vector in [0, P) for the 8 terms at k.
template <class Put>
void chirp_pass(Chirp terms, std::span<std::uint32_t> out, Put put) {
    const std::size_t n = out.size(), full = n / 32 * 32;
    std::size_t k = 0;
    for (; k < full; k += 32, terms.next())
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) store_unaligned(out.data() + k + 8 * v, put(k + 8 * v, terms[v]));
    if (k == n) return;
    alignas(32) std::uint32_t rest[32];
    for (int v = 0; v < 4 && k + 8 * v < n; ++v) store(rest + 8 * v, put(k + 8 * v, terms[v]));
    std::copy(rest, rest + (n - k), out.begin() + k);
}

}  // namespace detail

// out[k] = c s^k q^t(k) for k < out.size().
inline void chirp(std::uint32_t c, std::uint32_t s, std::uint32_t q, std::span<std::uint32_t> out) {
    using namespace detail;
    chirp_pass(Chirp(c, s, q), out, [](std::size_t, Vec x) { return reduce(x, kP); });
}

// f[k] = f[k] c s^k q^t(k) for k < f.size().
inline void multiply_chirp(std::uint32_t c, std::uint32_t s, std::uint32_t q, std::span<std::uint32_t> f) {
    using namespace detail;
    const std::size_t n = f.size();
    const auto factor = [f, n](std::size_t k) {
        if (k + 8 <= n) return load_unaligned(f.data() + k);
        alignas(32) std::uint32_t rest[8] = {};
        std::copy(f.begin() + k, f.end(), rest);
        return load(rest);
    };
    chirp_pass(Chirp(ntt::detail::multiply_mod(c, kR), s, q), f,
               [&factor](std::size_t k, Vec x) { return reduce(montgomery(factor(k), x), kP); });
}

}  // namespace poly
