// Polynomial products modulo 998244353 on std::vector, on lib/ntt alone: small enough, bundled
// by showcase/bundle.py --compact, for judges that limit source size (Codeforces 64 KB). x86-64
// with AVX2; Linux, macOS, or Windows when bundled by showcase/bundle.py.
//
//   #include <...>                       // standard headers first: this one enables AVX2 after it
//   #include "lib/easy/multiply.hpp"
//   easy::Poly c = easy::multiply(a, b);  // a * b, all n + m - 1 coefficients, canonical
//   easy::mul(a, b), easy::power(a, e)    // scalars mod P
//
// Products with a factor of at most 32 coefficients are schoolbook. Longer ones use one
// ntt::Convolution per transform length, kept for later products of that length (never freed),
// so repeated products fault in no new memory.
#pragma once

#include "lib/easy/target.hpp"
#include "lib/ntt/ntt.hpp"

namespace easy {

inline constexpr std::uint32_t kMod = ntt::kModulus;

using Poly = std::vector<std::uint32_t>;
using Span = std::span<const std::uint32_t>;

// a b mod P and a^e mod P, for canonical a, b.
constexpr std::uint32_t mul(std::uint32_t a, std::uint32_t b) { return std::uint32_t(std::uint64_t(a) * b % kMod); }

constexpr std::uint32_t power(std::uint32_t a, std::uint64_t e) {
    std::uint32_t r = 1;
    for (; e; e >>= 1, a = mul(a, a))
        if (e & 1) r = mul(r, a);
    return r;
}

namespace detail {

// Schoolbook product, for a short factor. Sums of 16 products < 16 P^2 < 2^64.
inline Poly multiply_naive(Span a, Span b) {
    if (a.size() < b.size()) std::swap(a, b);
    std::vector<std::uint64_t> sum(a.size() + b.size() - 1);
    for (std::size_t j0 = 0; j0 < b.size(); j0 += 16) {
        const std::size_t j1 = std::min(b.size(), j0 + 16);
        for (std::size_t j = j0; j < j1; ++j)
            for (std::size_t i = 0; i < a.size(); ++i) sum[i + j] += std::uint64_t(a[i]) * b[j];
        for (auto& s : sum) s %= kMod;
    }
    return Poly(sum.begin(), sum.end());
}

}  // namespace detail

inline Poly multiply(Span a, Span b) {
    if (a.empty() || b.empty()) return {};
    if (a.size() < b.size()) std::swap(a, b);  // b fills at most half the transform
    const std::size_t n = a.size() + b.size() - 1;
    if (b.size() <= 32 || n <= 64) return detail::multiply_naive(a, b);
    // Convolution(len - 1, 2) has length len and takes any a of up to len and b of up to len / 2
    // coefficients (a is not assumed sparse); multiply() may run again after a and b are refilled.
    const int lg = std::max(6, int(std::bit_width(n - 1)));
    const std::size_t len = std::size_t(1) << lg;
    static std::unique_ptr<ntt::Convolution> cache[ntt::kMaxLog + 1];
    auto& conv = cache[lg];
    if (!conv) conv = std::make_unique<ntt::Convolution>(len - 1, 2);
    std::fill(std::copy(a.begin(), a.end(), conv->a()), conv->a() + len, 0);
    std::fill(std::copy(b.begin(), b.end(), conv->b()), conv->b() + len, 0);
    const std::uint32_t* c = conv->multiply();
    return Poly(c, c + n);
}

}  // namespace easy
