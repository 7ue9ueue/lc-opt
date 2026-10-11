// Power series modulo 998244353 on std::vector: a small API over lib/poly for contest code.
// x86-64 with AVX2; Linux, macOS, or Windows when bundled by showcase/bundle.py.
//
//   #include <...>                // standard headers first: this one enables AVX2 after it
//   #include "lib/easy/poly.hpp"
//   easy::Poly c = easy::multiply(a, b);       // a * b, all n + m - 1 coefficients
//   easy::Poly g = easy::inverse(f, n);        // 1 / f mod x^n, f[0] != 0
//   easy::Poly e = easy::exp(f, n);            // f[0] == 0
//   easy::Poly l = easy::log(f, n);            // f[0] == 1
//   easy::Poly p = easy::pow(f, k, n);         // f^k mod x^n, any f, k < 2^64
//   std::vector<std::uint32_t> v = easy::evaluate(f, points);
//   easy::Poly c = easy::interpolate(points, values);  // distinct points
//   easy::mul(a, b), easy::power(a, e)          // scalars mod P
//
// Coefficients are canonical (< 998244353) in and out. An empty f stands for 0. multiply comes
// from lib/easy/multiply.hpp. inverse, exp, log and pow each keep their memory for the largest
// call so far (never freed), so repeated calls fault in no new pages; evaluate and interpolate
// map their own and return it.
#pragma once

#include "lib/easy/multiply.hpp"
#include "lib/poly/evaluation.hpp"
#include "lib/poly/exp.hpp"
#include "lib/poly/interpolation.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/pow.hpp"
#include "lib/poly/transform.hpp"

namespace easy {

namespace detail {

// Transform tables and K spans kept between calls of one kind, grown to the largest request so
// far and never shrunk, so repeated calls fault in no new memory. get() zeroes the first
// sizes[k] words of span k (and its padding), as a fresh arena would give them.
template <std::size_t K>
class Cache {
public:
    struct Lease {
        const poly::Transform& t;
        std::array<std::span<std::uint32_t>, K> spans;
    };

    Lease get(int lg, const std::array<std::size_t, K>& sizes, bool zero = true) {
        bool fits = lg <= lg_;
        for (std::size_t k = 0; k < K; ++k) fits &= sizes[k] <= capacity_[k];
        if (!fits) grow(std::max(lg, lg_), sizes);  // fresh memory is zero
        Lease lease{*transform_, {}};
        for (std::size_t k = 0; k < K; ++k) {
            if (zero && fits) std::fill_n(data_[k], poly::Arena::footprint(sizes[k]), 0);
            lease.spans[k] = {data_[k], sizes[k]};
        }
        return lease;
    }

private:
    void grow(int lg, const std::array<std::size_t, K>& sizes) {
        transform_.reset();
        arena_.reset();
        std::size_t words = poly::Transform::words(lg);
        for (std::size_t k = 0; k < K; ++k) {
            capacity_[k] = std::max(capacity_[k], sizes[k]);
            words += poly::Arena::footprint(capacity_[k]);
        }
        arena_ = std::make_unique<poly::Arena>(words);
        transform_.emplace(*arena_, lg);
        for (std::size_t k = 0; k < K; ++k) data_[k] = arena_->take(capacity_[k]).data();
        lg_ = lg;
    }

    int lg_ = 0;
    std::array<std::size_t, K> capacity_{};
    std::array<std::uint32_t*, K> data_{};
    std::unique_ptr<poly::Arena> arena_;
    std::optional<poly::Transform> transform_;
};

}  // namespace detail

inline Poly inverse(Span f, std::size_t n) {
    if (n == 0) return {};
    static detail::Cache<3> cache;
    const auto [t, spans] = cache.get(poly::inverse_log(n), {n, n, poly::inverse_scratch(n)});
    const auto [in, out, scratch] = spans;
    std::copy_n(f.begin(), std::min(n, f.size()), in.begin());
    poly::inverse(t, in, out, scratch);
    return Poly(out.begin(), out.end());
}

inline Poly exp(Span f, std::size_t n) {
    if (n == 0) return {};
    static detail::Cache<2> cache;
    const auto [t, spans] = cache.get(poly::exp_log(n), {n, poly::exp_scratch(n)});
    const auto [g, scratch] = spans;
    std::copy_n(f.begin(), std::min(n, f.size()), g.begin());
    poly::exp(t, g, g, scratch);
    return Poly(g.begin(), g.end());
}

inline Poly log(Span f, std::size_t n) {
    if (n == 0) return {};
    static detail::Cache<2> cache;
    const auto [t, spans] = cache.get(poly::log_log(n), {n, poly::log_scratch(n)});
    const auto [g, scratch] = spans;
    std::copy_n(f.begin(), std::min(n, f.size()), g.begin());
    poly::log(t, g, g, scratch);
    return Poly(g.begin(), g.end());
}

inline Poly pow(Span f, std::uint64_t k, std::size_t n) {
    Poly g(n);
    if (n == 0) return g;
    if (k == 0) return g[0] = 1, g;
    const std::size_t z = std::size_t(std::find_if(f.begin(), f.end(), [](std::uint32_t c) { return c != 0; }) - f.begin());
    if (z == f.size() || (z > 0 && k > (n - 1) / z)) return g;  // f^k = 0 mod x^n
    const std::size_t shift = z * std::size_t(k), size = n - shift;
    static detail::Cache<2> cache;
    const auto [t, spans] = cache.get(poly::power_log(size), {size, poly::power_scratch(size)});
    const auto [u, scratch] = spans;
    const Span tail = f.subspan(z);
    std::copy_n(tail.begin(), std::min(size, tail.size()), u.begin());
    const std::uint32_t c = power(u[0], k % (kMod - 1));
    poly::power(t, u, std::uint32_t(k % kMod), c, u, scratch);
    std::copy(u.begin(), u.end(), g.begin() + std::ptrdiff_t(shift));
    return g;
}

inline std::vector<std::uint32_t> evaluate(Span f, Span points) {
    std::vector<std::uint32_t> values(points.size());
    if (points.empty() || f.empty()) return values;
    poly::Arena arena(poly::evaluate_words(f.size(), points.size()) + 64);
    poly::evaluate(arena, f, points, values);
    return values;
}

inline Poly interpolate(Span points, Span values) {
    Poly c(points.size());
    if (points.empty()) return c;
    poly::Arena arena(poly::interpolate_words(points.size()));
    poly::interpolate(arena, points, values, c);
    return c;
}

}  // namespace easy
