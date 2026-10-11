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
// Coefficients are canonical (< 998244353) in and out. An empty f stands for 0. multiply keeps
// one workspace of transform buffers for the largest product so far (never shrunk), so repeated
// products of similar sizes fault in no new memory; the other calls map their own memory and
// return it.
#pragma once

// Every standard header lib/poly uses comes before the target pragma: GCC 13 and 14 fail to
// inline std::allocator's members into code compiled under it otherwise.
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include "lib/poly/evaluation.hpp"
#include "lib/poly/exp.hpp"
#include "lib/poly/interpolation.hpp"
#include "lib/poly/inverse.hpp"
#include "lib/poly/log.hpp"
#include "lib/poly/pow.hpp"
#include "lib/poly/transform.hpp"

namespace easy {

inline constexpr std::uint32_t kMod = poly::kModulus;

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

// Tables and two buffers for products of up to 2^lg coefficients.
class Workspace {
public:
    static Workspace& get() {
        static Workspace w;
        return w;
    }

    void reserve(int lg) {
        if (lg <= lg_) return;
        transform_.reset();
        arena_.reset();
        const std::size_t len = std::size_t(1) << lg;
        arena_ = std::make_unique<poly::Arena>(poly::Transform::words(lg) + 2 * poly::Arena::footprint(len));
        transform_.emplace(*arena_, lg);
        a_ = arena_->take(len).data();
        b_ = arena_->take(len).data();
        lg_ = lg;
    }

    const poly::Transform& transform() const { return *transform_; }
    std::span<std::uint32_t> a(std::size_t n) const { return {a_, n}; }
    std::span<std::uint32_t> b(std::size_t n) const { return {b_, n}; }

private:
    int lg_ = 0;
    std::unique_ptr<poly::Arena> arena_;
    std::optional<poly::Transform> transform_;
    std::uint32_t *a_ = nullptr, *b_ = nullptr;
};

// f mod x^n in a fresh span of the arena (zero past f).
inline std::span<std::uint32_t> copy(poly::Arena& arena, Span f, std::size_t n) {
    const auto out = arena.take(n);
    std::copy_n(f.begin(), std::min(n, f.size()), out.begin());
    return out;
}

}  // namespace detail

inline Poly multiply(Span a, Span b) {
    if (a.empty() || b.empty()) return {};
    const std::size_t n = a.size() + b.size() - 1;
    if (std::min(a.size(), b.size()) <= 32 || n <= 64) return detail::multiply_naive(a, b);
    const int lg = std::max(poly::Transform::kMinLog, int(std::bit_width(n - 1)));
    const std::size_t len = std::size_t(1) << lg;
    auto& w = detail::Workspace::get();
    w.reserve(lg);
    const auto& t = w.transform();
    const auto x = w.a(len);
    std::fill(std::copy(a.begin(), a.end(), x.begin()), x.end(), 0);
    if (a.data() == b.data() && a.size() == b.size()) {  // square: one forward transform
        t.forward(x);
        t.inverse_product(x, x, x);
    } else {
        const auto y = w.b(len);
        std::fill(std::copy(b.begin(), b.end(), y.begin()), y.end(), 0);
        t.forward(y);
        t.cyclic_product(x, y);
    }
    return Poly(x.begin(), x.begin() + std::ptrdiff_t(n));
}

inline Poly inverse(Span f, std::size_t n) {
    if (n == 0) return {};
    poly::Arena arena(poly::Transform::words(poly::inverse_log(n)) + 2 * poly::Arena::footprint(n) +
                      poly::inverse_scratch(n));
    const poly::Transform t(arena, poly::inverse_log(n));
    const auto in = detail::copy(arena, f, n), out = arena.take(n);
    poly::inverse(t, in, out, arena.take(poly::inverse_scratch(n)));
    return Poly(out.begin(), out.end());
}

inline Poly exp(Span f, std::size_t n) {
    if (n == 0) return {};
    poly::Arena arena(poly::Transform::words(poly::exp_log(n)) + poly::Arena::footprint(n) + poly::exp_scratch(n));
    const poly::Transform t(arena, poly::exp_log(n));
    const auto g = detail::copy(arena, f, n);
    poly::exp(t, g, g, arena.take(poly::exp_scratch(n)));
    return Poly(g.begin(), g.end());
}

inline Poly log(Span f, std::size_t n) {
    if (n == 0) return {};
    poly::Arena arena(poly::Transform::words(poly::log_log(n)) + poly::Arena::footprint(n) + poly::log_scratch(n));
    const poly::Transform t(arena, poly::log_log(n));
    const auto g = detail::copy(arena, f, n);
    poly::log(t, g, g, arena.take(poly::log_scratch(n)));
    return Poly(g.begin(), g.end());
}

inline Poly pow(Span f, std::uint64_t k, std::size_t n) {
    Poly g(n);
    if (n == 0) return g;
    if (k == 0) return g[0] = 1, g;
    const std::size_t z = std::size_t(std::find_if(f.begin(), f.end(), [](std::uint32_t c) { return c != 0; }) - f.begin());
    if (z == f.size() || (z > 0 && k > (n - 1) / z)) return g;  // f^k = 0 mod x^n
    const std::size_t shift = z * std::size_t(k), size = n - shift;
    poly::Arena arena(poly::Transform::words(poly::power_log(size)) + poly::Arena::footprint(size) +
                      poly::power_scratch(size));
    const poly::Transform t(arena, poly::power_log(size));
    const auto u = detail::copy(arena, f.subspan(z), size);
    const std::uint32_t c = power(u[0], k % (kMod - 1));
    poly::power(t, u, std::uint32_t(k % kMod), c, u, arena.take(poly::power_scratch(size)));
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
