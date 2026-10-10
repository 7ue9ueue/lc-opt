// lib/poly's Newton iteration for 1 / f (inverse.hpp) with the two products of each step in one
// call: between them, the first product's inverse top level (its upper half) and the second
// product's forward top level are one pass over the transform. A whole output half is written by
// the last top level instead of copied. Measurements: notes.md (round 3).
//
//   step::inverse(t, f, g, scratch);  // as poly::inverse
//   step::inverse_step(t, f, g, to, gt, work);  // as poly::inverse_step
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include "lib/poly/inverse.hpp"

namespace step {

namespace detail {

using poly::detail::add;
using poly::detail::broadcast;
using poly::detail::diff;
using poly::detail::Factor;
using poly::detail::kP;
using poly::detail::low;
using poly::detail::reduce;
using poly::detail::times;
using poly::detail::Vec;

// Radix 4, quarters of h vectors. In: the first product's subtrees, < 2P. Out: the forward top
// level of x^(n/2) e, < 3P, where e = s (upper half of the first product's inverse top level).
// The inverse top level gives e's quarters c (canonical here) and d (< 2P); the forward one
// with zero quarters a, b gives c + d, c - d, -c - z d, -c + z d.
[[gnu::noinline]] inline void middle4(Vec* v, std::size_t h, const std::uint32_t* roots,
                                      const std::uint32_t* inverse_roots, std::uint32_t scale) {
    const Factor zi(inverse_roots[1], inverse_roots[9]), z(roots[1], roots[9]), s(scale);
    const Vec p = broadcast(kP), p2 = broadcast(2 * kP);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec q0 = v[j], q1 = v[j + h], q2 = v[j + 2 * h], q3 = v[j + 3 * h];
        const Vec ab = low(add(q0, q1)), cd = low(add(q2, q3));
        const Vec amb = low(diff(q0, q1)), cmd = times(diff(q2, q3), zi);
        const Vec c = reduce(times(diff(ab, cd), s), kP), d = times(diff(amb, cmd), s);
        const Vec mc = _mm256_sub_epi32(p, c), zmd = times(_mm256_sub_epi32(p2, d), z);
        v[j] = add(c, d), v[j + h] = diff(c, d), v[j + 2 * h] = add(mc, zmd), v[j + 3 * h] = diff(mc, zmd);
    }
}

// Radix 2, halves of h vectors: e = s (u - v) < 2P, then e and -e.
[[gnu::noinline]] inline void middle2(Vec* v, std::size_t h, std::uint32_t scale) {
    const Factor s(scale);
    const Vec p2 = broadcast(2 * kP);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec e = times(diff(v[j], v[j + h]), s);
        v[j] = e, v[j + h] = _mm256_sub_epi32(p2, e);
    }
}

// The second product's inverse top level, radix 4: quarters 2 and 3 (canonical, times s) into
// out[0, 2h) instead of in place.
[[gnu::noinline]] inline void upper4(const Vec* v, std::size_t h, const std::uint32_t* inverse_roots,
                                     std::uint32_t scale, Vec* out) {
    const Factor zi(inverse_roots[1], inverse_roots[9]), s(scale);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec q0 = v[j], q1 = v[j + h], q2 = v[j + 2 * h], q3 = v[j + 3 * h];
        const Vec ab = low(add(q0, q1)), cd = low(add(q2, q3));
        const Vec amb = low(diff(q0, q1)), cmd = times(diff(q2, q3), zi);
        out[j] = reduce(times(diff(ab, cd), s), kP), out[j + h] = reduce(times(diff(amb, cmd), s), kP);
    }
}

// Radix 2: s (u - v), canonical, into out[0, h).
[[gnu::noinline]] inline void upper2(const Vec* v, std::size_t h, std::uint32_t scale, Vec* out) {
    const Factor s(scale);
    for (std::size_t j = 0; j < h; ++j) out[j] = reduce(times(diff(v[j], v[j + h]), s), kP);
}

}  // namespace detail

// poly::inverse_step: to = (1 / f)[k, k + to.size()) from g = 1 / f mod x^k, to.size() <= k,
// transforms of length n = 2k (t: lg_max >= log2(n)). gt, work: n words each, 32-byte aligned. f
// may start at work (f is then overwritten); to may start at work[k] (no copy is made then).
// Otherwise none may overlap.
//   e = (f g mod (x^n - 1))[k, n), the upper half of the first product;
//   to = -(x^k e g mod (x^n - 1))[k, k + to.size()), the second.
inline void inverse_step(const poly::Transform& t, std::span<const std::uint32_t> f, std::span<const std::uint32_t> g,
                         std::span<std::uint32_t> to, std::span<std::uint32_t> gt, std::span<std::uint32_t> work) {
    using namespace poly::detail;
    const std::size_t k = g.size(), n = 2 * k, nv = n / 8;
    t.forward(g, 0, gt.first(n));
    const Recursion recursion(t.roots(), t.inverse_roots(), ProductBottom{t.roots(), t.inverse_roots(), gt.data()});
    const Source in(f.data(), std::min(n, f.size()), 0);
    const std::uint32_t first = kInverseScales[1][std::countr_zero(n)];
    const std::uint32_t second = ntt::detail::multiply_mod(first, kP - 1);
    auto* v = reinterpret_cast<Vec*>(work.data());
    // A whole half elsewhere, 32-byte aligned: the last top level writes it there.
    const bool direct = to.size() == k && to.data() != work.data() + k && reinterpret_cast<std::uintptr_t>(to.data()) % 32 == 0;
    auto* out = reinterpret_cast<Vec*>(to.data());
    if (std::countr_zero(nv) % 2 == 0) {
        const std::size_t h = nv / 4;
        forward_top4(in, v, h, t.roots());
        for (std::size_t q = 0; q < 4; ++q) recursion.visit(work.data() + 8 * q * h, h, q);
        detail::middle4(v, h, t.roots(), t.inverse_roots(), first);
        for (std::size_t q = 0; q < 4; ++q) recursion.visit(work.data() + 8 * q * h, h, q);
        if (direct) return detail::upper4(v, h, t.inverse_roots(), second, out);
        inverse_top4(v, h, poly::Half::kUpper, t.inverse_roots(), Factor(second));
    } else {
        const std::size_t h = nv / 2;
        forward_top2(in, v, h);
        recursion.visit(work.data(), h, 0);
        recursion.visit(work.data() + 8 * h, h, 1);
        detail::middle2(v, h, first);
        recursion.visit(work.data(), h, 0);
        recursion.visit(work.data() + 8 * h, h, 1);
        if (direct) return detail::upper2(v, h, second, out);
        inverse_top2(v, h, poly::Half::kUpper, second);
    }
    if (to.data() != work.data() + k) std::copy_n(work.begin() + std::ptrdiff_t(k), to.size(), to.begin());
}

// poly::inverse with the step above: g = 1 / f mod x^n for n = g.size() >= 1, f[0] != 0.
// scratch: poly::inverse_scratch(n) words from an Arena; t: lg_max >= poly::inverse_log(n).
inline void inverse(const poly::Transform& t, std::span<const std::uint32_t> f, std::span<std::uint32_t> g,
                    std::span<std::uint32_t> scratch) {
    const std::size_t n = g.size();
    std::size_t k = std::min(n, poly::detail::kInverseBase);
    poly::detail::inverse_direct(f, g.first(k));
    for (; k < n; k *= 2) {
        const std::size_t len = 2 * k;
        step::inverse_step(t, f, g.first(k), g.subspan(k, std::min(k, n - k)), scratch.first(len),
                     scratch.subspan(poly::Arena::footprint(len), len));
    }
}

}  // namespace step
