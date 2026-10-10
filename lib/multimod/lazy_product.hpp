// multimod::LazyProduct: Product (product.hpp) for primes p < 2^28, whose kernels
// (lazy_kernels.hpp) reduce less often: with 16p < 2^32, values may grow to 16p.
//
//   multimod::LazyProduct product(lg, tables);   // 9 <= lg <= 20
//   product.multiply(a, b, out, work, mod, factor);
//
// a[i], b[i] < 4p, readable up to 2^lg / 2 + 8 words and zero from the factor's length up to
// 2^lg / 2. In place: out may be a's storage with b in its upper half (b's first level runs
// before a's), or work b's storage.
#pragma once

#include "lazy_kernels.hpp"
#include "product.hpp"

namespace multimod {

struct LazyKernels {
    using Input = const std::uint32_t*;

    static constexpr auto& forward = lazy_kernels::forward;
    static constexpr auto& forward_pair = lazy_kernels::forward_pair;
    static constexpr auto& inverse = lazy_kernels::inverse;
    static constexpr auto& forward_identity = lazy_kernels::forward_identity;
    static constexpr auto& inverse_identity = lazy_kernels::inverse_identity;
    static constexpr auto& bottom_first = lazy_kernels::bottom_first;
    static constexpr auto& bottom_last = lazy_kernels::bottom_last;
    static constexpr auto& bottom_both = lazy_kernels::bottom_both;
    static constexpr auto& inverse_top = lazy_kernels::inverse_top;

    static void select(const Modulus& m) {
        lazy_kernels::k4P = broadcast(4 * m.p), lazy_kernels::k8P = broadcast(8 * m.p);
    }

    static void first_radix4(Vec* a, Vec* b, std::size_t h, Input a_in, Input b_in, const std::uint32_t* roots,
                             const Modulus& m) {
        forward_radix4(b, h, b_in, roots, m);
        forward_radix4(a, h, a_in, roots, m);
    }

    // Last level for nv = 4h vectors (group 0, z = r^-1[1]) and the scale s: inputs < 4p, canonical
    // outputs.
    static void last_radix4(Vec* f, std::size_t h, const std::uint32_t* inverse_roots, const Factor& s,
                            std::uint32_t modulus) {
        const Factor z(inverse_roots[1], inverse_roots[9], modulus);
        const Vec p = s.p, p2 = add(p, p), p4 = add(p2, p2), p8 = add(p4, p4);
        const auto scale = [&](Vec x) { return reduce(multiply(x, s), p); };  // any x < 2^32
        for (std::size_t j = 0; j < h; ++j) {
            const Vec p0 = f[j], p1 = f[j + h], q0 = f[j + 2 * h], q1 = f[j + 3 * h];
            const Vec ab = add(p0, p1), cd = add(q0, q1), amb = diff(p0, p1, p4);  // < 8p
            const Vec cmd = multiply(diff(q0, q1, p4), z);                         // < 2p
            f[j] = scale(add(ab, cd)), f[j + h] = scale(add(amb, cmd));
            f[j + 2 * h] = scale(diff(ab, cd, p8)), f[j + 3 * h] = scale(diff(amb, cmd, p2));
        }
    }

    static void first_radix8(Vec* a, Vec* b, std::size_t q, Input a_in, Input b_in, const std::uint32_t* roots,
                             const Modulus&, Vec* w) {
        const std::uint32_t first[6] = {roots[1], roots[9], roots[2], roots[10], roots[3], roots[11]};
        for (int i = 0; i < 6; ++i) w[i] = broadcast(first[i]);
        lazy_kernels::forward_radix8(b, q, b_in, w);
        lazy_kernels::forward_radix8(a, q, a_in, w);
    }

private:
    // First level for nv = 4h vectors from x[0, 16h), words < 4p (the upper half is zero): group 0
    // with twiddle z = r[1]. Outputs < 8p. f may be x's storage.
    static void forward_radix4(Vec* f, std::size_t h, Input x, const std::uint32_t* roots, const Modulus& m) {
        const Factor z(roots[1], roots[9], m.p);
        const Vec p2 = broadcast(2 * m.p), p4 = broadcast(4 * m.p);
        for (std::size_t j = 0; j < h; ++j) {
            const Vec a = load(x + 8 * j), b = load(x + 8 * (j + h)), zb = multiply(b, z);  // zb < 2p
            f[j] = add(a, b), f[j + h] = diff(a, b, p4), f[j + 2 * h] = add(a, zb), f[j + 3 * h] = diff(a, zb, p2);
        }
    }

    static Vec load(const std::uint32_t* x) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(x)); }
};

using LazyProduct = Product<LazyKernels>;

}  // namespace multimod
