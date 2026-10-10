// multimod::WideProduct: Product (product.hpp) for factors of 64-bit words, 2^29 < p < 2^30, with
// lib/multimod's transform kernels and ntt::Product's bottom stage and inverse top modulo p
// (wide_kernels.hpp).
//
//   multimod::WideProduct product(lg, tables);   // 9 <= lg <= 20
//   product.multiply(multimod::Wide{a, n}, multimod::Wide{b, m}, out, work, mod, factor);
//
// a and b: any words, readable up to 2^lg / 2 and zero from n and m up to there. In place: work
// may be a's storage (a's first level runs before b's).
#pragma once

#include "product.hpp"
#include "wide_kernels.hpp"

namespace multimod {

struct WideKernels {
    using Input = Wide;
    static constexpr bool kBFirst = false;  // work may be a's storage

    static constexpr auto& forward = kernels::forward;
    static constexpr auto& forward_pair = kernels::forward_pair;
    static constexpr auto& inverse = kernels::inverse;
    static constexpr auto& forward_identity = kernels::forward_identity;
    static constexpr auto& inverse_identity = kernels::inverse_identity;
    static constexpr auto& bottom_first = wide_kernels::bottom_first;
    static constexpr auto& bottom_last = wide_kernels::bottom_last;
    static constexpr auto& bottom_both = wide_kernels::bottom_both;
    static constexpr auto& inverse_top = wide_kernels::inverse_top;

    static void select(const Modulus&) {}

    static void first_radix4(Vec* f, std::size_t h, Input x, const std::uint32_t* roots, const Modulus& m) {
        detail::forward_radix4(f, h, true, x, roots, m);
    }

    static constexpr auto& last_radix4 = detail::inverse_radix4;

    // 4p, then 2^32 mod p and the twiddles r[1], r[2], r[3], each followed by its Shoup quotient.
    static std::array<std::uint32_t, 9> radix8_constants(const std::uint32_t* roots, const Modulus& m) {
        return {4 * m.p, m.r, m.quotient(m.r), roots[1], roots[9], roots[2], roots[10], roots[3], roots[11]};
    }

    static void first_radix8(Vec* f, std::size_t q, Input x, const Vec* w) {
        wide_kernels::forward_radix8_wide(f, q, x.x, w);
    }
};

using WideProduct = Product<WideKernels>;

}  // namespace multimod
