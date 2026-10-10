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

    static void first_radix4(Vec* a, Vec* b, std::size_t h, Input a_in, Input b_in, const std::uint32_t* roots,
                             const Modulus& m) {
        detail::forward_radix4(a, h, true, a_in, roots, m);
        detail::forward_radix4(b, h, true, b_in, roots, m);
    }

    static constexpr auto& last_radix4 = detail::inverse_radix4;

    static void first_radix8(Vec* a, Vec* b, std::size_t q, Input a_in, Input b_in, const std::uint32_t* roots,
                             const Modulus& m, Vec* w) {
        const std::uint32_t first[9] = {4 * m.p,  m.r,       m.quotient(m.r), roots[1], roots[9],
                                        roots[2], roots[10], roots[3],        roots[11]};
        for (int i = 0; i < 9; ++i) w[i] = broadcast(first[i]);
        wide_kernels::forward_radix8_wide(a, q, a_in.x, w);
        wide_kernels::forward_radix8_wide(b, q, b_in.x, w);
    }
};

using WideProduct = Product<WideKernels>;

}  // namespace multimod
