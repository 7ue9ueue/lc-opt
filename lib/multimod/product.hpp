// Cyclic convolutions modulo a prime set at run time, for factors that fill at most half of the
// transform: lib/multimod's tables and recursion with ntt::Product's bottom stage and fused inverse
// top level (lib/ntt/notes.md, "Product"). Product<K> runs the kernel set K; the sets, with their
// inputs and primes, are LazyProduct (lazy_product.hpp) and WideProduct (wide_product.hpp).
//
//   if (multimod::LazyProduct::fits(lg, n, m)) {      // 9 <= lg <= 20, n, m <= 2^lg / 2
//       multimod::LazyProduct product(lg, tables);    // table_words(lg) words
//       product.multiply(a, b, out, work, mod, factor);   // out <- a b factor mod p, canonical
//   }
//
// out and work hold 2^lg + kPadding words; work is destroyed. A factor's words from 2^lg / 2 on
// do not change the result. Which words must be readable, and which arrays may share storage:
// the kernel set's header.
//
// A kernel set K has
//   Input                                              the factors' type
//   kBFirst                                            b's first level runs before a's, else after
//   forward, forward_pair, inverse, forward_identity,  transform kernels with lib/multimod's
//   inverse_identity                                   arguments (kernels.hpp)
//   bottom_first, bottom_last, bottom_both,            kernels with ntt::Product's arguments
//   inverse_top                                        (lib/ntt/product_kernels.hpp)
//   select(m)                                          sets its constants for the prime
//   first_radix4(f, h, x, roots, m)                    first level of a factor; 2^lg / 8 = 4h = 4^j
//   last_radix4(f, h, inverse_roots, s, p)             its last level with the scale s, canonical output
//   radix8_constants(roots, m)                         words that first_radix8 takes broadcast in w
//   first_radix8(f, q, x, w)                           first level of a factor; 2^lg / 8 = 8q = 2 * 4^j
#pragma once

#include <algorithm>
#include <array>
#include <bit>

#include "transform.hpp"

namespace multimod {

namespace detail {

// ntt::detail::Subtrees with the kernel set K: forward transforms of a and b, leaf products into
// a, inverse transform. Two groups per bottom kernel, so subtrees hold at least 16 vectors.
template <class K>
class Subtrees {
public:
    Subtrees(const std::uint32_t* roots, const std::uint32_t* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // nv = 4^j >= 16.
    void visit(Vec* a, Vec* b, std::size_t nv, std::size_t k) const {
        switch (nv) {
        case 16: return tile<16>(a, b, k);
        case 64: return tile<64>(a, b, k);
        case 256: return tile<256>(a, b, k);
        }
        const std::size_t h = nv / 4;
        forward(a, b, h, k);
        for (std::size_t t = 0; t < 4; ++t) visit(a + t * h, b + t * h, h, 4 * k + t);
        inverse(a, h, k);
    }

private:
    template <std::size_t NV>
    [[gnu::noinline]] void tile(Vec* a, Vec* b, std::size_t k) const {
        for (std::size_t h = NV / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < NV; j += 4 * h, ++g) forward(a + j, b + j, h, g);
        bottom(a, b, NV, k * (NV / 4));
        for (std::size_t h = 4; h < NV; h *= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < NV; j += 4 * h, ++g) inverse(a + j, h, g);
    }

    void forward(Vec* a, Vec* b, std::size_t h, std::size_t k) const {
        if (k == 0) {
            K::forward_identity(a, h, r_);
            K::forward_identity(b, h, r_);
            return;
        }
        const std::uint32_t *x = r_ + slot(k), *y = r_ + slot(2 * k);
        if (h == 4) return K::forward_pair(a, b, h, x, y);
        K::forward(a, h, x, y);
        K::forward(b, h, x, y);
    }

    void inverse(Vec* a, std::size_t h, std::size_t k) const {
        if (k == 0) return K::inverse_identity(a, h, ir_);
        K::inverse(a, h, ir_ + slot(k), ir_ + slot(2 * k));
    }

    struct alignas(64) Leaves {
        std::uint32_t window[4][16];  // [w A_t, A_t]: x^i A_t mod x^8 - w is a sliding window
        std::uint32_t coefficients[4][8];
    };

    // Groups [first, first + nv / 4) with h = 1, two per kernel; first is even. The next two
    // groups' forward half overlaps the current two's products and inverse.
    void bottom(Vec* a, Vec* b, std::size_t nv, std::size_t first) const {
        Leaves leaves[2][2];
        K::bottom_first(a, b, leaves[0], r_ + slot(first), r_ + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const std::uint32_t *ix = ir_ + slot(k), *iy = ir_ + slot(2 * k);
            if (j + 8 == nv) return K::bottom_last(a + j, leaves[cur], ix, iy);
            K::bottom_both(a + j + 8, b + j + 8, leaves[next], r_ + slot(k + 2), r_ + slot(2 * k + 4), a + j,
                           leaves[cur], ix, iy);
        }
    }

    const std::uint32_t *r_, *ir_;
};

}  // namespace detail

// Cyclic convolutions of length 2^lg with one pair of twiddle tables, rebuilt per modulus.
template <class K>
class Product {
public:
    using Input = typename K::Input;

    static constexpr std::size_t kPadding = 16;  // words after out and work: the kernels read 4 bytes past

    static bool fits(int lg, std::size_t n, std::size_t m) {
        return lg >= 9 && lg <= kMaxLog && 2 * std::max(n, m) <= std::size_t(1) << lg;
    }

    static std::size_t table_words(int lg) { return 2 * detail::table_words(lg); }

    // tables: table_words(lg) words, 32-byte aligned.
    Product(int lg, std::uint32_t* tables) : lg_(lg), roots_(tables), inverse_roots_(tables + detail::table_words(lg)) {}

    void multiply(Input a_in, Input b_in, std::uint32_t* out, std::uint32_t* work, const Modulus& m,
                  std::uint32_t factor) const {
        using detail::slot;
        m.select();
        K::select(m);
        const std::size_t len = std::size_t(1) << lg_, nv = len / 8;
        detail::build_table(roots_, len / 16, m.roots[0], m);
        detail::build_table(inverse_roots_, len / 16, m.roots[1], m);
        const detail::Subtrees<K> subtrees(roots_, inverse_roots_);
        // Undoes nv / 2^32 (the leaf products are Montgomery products).
        const std::uint32_t s = m.multiply(m.multiply(m.inverse(std::uint32_t(nv)), m.r), factor);
        auto* a = reinterpret_cast<Vec*>(out);
        auto* b = reinterpret_cast<Vec*>(work);
        if (std::countr_zero(nv) % 2 == 0) {  // nv = 4^j
            const std::size_t h = nv / 4;
            first_levels(a, b, a_in, b_in, [&](Vec* f, Input x) { K::first_radix4(f, h, x, roots_, m); });
            for (std::size_t t = 0; t < 4; ++t) subtrees.visit(a + t * h, b + t * h, h, t);
            return K::last_radix4(a, h, inverse_roots_, Factor(s, m), m.p);
        }
        // nv = 2 * 4^j: the first level from the input, each half's subtrees, then the last
        // radix-4 groups, the radix-2 level and the scale in one pass, s folded into the twiddles.
        const std::size_t q = nv / 8;
        alignas(32) Vec w[12];
        const auto first = K::radix8_constants(roots_, m);
        for (std::size_t i = 0; i < first.size(); ++i) w[i] = broadcast(first[i]);
        first_levels(a, b, a_in, b_in, [&](Vec* f, Input x) { K::first_radix8(f, q, x, w); });
        for (std::size_t c = 0; c < 8; ++c) subtrees.visit(a + c * q, b + c * q, q, c);
        const std::uint32_t z0 = inverse_roots_[1], x1 = inverse_roots_[slot(1)];
        const std::uint32_t* y1 = inverse_roots_ + slot(2);
        const std::uint32_t top[6] = {s, m.multiply(s, z0), m.multiply(s, y1[0]), m.multiply(s, y1[1]),
                                      m.multiply(s, x1), x1};
        for (int i = 0; i < 6; ++i) w[2 * i] = broadcast(top[i]), w[2 * i + 1] = broadcast(m.quotient(top[i]));
        K::inverse_top(a, q, w);
    }

private:
    // level(f, x) on both factors, in K's order.
    template <class Level>
    static void first_levels(Vec* a, Vec* b, Input a_in, Input b_in, Level level) {
        if constexpr (K::kBFirst) level(b, b_in), level(a, a_in);
        else level(a, a_in), level(b, b_in);
    }

    int lg_;
    std::uint32_t *roots_, *inverse_roots_;
};

}  // namespace multimod
