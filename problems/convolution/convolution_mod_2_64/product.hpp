// Products of 64-bit factors modulo a prime set at run time, for factors that fill at most half of
// the transform: lib/multimod's transform with ntt::Product's bottom stage and inverse top level
// (design in lib/ntt/notes.md, "Product"), and a scheduled first level (kernels.hpp).
//
//   if (wide::Product::fits(lg, n, m)) {
//       wide::Product product(lg, tables);      // multimod::Transform::table_words(lg) words
//       product.multiply(multimod::Wide{a, n}, multimod::Wide{b, m}, out, work, mod, factor);
//   }                                           // as multimod::Transform::multiply
#pragma once

#include "lib/multimod/transform.hpp"
#include "kernels.hpp"

namespace wide {

using multimod::Modulus;
using multimod::Vec;
using multimod::detail::slot;

// ntt::detail::Subtrees with lib/multimod's kernels: forward transforms of a and b, leaf products
// into a, inverse transform. Two groups per bottom kernel, so subtrees hold at least 16 vectors.
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
        namespace mk = multimod::kernels;
        if (k == 0) {
            mk::forward_identity(a, h, r_);
            mk::forward_identity(b, h, r_);
            return;
        }
        const std::uint32_t *x = r_ + slot(k), *y = r_ + slot(2 * k);
        if (h == 4) return mk::forward_pair(a, b, h, x, y);
        mk::forward(a, h, x, y);
        mk::forward(b, h, x, y);
    }

    void inverse(Vec* a, std::size_t h, std::size_t k) const {
        if (k == 0) return multimod::kernels::inverse_identity(a, h, ir_);
        multimod::kernels::inverse(a, h, ir_ + slot(k), ir_ + slot(2 * k));
    }

    struct alignas(64) Leaves {
        std::uint32_t window[4][16];  // [w A_t, A_t]: x^i A_t mod x^8 - w is a sliding window
        std::uint32_t coefficients[4][8];
    };

    // Groups [first, first + nv / 4) with h = 1, two per kernel; first is even. The next two
    // groups' forward half overlaps the current two's products and inverse.
    void bottom(Vec* a, Vec* b, std::size_t nv, std::size_t first) const {
        Leaves leaves[2][2];
        kernels::bottom_first(a, b, leaves[0], r_ + slot(first), r_ + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const std::uint32_t *ix = ir_ + slot(k), *iy = ir_ + slot(2 * k);
            if (j + 8 == nv) return kernels::bottom_last(a + j, leaves[cur], ix, iy);
            kernels::bottom_both(a + j + 8, b + j + 8, leaves[next], r_ + slot(k + 2), r_ + slot(2 * k + 4), a + j,
                                 leaves[cur], ix, iy);
        }
    }

    const std::uint32_t *r_, *ir_;
};

// Cyclic convolutions of length 2^lg, 9 <= lg <= 20, of factors with at most 2^lg / 2 words.
class Product {
public:
    static bool fits(int lg, std::size_t n, std::size_t m) {
        return lg >= 9 && lg <= multimod::kMaxLog && 2 * std::max(n, m) <= std::size_t(1) << lg;
    }

    // tables: multimod::Transform::table_words(lg) words, 32-byte aligned.
    Product(int lg, std::uint32_t* tables)
        : lg_(lg), roots_(tables), inverse_roots_(tables + multimod::detail::table_words(lg)) {}

    // a b factor mod p into out (canonical, 2^lg words); work: 2^lg words, destroyed. Inputs and
    // in-place use as multimod::Transform::multiply.
    void multiply(multimod::Wide a_in, multimod::Wide b_in, std::uint32_t* out, std::uint32_t* work,
                  const Modulus& m, std::uint32_t factor) const {
        using namespace multimod::detail;
        m.select();
        const std::size_t len = std::size_t(1) << lg_, nv = len / 8;
        build_table(roots_, len / 16, m.roots[0], m);
        build_table(inverse_roots_, len / 16, m.roots[1], m);
        const Subtrees subtrees(roots_, inverse_roots_);
        // Undoes nv / 2^32 (the leaf products are Montgomery products).
        const std::uint32_t s = m.multiply(m.multiply(m.inverse(std::uint32_t(nv)), m.r), factor);
        auto* a = reinterpret_cast<Vec*>(out);
        auto* b = reinterpret_cast<Vec*>(work);
        if (std::countr_zero(nv) % 2 == 0) {  // nv = 4^j
            const std::size_t h = nv / 4;
            forward_radix4(a, h, true, a_in, roots_, m);
            forward_radix4(b, h, true, b_in, roots_, m);
            for (std::size_t t = 0; t < 4; ++t) subtrees.visit(a + t * h, b + t * h, h, t);
            return inverse_radix4(a, h, inverse_roots_, multimod::Factor(s, m), m.p);
        }
        // nv = 2 * 4^j: the first level from the 64-bit words, each half's subtrees, then the last
        // radix-4 groups, the radix-2 level and the scale in one pass, s folded into the twiddles.
        const std::size_t q = nv / 8;
        const std::uint32_t first[9] = {4 * m.p,   m.r,       m.quotient(m.r), roots_[1], roots_[9],
                                        roots_[2], roots_[10], roots_[3],      roots_[11]};
        alignas(32) Vec w[12];
        for (int i = 0; i < 9; ++i) w[i] = multimod::broadcast(first[i]);
        kernels::forward_radix8_wide(a, q, a_in.x, w);
        kernels::forward_radix8_wide(b, q, b_in.x, w);
        for (std::size_t c = 0; c < 8; ++c) subtrees.visit(a + c * q, b + c * q, q, c);
        const std::uint32_t z0 = inverse_roots_[1], x1 = inverse_roots_[slot(1)];
        const std::uint32_t* y1 = inverse_roots_ + slot(2);
        const std::uint32_t top[6] = {s, m.multiply(s, z0), m.multiply(s, y1[0]), m.multiply(s, y1[1]),
                                      m.multiply(s, x1), x1};
        for (int i = 0; i < 6; ++i)
            w[2 * i] = multimod::broadcast(top[i]), w[2 * i + 1] = multimod::broadcast(m.quotient(top[i]));
        kernels::inverse_top(a, q, w);
    }

private:
    int lg_;
    std::uint32_t *roots_, *inverse_roots_;
};

}  // namespace wide
