// Cyclic convolutions modulo a prime P < 2^28 set at run time, for factors that fill at most half
// of the transform: lib/multimod's tables and recursion with ntt::Product's bottom stage and
// inverse top level (lib/ntt/notes.md, "Product"), and kernels with lazier reductions
// (kernels.hpp: with 16P < 2^32, values may grow to 16P).
//
//   lazy::Product product(lg, tables);   // 9 <= lg <= 20; Product::table_words(lg) words
//   product.multiply(a, b, out, work, mod, factor);   // out <- a b factor mod p, canonical
//
// a[i], b[i] < 4P, readable up to 2^lg / 2 + 8 words and zero from the factor's length up to
// 2^lg / 2. out and work hold 2^lg + kPadding words; work is destroyed and may be b's storage.
#pragma once

#include "kernels.hpp"
#include "lib/multimod/transform.hpp"

namespace lazy {

using multimod::Modulus;
using multimod::Vec;
using multimod::detail::slot;

inline constexpr std::size_t kPadding = 16;  // words after out and work: the kernels read 4 bytes past

// Forward transforms of a and b, leaf products into a, inverse transform; nv = 4^j >= 16 vectors.
// Two groups per bottom kernel. As ntt::detail::Subtrees.
class Subtrees {
public:
    Subtrees(const std::uint32_t* roots, const std::uint32_t* inverse_roots) : r_(roots), ir_(inverse_roots) {}

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
            kernels::forward_identity(a, h, r_);
            kernels::forward_identity(b, h, r_);
            return;
        }
        const std::uint32_t *x = r_ + slot(k), *y = r_ + slot(2 * k);
        if (h == 4) return kernels::forward_pair(a, b, h, x, y);
        kernels::forward(a, h, x, y);
        kernels::forward(b, h, x, y);
    }

    void inverse(Vec* a, std::size_t h, std::size_t k) const {
        if (k == 0) return kernels::inverse_identity(a, h, ir_);
        kernels::inverse(a, h, ir_ + slot(k), ir_ + slot(2 * k));
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

namespace detail {

using multimod::add;
using multimod::broadcast;
using multimod::diff;
using multimod::multiply;

inline Vec load(const std::uint32_t* x) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(x)); }

// First level for nv = 4h vectors from x[0, 16h), words < 4P (the upper half is zero): group 0
// with twiddle z = r[1]. Outputs < 8P. f may be x's storage.
inline void forward_radix4(Vec* f, std::size_t h, const std::uint32_t* x, const std::uint32_t* roots, const Modulus& m) {
    const multimod::Factor z(roots[1], roots[9], m.p);
    const Vec p2 = broadcast(2 * m.p), p4 = broadcast(4 * m.p);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec a = load(x + 8 * j), b = load(x + 8 * (j + h)), zb = multiply(b, z);  // zb < 2P
        f[j] = add(a, b), f[j + h] = diff(a, b, p4), f[j + 2 * h] = add(a, zb), f[j + 3 * h] = diff(a, zb, p2);
    }
}

// Last level for nv = 4h vectors (group 0, z = r^-1[1]) and the scale s: inputs < 4P, canonical
// outputs.
inline void inverse_radix4(Vec* f, std::size_t h, const std::uint32_t* inverse_roots, const multimod::Factor& s,
                           std::uint32_t modulus) {
    const multimod::Factor z(inverse_roots[1], inverse_roots[9], modulus);
    const Vec p = s.p, p2 = add(p, p), p4 = add(p2, p2), p8 = add(p4, p4);
    const auto scale = [&](Vec x) { return multimod::reduce(multiply(x, s), p); };  // any x < 2^32
    for (std::size_t j = 0; j < h; ++j) {
        const Vec p0 = f[j], p1 = f[j + h], q0 = f[j + 2 * h], q1 = f[j + 3 * h];
        const Vec ab = add(p0, p1), cd = add(q0, q1), amb = diff(p0, p1, p4);  // < 8P
        const Vec cmd = multiply(diff(q0, q1, p4), z);                         // < 2P
        f[j] = scale(add(ab, cd)), f[j + h] = scale(add(amb, cmd));
        f[j + 2 * h] = scale(diff(ab, cd, p8)), f[j + 3 * h] = scale(diff(amb, cmd, p2));
    }
}

}  // namespace detail

// Cyclic convolutions of length 2^lg with one pair of twiddle tables, rebuilt per modulus.
class Product {
public:
    // tables: table_words(lg) words, 32-byte aligned.
    Product(int lg, std::uint32_t* tables)
        : lg_(lg), roots_(tables), inverse_roots_(tables + multimod::detail::table_words(lg)) {}

    static std::size_t table_words(int lg) { return 2 * multimod::detail::table_words(lg); }

    void multiply(const std::uint32_t* a_in, const std::uint32_t* b_in, std::uint32_t* out, std::uint32_t* work,
                  const Modulus& m, std::uint32_t factor) const {
        using multimod::detail::build_table;
        m.select();
        kernels::k4P = detail::broadcast(4 * m.p), kernels::k8P = detail::broadcast(8 * m.p);
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
            detail::forward_radix4(a, h, a_in, roots_, m);
            detail::forward_radix4(b, h, b_in, roots_, m);
            for (std::size_t t = 0; t < 4; ++t) subtrees.visit(a + t * h, b + t * h, h, t);
            return detail::inverse_radix4(a, h, inverse_roots_, multimod::Factor(s, m), m.p);
        }
        // nv = 2 * 4^j: the first level from the input, each half's subtrees, then the last
        // radix-4 groups, the radix-2 level and the scale in one pass, s folded into the twiddles.
        const std::size_t q = nv / 8;
        alignas(32) Vec w[12];
        const std::uint32_t first[6] = {roots_[1], roots_[9], roots_[2], roots_[10], roots_[3], roots_[11]};
        for (int i = 0; i < 6; ++i) w[i] = detail::broadcast(first[i]);
        kernels::forward_radix8(a, q, a_in, w);
        kernels::forward_radix8(b, q, b_in, w);
        for (std::size_t c = 0; c < 8; ++c) subtrees.visit(a + c * q, b + c * q, q, c);
        const std::uint32_t z0 = inverse_roots_[1], x1 = inverse_roots_[slot(1)];
        const std::uint32_t* y1 = inverse_roots_ + slot(2);
        const std::uint32_t top[6] = {s, m.multiply(s, z0), m.multiply(s, y1[0]), m.multiply(s, y1[1]),
                                      m.multiply(s, x1), x1};
        for (int i = 0; i < 6; ++i)
            w[2 * i] = detail::broadcast(top[i]), w[2 * i + 1] = detail::broadcast(m.quotient(top[i]));
        kernels::inverse_top(a, q, w);
    }

private:
    int lg_;
    std::uint32_t *roots_, *inverse_roots_;
};

}  // namespace lazy
