// Cyclic convolution modulo an NTT prime p <= 998244353 chosen at run time: lib/ntt's transform
// (same recursion, tables and kernels; design in lib/ntt/notes.md) with p a variable.
//
//   multimod::Modulus mod(985661441, 3);   // prime, primitive root
//   multimod::Transform t(lg, tables);     // length 2^lg, 6 <= lg <= 20
//   t.multiply(multimod::Padded{a, n}, multimod::Padded{b, m}, out, work, mod, factor);
//                                          // out <- a * b * factor mod p
//
// Inputs (the transform holds a * b without wrap-around iff n + m - 1 <= 2^lg):
//   Padded{x, n}   x[i] < 2p; readable and zero up to max(n rounded up to 8, 2^lg / 2)
//   Bounded{x, n}  x[i] < 2p; no word past n is read
//   Wide{x, n}     any 64-bit x[i], p > 2^29; readable and zero up to max(n rounded up to 8, 2^lg / 2)
// out and work hold 2^lg words plus kPadding (the kernels read 4 bytes past). work may be b's
// storage (a's too for Wide): each input is read in full before its own array or work is written.
// Also exported: the AVX2 helpers broadcast, add, diff, reduce, Factor and multiply.
#pragma once

#include <immintrin.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernels.hpp"

namespace multimod {

using kernels::Vec;

inline constexpr int kMaxLog = 20;
inline constexpr int kRootCount = kMaxLog - 4;  // roots of order 2^2 .. 2^(kMaxLog - 3)

// Arithmetic modulo a prime p < 2^30 with 2^(kMaxLog - 3) | p - 1.
struct Modulus {
    std::uint32_t p, r, ni;  // p, 2^32 mod p, -1 / p mod 2^32
    std::array<std::array<std::uint32_t, kRootCount>, 2> roots;  // forward, inverse; roots[.][j] of order 2^(j + 2)

    Modulus(std::uint32_t prime, std::uint32_t generator)
        : p(prime), r(std::uint32_t((std::uint64_t(1) << 32) % prime)), ni(negative_inverse(prime)) {
        for (int j = 0; j < kRootCount; ++j) {
            roots[0][j] = power(generator, (p - 1) >> (j + 2));
            roots[1][j] = inverse(roots[0][j]);
        }
    }

    std::uint32_t multiply(std::uint32_t x, std::uint32_t y) const { return std::uint32_t(std::uint64_t(x) * y % p); }
    std::uint32_t power(std::uint32_t x, std::uint32_t e) const {
        std::uint32_t result = 1;
        for (; e; e >>= 1, x = multiply(x, x))
            if (e & 1) result = multiply(result, x);
        return result;
    }
    std::uint32_t inverse(std::uint32_t x) const { return power(x, p - 2); }
    // Shoup quotient floor(w 2^32 / p) of w < p.
    std::uint32_t quotient(std::uint32_t w) const { return std::uint32_t((std::uint64_t(w) << 32) / p); }

    // Makes the kernels work modulo p.
    void select() const { kernels::kP = p, kernels::k2P = 2 * p, kernels::kNI = ni; }

private:
    static constexpr std::uint32_t negative_inverse(std::uint32_t p) {
        std::uint32_t x = p;  // Newton: correct to 3, 6, 12, 24, 48 bits
        for (int i = 0; i < 4; ++i) x *= 2 - p * x;
        return 0 - x;
    }
};

// AVX2 arithmetic on 8 lanes.
inline Vec broadcast(std::uint32_t x) { return _mm256_set1_epi32(int(x)); }
inline Vec add(Vec x, Vec y) { return _mm256_add_epi32(x, y); }
// x - y + bias.
inline Vec diff(Vec x, Vec y, Vec bias) { return _mm256_sub_epi32(_mm256_add_epi32(x, bias), y); }
// x < 2 bound -> x mod bound.
inline Vec reduce(Vec x, Vec bound) { return _mm256_min_epu32(x, _mm256_sub_epi32(x, bound)); }

// A constant factor w < p with its Shoup quotient, and p.
struct Factor {
    Vec w, q, p;
    Factor(std::uint32_t value, std::uint32_t value_quotient, std::uint32_t modulus)
        : w(broadcast(value)), q(broadcast(value_quotient)), p(broadcast(modulus)) {}
    Factor(std::uint32_t value, const Modulus& m) : Factor(value, m.quotient(value), m.p) {}
};

// x w mod p in [0, 2p), for any x < 2^32.
inline Vec multiply(Vec x, const Factor& f) {
    const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(x, f.q), 32);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), f.q);
    const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
    return _mm256_sub_epi32(_mm256_mullo_epi32(x, f.w), _mm256_mullo_epi32(q, f.p));
}

// Inputs of Transform::multiply (see the header).
struct Padded {
    const std::uint32_t* x;
    std::size_t count;
};
struct Bounded {
    const std::uint32_t* x;
    std::size_t count;
};
struct Wide {
    const std::uint64_t* x;
    std::size_t count;
};

namespace detail {

// Twiddle tables: entry k is the product of roots[j] over the set bits j of k. Blocks of 8 values,
// then their 8 quotients.
inline std::size_t slot(std::size_t k) { return (k >> 3 << 4) | (k & 7); }
inline std::size_t table_words(int lg) { return (std::size_t(1) << lg) / 8 + 16; }

inline void build_table(std::uint32_t* t, std::size_t count, const std::array<std::uint32_t, kRootCount>& roots,
                        const Modulus& m) {
    t[slot(0)] = 1;
    t[slot(0) + 8] = m.quotient(1);
    const Vec p = broadcast(m.p), ni = broadcast(m.ni);
    for (std::size_t h = 1; h < count; h *= 2) {
        const std::uint32_t root = roots[std::countr_zero(h)];
        if (h < 8) {
            for (std::size_t j = 0; j < h; ++j) {
                const std::uint32_t v = m.multiply(t[slot(j)], root);
                t[slot(h + j)] = v;
                t[slot(h + j) + 8] = m.quotient(v);
            }
            continue;
        }
        // Quotient of v: (v 2^32 mod p) * ni mod 2^32, ni = -1 / p.
        const Factor by(root, m), to_r(m.r, m);
        for (std::size_t j = 0; j < h; j += 8) {
            const Vec v = reduce(multiply(_mm256_load_si256(reinterpret_cast<const Vec*>(t + slot(j))), by), p);
            const Vec vr = reduce(multiply(v, to_r), p);
            _mm256_store_si256(reinterpret_cast<Vec*>(t + slot(h + j)), v);
            _mm256_store_si256(reinterpret_cast<Vec*>(t + slot(h + j) + 8), _mm256_mullo_epi32(vr, ni));
        }
    }
}

// lib/ntt's depth-first recursion over radix-4 groups (see ntt::detail::Recursion).
class Recursion {
public:
    Recursion(const std::uint32_t* roots, const std::uint32_t* inverse_roots, std::uint32_t p)
        : r_(roots), ir_(inverse_roots), p_(p) {}

    void visit(Vec* a, Vec* b, std::size_t nv, std::size_t k) const {
        switch (nv) {
        case 4: return tile<4>(a, b, k);
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
        std::uint32_t window[4][16];
        std::uint32_t coefficients[4][8];
    };

    void leaf_weights(std::size_t k, std::uint32_t* w) const {
        const std::uint32_t* y = r_ + slot(2 * k);
        w[0] = y[0], w[1] = p_ - y[0], w[2] = y[1], w[3] = p_ - y[1];
        w[4] = y[8], w[5] = ~y[8], w[6] = y[9], w[7] = ~y[9];  // quotient(p - w) = ~quotient(w)
    }

    void bottom(Vec* a, Vec* b, std::size_t nv, std::size_t first) const {
        Leaves leaves[2];
        alignas(32) std::uint32_t w[2][8];
        leaf_weights(first, w[0]);
        kernels::bottom_first(a, b, &leaves[0], r_ + slot(first), r_ + slot(2 * first), w[0]);
        for (std::size_t j = 0, k = first; j < nv; j += 4, ++k) {
            const std::size_t cur = j / 4 % 2, next = cur ^ 1;
            const std::uint32_t *ix = ir_ + slot(k), *iy = ir_ + slot(2 * k);
            if (j + 4 == nv) {
                kernels::bottom_last(a + j, &leaves[cur], ix, iy);
                break;
            }
            leaf_weights(k + 1, w[next]);
            kernels::bottom_both(a + j + 4, b + j + 4, &leaves[next], r_ + slot(k + 1), r_ + slot(2 * k + 2), w[next],
                                 a + j, &leaves[cur], ix, iy);
        }
    }

    const std::uint32_t *r_, *ir_;
    std::uint32_t p_;
};

// f[i] and f[i + h] for i < h -> their sum and difference, < 2p. Sparse: f[h, 2h) is zero.
inline void forward_radix2(Vec* f, std::size_t h, bool sparse, std::uint32_t modulus) {
    if (sparse) {
        std::memcpy(f + h, f, h * sizeof(Vec));
        return;
    }
    const Vec p2 = broadcast(2 * modulus);
    for (std::size_t i = 0; i < h; ++i) {
        const Vec x = f[i], y = f[i + h];
        f[i] = reduce(add(x, y), p2);
        f[i + h] = reduce(diff(x, y, p2), p2);
    }
}

// kernels::inverse_identity() and the scale s: canonical output; z = r^-1[1].
inline void inverse_radix4(Vec* f, std::size_t h, const std::uint32_t* inverse_roots, const Factor& s,
                           std::uint32_t modulus) {
    const Factor z(inverse_roots[1], inverse_roots[9], modulus);
    const Vec p = s.p, p2 = add(p, p);
    const auto scale = [&](Vec x) { return reduce(multiply(x, s), p); };
    for (std::size_t j = 0; j < h; ++j) {
        const Vec p0 = f[j], p1 = f[j + h], q0 = f[j + 2 * h], q1 = f[j + 3 * h];  // < 2p
        const Vec ab = reduce(add(p0, p1), p2), cd = reduce(add(q0, q1), p2);
        const Vec amb = reduce(diff(p0, p1, p2), p2), cmd = multiply(diff(q0, q1, p2), z);
        f[j] = scale(add(ab, cd)), f[j + h] = scale(add(amb, cmd));
        f[j + 2 * h] = scale(diff(ab, cd, p2)), f[j + 3 * h] = scale(diff(amb, cmd, p2));
    }
}

// An input's vectors as canonical residues, zero past its count.
template <class Input>
class Narrow;

template <>
class Narrow<Padded> {
public:
    Narrow(const Modulus& m, const Padded& in) : p_(broadcast(m.p)), x_(in.x), count_(in.count) {}

    std::size_t count() const { return count_; }

    // Words [8 v, 8 v + 8).
    Vec operator()(std::size_t v) const {
        return reduce(_mm256_loadu_si256(reinterpret_cast<const Vec*>(x_ + 8 * v)), p_);
    }

private:
    Vec p_;
    const std::uint32_t* x_;
    std::size_t count_;
};

template <>
class Narrow<Bounded> {
public:
    Narrow(const Modulus& m, const Bounded& in) : p_(broadcast(m.p)), x_(in.x), count_(in.count) {}

    std::size_t count() const { return count_; }

    // Words [8 v, 8 v + 8).
    Vec operator()(std::size_t v) const {
        const std::size_t i = 8 * v;
        if (i + 8 <= count_) return reduce(_mm256_loadu_si256(reinterpret_cast<const Vec*>(x_ + i)), p_);
        if (i >= count_) return _mm256_setzero_si256();
        const Vec keep = _mm256_cmpgt_epi32(broadcast(std::uint32_t(count_ - i)),
                                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        return reduce(_mm256_maskload_epi32(reinterpret_cast<const int*>(x_ + i), keep), p_);
    }

private:
    Vec p_;
    const std::uint32_t* x_;
    std::size_t count_;
};

// hi 2^32 + lo = hi (2^32 mod p) + lo mod p, for 2^29 < p < 2^30.
template <>
class Narrow<Wide> {
public:
    Narrow(const Modulus& m, const Wide& in) : high_(m.r, m), p_(broadcast(m.p)), x_(in.x), count_(in.count) {}

    std::size_t count() const { return count_; }

    // Words [8 v, 8 v + 8).
    Vec operator()(std::size_t v) const {
        const std::uint64_t* x = x_ + 8 * v;
        const __m256 v0 = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const Vec*>(x)));
        const __m256 v1 = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const Vec*>(x + 4)));
        // Dwords of coefficients [0 1 4 5 | 2 3 6 7]; lo < 2^32 < 8p, two halvings bring it below 2p.
        const Vec lo = _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0x88));
        const Vec hi = _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0xDD));
        const Vec p2 = add(p_, p_), low = reduce(reduce(lo, add(p2, p2)), p2);
        const Vec r = reduce(reduce(add(multiply(hi, high_), low), p2), p_);  // from < 4p
        return _mm256_permute4x64_epi64(r, 0xD8);
    }

private:
    Factor high_;  // 2^32 mod p
    Vec p_;
    const std::uint64_t* x_;
    std::size_t count_;
};

// The narrowed input into dst[0, extent), zero past its count.
template <class Input>
inline void narrow_into(std::uint32_t* dst, std::size_t extent, const Narrow<Input>& narrow) {
    std::size_t v = 0;
    for (; 8 * v < narrow.count(); ++v) _mm256_storeu_si256(reinterpret_cast<Vec*>(dst + 8 * v), narrow(v));
    if (8 * v < extent) std::memset(dst + 8 * v, 0, (extent - 8 * v) * sizeof(std::uint32_t));
}

// kernels::forward_identity() on the narrowed input, f of 4h vectors; z = r[1]. Sparse: the
// input's upper half is zero, and one pass narrows and transforms.
template <class Input>
inline void forward_radix4(Vec* f, std::size_t h, bool sparse, const Narrow<Input>& narrow,
                           const std::uint32_t* roots, std::uint32_t modulus) {
    if (!sparse) {
        narrow_into(reinterpret_cast<std::uint32_t*>(f), 32 * h, narrow);
        return kernels::forward_identity(f, h, roots);
    }
    const Factor z(roots[1], roots[9], modulus);
    const Vec p2 = broadcast(2 * modulus);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec a = narrow(j), b = narrow(j + h), zb = multiply(b, z);  // < 4p
        f[j] = add(a, b), f[j + h] = diff(a, b, p2), f[j + 2 * h] = add(a, zb), f[j + 3 * h] = diff(a, zb, p2);
    }
}

// First level of a factor of 64q words whose upper half is zero, for
// 2^lg = 2 * 4^j: one pass narrows x and writes the first radix-4 group of both halves of f
// (as convolution_mod's Product). Outputs < 4p.
template <class Input>
inline void forward_radix8(Vec* f, std::size_t q, const std::uint32_t* roots, const Narrow<Input>& narrow,
                           std::uint32_t modulus) {
    const Factor i(roots[1], roots[9], modulus), y(roots[2], roots[10], modulus), z(roots[3], roots[11], modulus);
    const Vec p = broadcast(modulus), p2 = add(p, p);
    for (std::size_t j = 0; j < q; ++j) {
        const Vec f0 = narrow(j), f1 = narrow(j + q), f2 = narrow(j + 2 * q), f3 = narrow(j + 3 * q);
        const Vec g0 = add(f0, f2), g1 = add(f1, f3);
        const Vec h0 = diff(f0, f2, p), ih1 = multiply(diff(f1, f3, p), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1, p2);
        f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1, p2);
        const Vec if2 = multiply(f2, i), if3 = multiply(f3, i);
        const Vec u0 = reduce(add(f0, if2), p2), v0 = reduce(diff(f0, if2, p2), p2);
        const Vec yu1 = multiply(add(f1, if3), y), zv1 = multiply(diff(f1, if3, p2), z);
        f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1, p2);
        f[j + 6 * q] = add(v0, zv1), f[j + 7 * q] = diff(v0, zv1, p2);
    }
}

}  // namespace detail

// Cyclic convolutions of length 2^lg with one pair of twiddle tables, rebuilt per modulus.
class Transform {
public:
    static constexpr std::size_t kPadding = 16;  // words after each array: the kernels read 4 bytes past

    // tables: 2 * table_words(lg) words, 32-byte aligned.
    Transform(int lg, std::uint32_t* tables)
        : lg_(lg), roots_(tables), inverse_roots_(tables + detail::table_words(lg)) {}

    static std::size_t table_words(int lg) { return 2 * detail::table_words(lg); }

    // a b factor mod p into out (canonical, 2^lg words); work: 2^lg words, destroyed.
    template <class Input>
    void multiply(const Input& a_in, const Input& b_in, std::uint32_t* out, std::uint32_t* work, const Modulus& m,
                  std::uint32_t factor) const {
        using namespace detail;
        m.select();
        const std::size_t len = std::size_t(1) << lg_, nv = len / 8;
        build_table(roots_, len / 16, m.roots[0], m);
        build_table(inverse_roots_, len / 16, m.roots[1], m);
        const Recursion recursion(roots_, inverse_roots_, m.p);
        // Undoes nv / 2^32 (the leaf products are Montgomery products).
        const std::uint32_t scale = m.multiply(m.multiply(m.inverse(std::uint32_t(nv)), m.r), factor);
        auto* a = reinterpret_cast<Vec*>(out);
        auto* b = reinterpret_cast<Vec*>(work);
        const detail::Narrow<Input> narrow_a(m, a_in), narrow_b(m, b_in);
        const bool sparse_a = a_in.count <= len / 2, sparse_b = b_in.count <= len / 2;
        if (std::countr_zero(nv) % 2 == 1 && nv >= 32 && sparse_a && sparse_b) {  // nv = 2 * 4^j
            const std::size_t h = nv / 2, q = nv / 8;
            forward_radix8(a, q, roots_, narrow_a, m.p);
            forward_radix8(b, q, roots_, narrow_b, m.p);
            for (std::size_t c = 0; c < 4; ++c) recursion.visit(a + c * q, b + c * q, q, c);
            kernels::inverse_identity(a, q, inverse_roots_);
            for (std::size_t c = 4; c < 8; ++c) recursion.visit(a + c * q, b + c * q, q, c);
            kernels::inverse(a + h, q, inverse_roots_ + slot(1), inverse_roots_ + slot(2));
            return scale_radix2(a, h, scale, m);
        }
        if (std::countr_zero(nv) % 2 == 0) {  // nv = 4^j
            const std::size_t h = nv / 4;
            forward_radix4(a, h, sparse_a, narrow_a, roots_, m.p);
            forward_radix4(b, h, sparse_b, narrow_b, roots_, m.p);
            for (std::size_t t = 0; t < 4; ++t) recursion.visit(a + t * h, b + t * h, h, t);
            inverse_radix4(a, h, inverse_roots_, Factor(scale, m), m.p);
            return;
        }
        narrow_into(out, sparse_a ? len / 2 : len, narrow_a);
        narrow_into(work, sparse_b ? len / 2 : len, narrow_b);
        const std::size_t h = nv / 2;  // nv = 2 * 4^j
        forward_radix2(a, h, sparse_a, m.p);
        forward_radix2(b, h, sparse_b, m.p);
        recursion.visit(a, b, h, 0);
        recursion.visit(a + h, b + h, h, 1);
        scale_radix2(a, h, scale, m);
    }

private:
    // The last radix-2 level and the scale s.
    static void scale_radix2(Vec* a, std::size_t h, std::uint32_t s, const Modulus& m) {
        alignas(32) std::uint32_t table[16] = {};  // table layout: s at entry 1
        table[1] = s;
        table[9] = m.quotient(s);
        kernels::scale_radix2(a, h, table);
    }

    int lg_;
    std::uint32_t *roots_, *inverse_roots_;
};

}  // namespace multimod
