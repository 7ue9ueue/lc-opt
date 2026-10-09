// Polynomial multiplication modulo 998244353 with AVX2 number-theoretic transforms.
// Linux or macOS, x86-64 with AVX2. Design, measurements and sources: lib/ntt/notes.md.
//
//   ntt::Convolution conv(n, m);               // factors of n and m coefficients
//   in.read(conv.a(), n);                      // coefficients < 998244353
//   in.read(conv.b(), m);
//   const std::uint32_t* c = conv.multiply();  // the n + m - 1 coefficients of a * b
//
// A product of length L uses a cyclic transform of length 2^lg >= L (lg >= 6): 2^lg / 2 bytes
// of tables and 2^lg * 8 bytes for a and b, in huge pages where the kernel allows.
#pragma once

#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "kernels.hpp"

namespace ntt {

inline constexpr std::uint32_t kModulus = kernels::kP;
inline constexpr int kMaxLog = 25;  // products of up to 2^25 coefficients

namespace detail {

using kernels::Vec;
using kernels::kP;

constexpr std::uint32_t multiply_mod(std::uint32_t x, std::uint32_t y) {
    return std::uint32_t(std::uint64_t(x) * y % kP);
}

constexpr std::uint32_t power(std::uint32_t x, std::uint32_t e) {
    std::uint32_t r = 1;
    for (; e; e >>= 1, x = multiply_mod(x, x))
        if (e & 1) r = multiply_mod(r, x);
    return r;
}

// Shoup quotient floor(w 2^32 / P) of a constant w < P.
constexpr std::uint32_t quotient(std::uint32_t w) { return std::uint32_t((std::uint64_t(w) << 32) / kP); }

inline constexpr std::uint32_t kR = std::uint32_t((std::uint64_t(1) << 32) % kP);  // 2^32 mod P

// kRoots[j] has order 2^(j + 2); 3 generates the multiplicative group.
inline constexpr int kRootCount = kMaxLog - 4;
inline constexpr auto kRoots = [] {
    std::array<std::array<std::uint32_t, kRootCount>, 2> t{};  // forward, inverse
    for (int j = 0; j < kRootCount; ++j) {
        t[0][j] = power(3, (kP - 1) >> (j + 2));
        t[1][j] = power(t[0][j], kP - 2);
    }
    return t;
}();

inline Vec broadcast(std::uint32_t x) { return _mm256_set1_epi32(int(x)); }
inline Vec add(Vec x, Vec y) { return _mm256_add_epi32(x, y); }
inline Vec diff(Vec x, Vec y) { return _mm256_sub_epi32(_mm256_add_epi32(x, broadcast(2 * kP)), y); }  // x - y + 2P

// x < 2 bound -> x mod bound.
inline Vec reduce(Vec x, std::uint32_t bound) { return _mm256_min_epu32(x, _mm256_sub_epi32(x, broadcast(bound))); }

// A constant factor w < P and its Shoup quotient.
struct Factor {
    Vec w, q;
    explicit Factor(std::uint32_t value) : w(broadcast(value)), q(broadcast(quotient(value))) {}
    Factor(std::uint32_t value, std::uint32_t value_quotient) : w(broadcast(value)), q(broadcast(value_quotient)) {}
};

// x w mod P in [0, 2P), for any x < 2^32.
inline Vec multiply(Vec x, const Factor& f) {
    const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(x, f.q), 32);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), f.q);
    const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
    return _mm256_sub_epi32(_mm256_mullo_epi32(x, f.w), _mm256_mullo_epi32(q, broadcast(kP)));
}

// Twiddle tables: entry k is the product of roots[j] over the set bits j of k, so entry 1 is
// sqrt(-1) and entry 2k is a square root of entry k. Blocks of 8 values, then their 8 quotients.
inline std::size_t slot(std::size_t k) { return (k >> 3 << 4) | (k & 7); }

inline std::size_t table_words(int lg) { return (std::size_t(1) << lg) / 8 + 16; }

// Entries [0, count) of the table for roots.
inline void build_table(std::uint32_t* t, std::size_t count, const std::array<std::uint32_t, kRootCount>& roots) {
    t[slot(0)] = 1;
    t[slot(0) + 8] = quotient(1);
    for (std::size_t h = 1; h < count; h *= 2) {
        const std::uint32_t root = roots[std::countr_zero(h)];
        if (h < 8) {
            for (std::size_t j = 0; j < h; ++j) {
                const std::uint32_t v = multiply_mod(t[slot(j)], root);
                t[slot(h + j)] = v;
                t[slot(h + j) + 8] = quotient(v);
            }
            continue;
        }
        // Quotient of v: (v 2^32 mod P) / -P mod 2^32.
        const Factor by(root), to_r(kR);
        for (std::size_t j = 0; j < h; j += 8) {
            const Vec v = reduce(multiply(_mm256_load_si256(reinterpret_cast<const Vec*>(t + slot(j))), by), kP);
            const Vec vr = reduce(multiply(v, to_r), kP);
            _mm256_store_si256(reinterpret_cast<Vec*>(t + slot(h + j)), v);
            _mm256_store_si256(reinterpret_cast<Vec*>(t + slot(h + j) + 8), _mm256_mullo_epi32(vr, broadcast(kernels::kNI)));
        }
    }
}

// Depth-first recursion over radix-4 groups. Group k at stride h holds 4h vectors, a polynomial
// mod X^4 - r[k]^2 with X = x^(8h); its outputs are the children 4k + t. A vector is a leaf: a
// polynomial mod x^8 - w, multiplied directly.
class Recursion {
public:
    Recursion(const std::uint32_t* roots, const std::uint32_t* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // Forward transforms of a and b, leaf products into a, inverse transform; nv a power of 4.
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
    // A subtree small enough for L2, level by level.
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

    // Moduli x^8 - w of the four leaves under group k: w = y, -y, z, -z, then their quotients.
    void leaf_weights(std::size_t k, std::uint32_t* w) const {
        const std::uint32_t* y = r_ + slot(2 * k);
        w[0] = y[0], w[1] = kP - y[0], w[2] = y[1], w[3] = kP - y[1];
        w[4] = y[8], w[5] = ~y[8], w[6] = y[9], w[7] = ~y[9];  // quotient(P - w) = ~quotient(w)
    }

    // Groups [first, first + nv / 4) with h = 1 and their leaves; batch j + 1's forward half
    // overlaps batch j's products and inverse.
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
};

// Inputs of the first level are canonical. If an input's upper half is zero ("sparse"), the
// first radix-2 level is a copy and the first radix-4 level reads only the lower half.

// f[i] and f[i + h] for i < h -> their sum and difference, < 2P.
inline void forward_radix2(Vec* f, std::size_t h, bool sparse) {
    if (sparse) {
        std::memcpy(f + h, f, h * sizeof(Vec));
        return;
    }
    for (std::size_t i = 0; i < h; ++i) {
        const Vec x = f[i], y = f[i + h];
        f[i] = reduce(add(x, y), 2 * kP);
        f[i + h] = reduce(diff(x, y), 2 * kP);
    }
}

// forward_identity() on canonical input; z = r[1].
inline void forward_radix4(Vec* f, std::size_t h, bool sparse, const std::uint32_t* roots) {
    if (!sparse) return kernels::forward_identity(f, h, roots);
    const Factor z(roots[1], roots[9]);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec a = f[j], b = f[j + h], zb = multiply(b, z);  // < 4P
        f[j] = add(a, b), f[j + h] = diff(a, b), f[j + 2 * h] = add(a, zb), f[j + 3 * h] = diff(a, zb);
    }
}

// inverse_identity() and the scale s: canonical output; z = r^-1[1].
inline void inverse_radix4(Vec* f, std::size_t h, const std::uint32_t* inverse_roots, const Factor& s) {
    const Factor z(inverse_roots[1], inverse_roots[9]);
    const auto scale = [&s](Vec x) { return reduce(multiply(x, s), kP); };
    for (std::size_t j = 0; j < h; ++j) {
        const Vec p0 = f[j], p1 = f[j + h], p2 = f[j + 2 * h], p3 = f[j + 3 * h];  // < 2P
        const Vec ab = reduce(add(p0, p1), 2 * kP), cd = reduce(add(p2, p3), 2 * kP);
        const Vec amb = reduce(diff(p0, p1), 2 * kP), cmd = multiply(diff(p2, p3), z);
        f[j] = scale(add(ab, cd)), f[j + h] = scale(add(amb, cmd));
        f[j + 2 * h] = scale(diff(ab, cd)), f[j + 3 * h] = scale(diff(amb, cmd));
    }
}

}  // namespace detail

// a * b for a of n and b of m coefficients, n, m >= 1, n + m - 1 <= 2^kMaxLog. Single use.
class Convolution {
public:
    Convolution(std::size_t n, std::size_t m)
        : n_(n), m_(m), lg_(std::max(6, int(std::bit_width(n + m - 2)))) {
        if (n == 0 || m == 0 || lg_ > kMaxLog) std::abort();
        const std::size_t len = length(), words = 2 * (len + kPadding) + 2 * detail::table_words(lg_);
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes_ = (words * sizeof(std::uint32_t) + kHuge - 1) / kHuge * kHuge + kHuge;
        region_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) std::abort();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region_) + kHuge - 1) & ~(kHuge - 1);
        auto* start = reinterpret_cast<std::uint32_t*>(aligned);
#ifdef MADV_HUGEPAGE
        ::madvise(start, bytes_ - kHuge, MADV_HUGEPAGE);
#endif
        a_ = start;
        b_ = a_ + len + kPadding;
        roots_ = b_ + len + kPadding;
        inverse_roots_ = roots_ + detail::table_words(lg_);
    }

    ~Convolution() { ::munmap(region_, bytes_); }

    Convolution(const Convolution&) = delete;
    Convolution& operator=(const Convolution&) = delete;

    // The factors, zero-filled: n and m coefficients < kModulus.
    std::uint32_t* a() { return a_; }
    std::uint32_t* b() { return b_; }

    // The n + m - 1 coefficients of a * b, canonical, in a(); b() is destroyed.
    const std::uint32_t* multiply() {
        using namespace detail;
        const std::size_t len = length(), nv = len / 8;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        const Recursion recursion(roots_, inverse_roots_);
        const std::uint32_t scale = multiply_mod(power(std::uint32_t(nv), kP - 2), kR);  // undoes nv / 2^32
        auto* a = reinterpret_cast<Vec*>(a_);
        auto* b = reinterpret_cast<Vec*>(b_);
        const bool sparse_a = n_ <= len / 2, sparse_b = m_ <= len / 2;
        if (std::countr_zero(nv) % 2) {  // nv = 2 * 4^j
            const std::size_t h = nv / 2;
            forward_radix2(a, h, sparse_a);
            forward_radix2(b, h, sparse_b);
            recursion.visit(a, b, h, 0);
            recursion.visit(a + h, b + h, h, 1);
            alignas(32) std::uint32_t s[16] = {};  // table layout: s at entry 1
            s[1] = scale;
            s[9] = quotient(scale);
            kernels::scale_radix2(a, h, s);
        } else {  // nv = 4^j
            const std::size_t h = nv / 4;
            forward_radix4(a, h, sparse_a, roots_);
            forward_radix4(b, h, sparse_b, roots_);
            for (std::size_t t = 0; t < 4; ++t) recursion.visit(a + t * h, b + t * h, h, t);
            inverse_radix4(a, h, inverse_roots_, Factor(scale));
        }
        return a_;
    }

private:
    static constexpr std::size_t kPadding = 16;  // words after each factor: the kernels read 4 bytes past

    std::size_t length() const { return std::size_t(1) << lg_; }

    std::size_t n_, m_;
    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
};

}  // namespace ntt
