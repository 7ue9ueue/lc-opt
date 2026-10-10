// Polynomial multiplication modulo P = 998244353 for factors that fill at most half of the
// transform: ntt::Convolution's transform with a faster bottom stage and top levels.
// Linux or macOS, x86-64 with AVX2. Design and measurements: lib/ntt/notes.md ("Product").
//
//   if (ntt::Product::fits(n, m)) {
//       ntt::Product product(n, m, extra);     // extra: caller's bytes in the same mapping
//       in.read(product.a(), n);               // coefficients < 998244353
//       in.read(product.b(), m);
//       const std::uint32_t* c = product.multiply();  // the n + m - 1 coefficients of a * b
//   }
//
// fits(n, m): the transform length 2^lg >= n + m - 1 has lg >= 9, and n, m <= 2^lg / 2. Other
// sizes need ntt::Convolution. Memory as ntt::Convolution, plus the extra bytes.
#pragma once

#include "lib/ntt/ntt.hpp"
#include "lib/ntt/product_kernels.hpp"

namespace ntt {

namespace detail {

// x - y + P for y < P.
inline Vec diff_canonical(Vec x, Vec y) { return _mm256_sub_epi32(_mm256_add_epi32(x, broadcast(kP)), y); }

// First level of a factor f[0, 4q) whose upper half f[4q, 8q) is zero (and not read). Modulo
// x^(n/2) - 1 and x^(n/2) + 1 the factor is unchanged, so one pass reads it once and writes the
// first radix-4 group of each half: group 0 to f[0, 4q), group 1 to f[4q, 8q). Inputs canonical;
// outputs < 4P.
inline void forward_radix8(Vec* f, std::size_t q, const std::uint32_t* roots) {
    const Factor i(roots[1], roots[9]), y(roots[2], roots[10]), z(roots[3], roots[11]);
    for (std::size_t j = 0; j < q; ++j) {
        const Vec f0 = f[j], f1 = f[j + q], f2 = f[j + 2 * q], f3 = f[j + 3 * q];
        // Group 0 (twiddles 1, 1, i): every term < 2P.
        const Vec g0 = add(f0, f2), g1 = add(f1, f3);
        const Vec h0 = diff_canonical(f0, f2), ih1 = multiply(diff_canonical(f1, f3), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
        f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
        // Group 1 (twiddles i, y, z).
        const Vec if2 = multiply(f2, i), if3 = multiply(f3, i);
        const Vec u0 = reduce(add(f0, if2), 2 * kP), v0 = reduce(diff(f0, if2), 2 * kP);
        const Vec yu1 = multiply(add(f1, if3), y), zv1 = multiply(diff(f1, if3), z);
        f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1);
        f[j + 6 * q] = add(v0, zv1), f[j + 7 * q] = diff(v0, zv1);
    }
}

// Recursion with the bottom stage of product_kernels.hpp: two groups per inlined kernel and no
// leaf weight array. Subtrees of at least 16 vectors, so tiles start at even group indices.
class Subtrees {
public:
    Subtrees(const std::uint32_t* roots, const std::uint32_t* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // Forward transforms of a and b, leaf products into a, inverse transform; nv = 4^j >= 16.
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

    // Groups [first, first + nv / 4) with h = 1 and their leaves, two per kernel; first is even.
    // The next two groups' forward half overlaps the current two's products and inverse.
    void bottom(Vec* a, Vec* b, std::size_t nv, std::size_t first) const {
        Leaves leaves[2][2];
        product_kernels::bottom_first(a, b, leaves[0], r_ + slot(first), r_ + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const std::uint32_t *ix = ir_ + slot(k), *iy = ir_ + slot(2 * k);
            if (j + 8 == nv) return product_kernels::bottom_last(a + j, leaves[cur], ix, iy);
            product_kernels::bottom_both(a + j + 8, b + j + 8, leaves[next], r_ + slot(k + 2), r_ + slot(2 * k + 4),
                                         a + j, leaves[cur], ix, iy);
        }
    }

    const std::uint32_t *r_, *ir_;
};

}  // namespace detail

// a * b for a of n and b of m coefficients when fits(n, m). Single use.
class Product {
public:
    static bool fits(std::size_t n, std::size_t m) {
        const int lg = log_length(n, m);
        return n && m && lg >= 9 && lg <= kMaxLog && 2 * std::max(n, m) <= std::size_t(1) << lg;
    }

    // extra: bytes for the caller after the tables, 64-byte aligned (extra()).
    Product(std::size_t n, std::size_t m, std::size_t extra = 0) : lg_(log_length(n, m)) {
        if (!fits(n, m)) std::abort();
        const std::size_t len = length(), words = 2 * (len + kPadding) + 2 * detail::table_words(lg_);
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes_ = (words * sizeof(std::uint32_t) + extra + kHuge - 1) / kHuge * kHuge + kHuge;
        region_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) std::abort();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region_) + kHuge - 1) & ~(kHuge - 1);
        a_ = reinterpret_cast<std::uint32_t*>(aligned);
#ifdef MADV_HUGEPAGE
        ::madvise(a_, bytes_ - kHuge, MADV_HUGEPAGE);
#endif
        b_ = a_ + len + kPadding;  // a different cache set from a at equal offsets
        roots_ = b_ + len + kPadding;
        inverse_roots_ = roots_ + detail::table_words(lg_);
        extra_ = inverse_roots_ + detail::table_words(lg_);
    }

    ~Product() { ::munmap(region_, bytes_); }

    Product(const Product&) = delete;
    Product& operator=(const Product&) = delete;

    // The factors, zero-filled: n and m coefficients < kModulus.
    std::uint32_t* a() { return a_; }
    std::uint32_t* b() { return b_; }
    // The caller's extra bytes, zero-filled.
    void* extra() { return extra_; }

    // The n + m - 1 coefficients of a * b, canonical, in a(); b() is destroyed.
    const std::uint32_t* multiply() {
        using namespace detail;
        const std::size_t len = length(), nv = len / 8;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        auto* a = reinterpret_cast<Vec*>(a_);
        auto* b = reinterpret_cast<Vec*>(b_);
        const Subtrees subtrees(roots_, inverse_roots_);
        if (std::countr_zero(nv) % 2 == 0) {  // nv = 4^j: radix-4 top level as ntt::Convolution
            const std::size_t h = nv / 4;
            forward_radix4(a, h, true, roots_);
            forward_radix4(b, h, true, roots_);
            for (std::size_t t = 0; t < 4; ++t) subtrees.visit(a + t * h, b + t * h, h, t);
            inverse_radix4(a, h, inverse_roots_, Factor(scale(nv)));
            return a_;
        }
        // nv = 2 * 4^j: forward_radix8, then each half's subtrees, then its last group, the radix-2
        // level and the scale in one pass (inverse_top).
        const std::size_t q = nv / 8;
        forward_radix8(a, q, roots_);
        forward_radix8(b, q, roots_);
        for (std::size_t c = 0; c < 8; ++c) subtrees.visit(a + c * q, b + c * q, q, c);
        const std::uint32_t s = scale(nv);
        const std::uint32_t *z0 = inverse_roots_ + 1, *x1 = inverse_roots_ + slot(1), *y1 = inverse_roots_ + slot(2);
        const std::uint32_t twiddles[6] = {s, multiply_mod(s, *z0), multiply_mod(s, y1[0]), multiply_mod(s, y1[1]),
                                           multiply_mod(s, *x1), *x1};
        alignas(32) Vec w[12];
        for (int i = 0; i < 6; ++i) w[2 * i] = broadcast(twiddles[i]), w[2 * i + 1] = broadcast(quotient(twiddles[i]));
        product_kernels::inverse_top(a, q, w);
        return a_;
    }

private:
    static constexpr std::size_t kPadding = 16;  // words after each factor: the kernels read 4 bytes past

    static int log_length(std::size_t n, std::size_t m) { return std::max(6, int(std::bit_width(n + m - 2))); }
    std::size_t length() const { return std::size_t(1) << lg_; }

    // nv^-1 2^32 mod P: undoes the factor nv / 2^32 of the transform and the leaf products.
    static std::uint32_t scale(std::size_t nv) {
        return detail::multiply_mod(detail::power(std::uint32_t(nv), detail::kP - 2), detail::kR);
    }

    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_, *extra_;
};

}  // namespace ntt
