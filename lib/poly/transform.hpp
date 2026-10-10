// Transforms modulo 998244353 for power series code. Linux or macOS, x86-64 with AVX2.
// Design, measurements and sources: lib/poly/notes.md.
//
//   poly::Arena arena(words);                  // memory for spans (huge pages), owned by the caller
//   poly::Transform t(arena, lg_max);          // tables for lengths 2^6 .. 2^lg_max
//   auto a = arena.take(n), b = arena.take(n); // n = 2^lg; coefficients in [0, P)
//   t.forward(b);                              // b becomes its transform
//   t.cyclic_product(a, b);                    // a = a b mod (x^n - 1), b unchanged
//   t.multiply(b, c);                          // b = b c, both transforms
//   t.inverse(b);                              // a transform back to coefficients
//   t.inverse_product(b, c, a);                // a = b c mod (x^n - 1) from transforms b, c
//   t.inverse_product_sum(pairs, a);           // a = the sum of the pairs' products, up to 3 pairs
//   t.forward_product(a, 0, c, b);             // c = the transform of a b mod (x^n - 1)
//   t.forward(f.first(m), 0, a);               // out of place: a = transform of f[0, m), m <= n
//   t.cyclic_product(a.subspan(n / 2), n / 2, a, b, poly::Half::kUpper);
//                                              // a = x^(n/2) a[n/2, n) b mod (x^n - 1), upper half
//
// Spans: size n = 2^lg with 6 <= lg <= lg_max, 32-byte aligned, 4 readable bytes after the end
// (the forward kernels read past the last vector). Arena::take gives such spans.
//
// The transform of a polynomial a of length n holds n / 8 leaves: leaf p is a mod (x^8 - w_p),
// 8 coefficients in [0, P). Here w_p = r[p / 2] for even p,
// -r[p / 2] for odd p, and r[m] is the product of roots[j] over the set bits j of m, roots[j]
// of order 2^(j + 2) (lib/ntt's twiddle table). w_p does not depend on n, so the transform of
// length n is the first half of the transform of length 2n.
#pragma once

#include <immintrin.h>
#include <sys/mman.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "lib/ntt/ntt.hpp"

namespace poly {

inline constexpr std::uint32_t kModulus = ntt::kModulus;

// The half of a transform's output that is computed; the other half is left unspecified.
enum class Half { kBoth, kLower, kUpper };

// One anonymous mapping in transparent huge pages, handed out as zero-filled spans that meet the
// transforms' requirements. Each span starts 64 bytes after the previous one ends, so equal
// offsets in two spans fall in different cache sets.
class Arena {
public:
    // Room for spans whose sizes, each rounded up to a multiple of 8, sum to at most words - 16 per span.
    explicit Arena(std::size_t words) {
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes_ = (words * sizeof(std::uint32_t) + kHuge - 1) / kHuge * kHuge + kHuge;
        region_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) std::abort();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region_) + kHuge - 1) & ~(kHuge - 1);
        next_ = reinterpret_cast<std::uint32_t*>(aligned);
        end_ = next_ + (bytes_ - kHuge) / sizeof(std::uint32_t);
#ifdef MADV_HUGEPAGE
        ::madvise(next_, bytes_ - kHuge, MADV_HUGEPAGE);
#endif
    }

    ~Arena() { ::munmap(region_, bytes_); }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Words a span of size n uses.
    static constexpr std::size_t footprint(std::size_t n) { return (n + 7) / 8 * 8 + 16; }

    std::span<std::uint32_t> take(std::size_t n) {
        if (std::size_t(end_ - next_) < footprint(n)) std::abort();
        std::uint32_t* const start = next_;
        next_ += footprint(n);
        return {start, n};
    }

private:
    void* region_;
    std::size_t bytes_;
    std::uint32_t *next_, *end_;
};

namespace detail {

using ntt::detail::add;
using ntt::detail::broadcast;
using ntt::detail::diff;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::reduce;
using ntt::detail::slot;
using ntt::detail::Vec;

inline Vec load(const std::uint32_t* p) { return _mm256_load_si256(reinterpret_cast<const Vec*>(p)); }
inline void store(std::uint32_t* p, Vec x) { _mm256_store_si256(reinterpret_cast<Vec*>(p), x); }

inline Vec low(Vec x) { return reduce(x, 2 * kP); }                // [0, 4P) -> [0, 2P)
inline Vec canonical(Vec x) { return reduce(reduce(x, 2 * kP), kP); }  // [0, 4P) -> [0, P)
// x - y mod 2P in [0, 2P) for x, y < 2P.
inline Vec low_difference(Vec x, Vec y) {
    const Vec t = _mm256_sub_epi32(x, y);
    return _mm256_min_epu32(t, _mm256_add_epi32(t, broadcast(2 * kP)));
}

// x w mod P in [0, 2P), any x < 2^32.
inline Vec times(Vec x, const Factor& w) { return ntt::detail::multiply(x, w); }

// Entry k of a twiddle table with its Shoup quotient.
inline Factor entry(const std::uint32_t* table, std::size_t k) {
    const std::uint32_t* e = table + slot(k);
    return Factor(e[0], e[8]);
}

// w_p and its quotient; the quotient of P - w is ~quotient(w).
inline Factor leaf_weight(const std::uint32_t* roots, std::size_t p) {
    const std::uint32_t* e = roots + slot(p >> 1);
    return p & 1 ? Factor(kP - e[0], ~e[8]) : Factor(e[0], e[8]);
}

// The radix-4 group at h = 1 of 4 vectors: a polynomial mod x^32 - r[g]^2 whose outputs are the
// leaves 4g .. 4g + 3. x, y, z: entries g, 2g, 2g + 1 of the table (or of the inverse table).
struct Group {
    Factor x, y, z;
    Group(const std::uint32_t* table, std::size_t g) : x(entry(table, g)), y(entry(table, 2 * g)), z(entry(table, 2 * g + 1)) {}
};

// Forward butterfly, inputs and outputs < 4P.
inline void forward_h1(Vec (&f)[4], const Group& w) {
    const Vec a = low(f[0]), b = low(f[1]), c = times(f[2], w.x), d = times(f[3], w.x);
    const Vec ac = low(add(a, c)), amc = low_difference(a, c);
    const Vec bd = times(add(b, d), w.y), bmd = times(diff(b, d), w.z);
    f[0] = add(ac, bd), f[1] = diff(ac, bd), f[2] = add(amc, bmd), f[3] = diff(amc, bmd);
}

// Inverse butterfly with inverse twiddles, inputs and outputs < 2P; outputs 4 times the input
// polynomial's coefficients.
inline void inverse_h1(Vec (&f)[4], const Group& w) {
    const Vec ab = low(add(f[0], f[1])), cd = low(add(f[2], f[3]));
    const Vec amb = times(diff(f[0], f[1]), w.y), cmd = times(diff(f[2], f[3]), w.z);
    f[0] = low(add(ab, cd)), f[1] = low(add(amb, cmd));
    f[2] = times(diff(ab, cd), w.x), f[3] = times(diff(amb, cmd), w.x);
}

// window = [w a, a] for a leaf a mod (x^8 - w): x^i a mod (x^8 - w) is window[8 - i, 16 - i).
struct alignas(64) Window {
    std::uint32_t word[17];  // the window at 9 reads word 16 into an unused lane
};

inline void fill_window(Window& window, Vec a, const Factor& w) {
    store(window.word + 8, a);
    store(window.word, reduce(times(a, w), kP));
}

// The windows of leaves 4g .. 4g + 3 from canonical a: weights y, -y, z, -z (y, z: entries 2g,
// 2g + 1 of roots). The odd leaves' w a is 2P - y a (or z a), in (0, P] after the reduction.
[[gnu::always_inline]] inline void fill_windows(Window (&window)[4], const Vec (&a)[4], const std::uint32_t* roots,
                                                std::size_t g) {
    const Factor y = entry(roots, 2 * g), z = entry(roots, 2 * g + 1);
#pragma GCC unroll 4
    for (int t = 0; t < 4; ++t) {
        store(window[t].word + 8, a[t]);
        const Vec wa = times(a[t], t < 2 ? y : z);
        store(window[t].word, reduce(t % 2 ? _mm256_sub_epi32(broadcast(2 * kP), wa) : wa, kP));
    }
}

// a b mod (x^8 - w) / 2^32 in [0, 2P) for a as a window (words <= P) and canonical b. Even
// outputs 2m are the sums over windows k = 8 .. 1 times b[8 - k], odd outputs over k = 9 .. 2
// times b[9 - k]. Step i = 0 .. 7 broadcasts b[i], multiplies it with windows 9 - i (odd) and
// 8 - i (even), then adds step i - 1's products. Each output is a sum of 8 products < P^2, plus
// the Montgomery term < 2^32 P: below 2^64. Assembly keeps this order; GCC's schedule of the same
// intrinsics was 2-5% slower in the bottoms below (lib/poly/notes.md).
[[gnu::always_inline]] inline Vec leaf_product(const Window& window, const std::uint32_t* b) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP), p2 = broadcast(2 * kP);
    Vec even, odd, bi, t0, t1, t2, t3;
    asm("vpbroadcastd (%[b]), %[bi]\n\t"
        "vpmuludq 36(%[w]), %[bi], %[odd]\n\t"
        "vpmuludq 32(%[w]), %[bi], %[even]\n\t"
        "vpbroadcastd 4(%[b]), %[bi]\n\t"
        "vpmuludq 32(%[w]), %[bi], %[t0]\n\t"
        "vpmuludq 28(%[w]), %[bi], %[t1]\n\t"
        "vpbroadcastd 8(%[b]), %[bi]\n\t"
        "vpmuludq 28(%[w]), %[bi], %[t2]\n\t"
        "vpmuludq 24(%[w]), %[bi], %[t3]\n\t"
        "vpaddq %[t0], %[odd], %[odd]\n\t"
        "vpaddq %[t1], %[even], %[even]\n\t"
        "vpbroadcastd 12(%[b]), %[bi]\n\t"
        "vpmuludq 24(%[w]), %[bi], %[t0]\n\t"
        "vpmuludq 20(%[w]), %[bi], %[t1]\n\t"
        "vpaddq %[t2], %[odd], %[odd]\n\t"
        "vpaddq %[t3], %[even], %[even]\n\t"
        "vpbroadcastd 16(%[b]), %[bi]\n\t"
        "vpmuludq 20(%[w]), %[bi], %[t2]\n\t"
        "vpmuludq 16(%[w]), %[bi], %[t3]\n\t"
        "vpaddq %[t0], %[odd], %[odd]\n\t"
        "vpaddq %[t1], %[even], %[even]\n\t"
        "vpbroadcastd 20(%[b]), %[bi]\n\t"
        "vpmuludq 16(%[w]), %[bi], %[t0]\n\t"
        "vpmuludq 12(%[w]), %[bi], %[t1]\n\t"
        "vpaddq %[t2], %[odd], %[odd]\n\t"
        "vpaddq %[t3], %[even], %[even]\n\t"
        "vpbroadcastd 24(%[b]), %[bi]\n\t"
        "vpmuludq 12(%[w]), %[bi], %[t2]\n\t"
        "vpmuludq 8(%[w]), %[bi], %[t3]\n\t"
        "vpaddq %[t0], %[odd], %[odd]\n\t"
        "vpaddq %[t1], %[even], %[even]\n\t"
        "vpbroadcastd 28(%[b]), %[bi]\n\t"
        "vpmuludq 8(%[w]), %[bi], %[t0]\n\t"
        "vpmuludq 4(%[w]), %[bi], %[t1]\n\t"
        "vpaddq %[t2], %[odd], %[odd]\n\t"
        "vpaddq %[t3], %[even], %[even]\n\t"
        "vpaddq %[t0], %[odd], %[odd]\n\t"
        "vpaddq %[t1], %[even], %[even]\n\t"
        // Montgomery: (s + (s / -P mod 2^32) P) / 2^32 in [0, 3P), then [0, 2P).
        "vpmuludq %[ni], %[even], %[t0]\n\t"
        "vpmuludq %[ni], %[odd], %[t1]\n\t"
        "vpmuludq %[p], %[t0], %[t0]\n\t"
        "vpmuludq %[p], %[t1], %[t1]\n\t"
        "vpaddq %[t0], %[even], %[even]\n\t"
        "vpaddq %[t1], %[odd], %[odd]\n\t"
        "vpsrlq $32, %[even], %[even]\n\t"
        "vpblendd $0xaa, %[odd], %[even], %[even]\n\t"
        "vpsubd %[p2], %[even], %[t0]\n\t"
        "vpminud %[t0], %[even], %[even]"
        : [even] "=&x"(even), [odd] "=&x"(odd), [bi] "=&x"(bi), [t0] "=&x"(t0), [t1] "=&x"(t1), [t2] "=&x"(t2),
          [t3] "=&x"(t3)
        : [w] "r"(window.word), [b] "r"(b), [ni] "x"(ni), [p] "x"(p), [p2] "x"(p2),
          "m"(*reinterpret_cast<const std::uint32_t(*)[17]>(window.word)),
          "m"(*reinterpret_cast<const std::uint32_t(*)[8]>(b)));
    return even;
}

// The bottom of a subtree: count groups at h = 1 (4 vectors each, at a), the first with index
// first. Group g holds leaves 4g .. 4g + 3, so another operand's leaves are found by index.
// Forward: forward butterflies, canonical leaves.
struct ForwardBottom {
    static constexpr bool kForward = true, kInverse = false;
    const std::uint32_t* roots;

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        for (std::size_t j = 0; j < count; ++j, a += 32) {
            const Group w(roots, first + j);
            Vec f[4] = {load(a), load(a + 8), load(a + 16), load(a + 24)};
            forward_h1(f, w);
            for (int t = 0; t < 4; ++t) store(a + 8 * t, canonical(f[t]));
        }
    }
};

// Inverse: inverse butterflies of the leaves of in (read by index; in may be the output).
struct InverseBottom {
    static constexpr bool kForward = false, kInverse = true;
    const std::uint32_t* inverse_roots;
    const std::uint32_t* in;

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        const std::uint32_t* from = in + 32 * first;
        for (std::size_t j = 0; j < count; ++j, a += 32, from += 32) {
            const Group w(inverse_roots, first + j);
            Vec f[4] = {load(from), load(from + 8), load(from + 16), load(from + 24)};
            inverse_h1(f, w);
            for (int t = 0; t < 4; ++t) store(a + 8 * t, f[t]);
        }
    }
};

// Product: forward butterflies of a, leaf products with b (a transform; each product carries a
// factor 2^-32 that the final scale undoes), inverse butterflies. The windows of group j + 1 are
// written before the products of group j read those of group j. prepare and finish are inlined
// into the loops (as calls they cost 6-7% of cyclic_product, lib/poly/notes.md).
struct ProductBottom {
    static constexpr bool kForward = true, kInverse = true;
    const std::uint32_t* roots;
    const std::uint32_t* inverse_roots;
    const std::uint32_t* b;

    [[gnu::always_inline]] void prepare(const std::uint32_t* a, std::size_t g, Window (&window)[4]) const {
        Vec f[4] = {load(a), load(a + 8), load(a + 16), load(a + 24)};
        forward_h1(f, Group(roots, g));
#pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) f[t] = canonical(f[t]);
        fill_windows(window, f, roots, g);
    }

    [[gnu::always_inline]] void finish(std::uint32_t* a, std::size_t g, const Window (&window)[4]) const {
        Vec f[4];
#pragma GCC unroll 4
        for (int t = 0; t < 4; ++t) f[t] = leaf_product(window[t], b + 8 * (4 * g + t));
        inverse_h1(f, Group(inverse_roots, g));
        for (int t = 0; t < 4; ++t) store(a + 8 * t, f[t]);
    }

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        Window window[2][4];
        prepare(a, first, window[0]);
        for (std::size_t j = 0; j < count; ++j, a += 32) {
            if (j + 1 < count) prepare(a + 32, first + j + 1, window[(j + 1) & 1]);
            finish(a, first + j, window[j & 1]);
        }
    }
};

// Inverse of a product of two transforms: leaf products of a and b (both read by index; either
// may be the output), then inverse butterflies as in ProductBottom.
struct InverseProductBottom {
    static constexpr bool kForward = false, kInverse = true;
    ProductBottom product;  // holds b
    const std::uint32_t* a;

    [[gnu::always_inline]] void prepare(std::size_t g, Window (&window)[4]) const {
        const std::uint32_t* leaves = a + 32 * g;
        const Vec f[4] = {load(leaves), load(leaves + 8), load(leaves + 16), load(leaves + 24)};
        fill_windows(window, f, product.roots, g);
    }

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        Window window[2][4];
        prepare(first, window[0]);
        for (std::size_t j = 0; j < count; ++j, out += 32) {
            if (j + 1 < count) prepare(first + j + 1, window[(j + 1) & 1]);
            product.finish(out, first + j, window[j & 1]);
        }
    }
};

// The same for a sum of K products: leaf products of each pair, reduced and added. (A separate
// type: folding K = 1 into it changed the code of exp's products, 7% slower on lc-intel.)
template <std::size_t K>
struct InverseProductSumBottom {
    static constexpr bool kForward = false, kInverse = true;
    std::array<InverseProductBottom, K> terms;

    void prepare(std::size_t g, Window (&window)[K][4]) const {
        for (std::size_t k = 0; k < K; ++k) terms[k].prepare(g, window[k]);
    }

    void operator()(std::uint32_t* out, std::size_t count, std::size_t first) const {
        Window window[2][K][4];
        prepare(first, window[0]);
        for (std::size_t j = 0; j < count; ++j, out += 32) {
            if (j + 1 < count) prepare(first + j + 1, window[(j + 1) & 1]);
            const std::size_t g = first + j;
            Vec f[4];
            for (std::size_t t = 0; t < 4; ++t) {
                f[t] = leaf_product(window[j & 1][0][t], terms[0].product.b + 8 * (4 * g + t));
                for (std::size_t k = 1; k < K; ++k)
                    f[t] = low(add(f[t], leaf_product(window[j & 1][k][t], terms[k].product.b + 8 * (4 * g + t))));
            }
            inverse_h1(f, Group(terms[0].product.inverse_roots, g));
            for (std::size_t t = 0; t < 4; ++t) store(out + 8 * t, f[t]);
        }
    }
};

// Forward butterflies, then leaf products with b, canonical: the transform of a product. The
// factor 2^-32 of each leaf product is undone by a Shoup multiplication by 2^32.
struct ForwardProductBottom {
    static constexpr bool kForward = true, kInverse = false;
    ProductBottom product;  // holds b

    void operator()(std::uint32_t* a, std::size_t count, std::size_t first) const {
        const Factor undo_montgomery(kR);
        Window window[2][4];
        product.prepare(a, first, window[0]);
        for (std::size_t j = 0; j < count; ++j, a += 32) {
            if (j + 1 < count) product.prepare(a + 32, first + j + 1, window[(j + 1) & 1]);
            const std::uint32_t* b = product.b + 32 * (first + j);
#pragma GCC unroll 4
            for (int t = 0; t < 4; ++t)
                store(a + 8 * t, reduce(times(leaf_product(window[j & 1][t], b + 8 * t), undo_montgomery), kP));
        }
    }
};

// Depth-first recursion over radix-4 groups, as in lib/ntt: group k at stride h holds 4h vectors
// (a polynomial mod X^4 - r[k]^2, X = x^(8h)); its children are groups 4k + t. Subtrees of at most
// kTile vectors run level by level.
template <class Bottom>
class Recursion {
public:
    Recursion(const std::uint32_t* roots, const std::uint32_t* inverse_roots, const Bottom& bottom)
        : r_(roots), ir_(inverse_roots), bottom_(bottom) {}

    // nv = 4^j >= 4 vectors at a, group k.
    void visit(std::uint32_t* a, std::size_t nv, std::size_t k) const {
        if (nv <= kTile) return tile(a, nv, k);
        const std::size_t h = nv / 4;
        if constexpr (Bottom::kForward) forward(a, h, k);
        for (std::size_t t = 0; t < 4; ++t) visit(a + 8 * t * h, h, 4 * k + t);
        if constexpr (Bottom::kInverse) inverse(a, h, k);
    }

private:
    static constexpr std::size_t kTile = 256;

    void tile(std::uint32_t* a, std::size_t nv, std::size_t k) const {
        if constexpr (Bottom::kForward)
            for (std::size_t h = nv / 4; h >= 4; h /= 4)
                for (std::size_t j = 0, g = k * (nv / (4 * h)); j < nv; j += 4 * h, ++g) forward(a + 8 * j, h, g);
        bottom_(a, nv / 4, k * (nv / 4));
        if constexpr (Bottom::kInverse)
            for (std::size_t h = 4; h < nv; h *= 4)
                for (std::size_t j = 0, g = k * (nv / (4 * h)); j < nv; j += 4 * h, ++g) inverse(a + 8 * j, h, g);
    }

    static Vec* vectors(std::uint32_t* a) { return reinterpret_cast<Vec*>(a); }

    void forward(std::uint32_t* a, std::size_t h, std::size_t k) const {
        if (k == 0) return ntt::kernels::forward_identity(vectors(a), h, r_);
        ntt::kernels::forward(vectors(a), h, r_ + slot(k), r_ + slot(2 * k));
    }

    void inverse(std::uint32_t* a, std::size_t h, std::size_t k) const {
        if (k == 0) return ntt::kernels::inverse_identity(vectors(a), h, ir_);
        ntt::kernels::inverse(vectors(a), h, ir_ + slot(k), ir_ + slot(2 * k));
    }

    const std::uint32_t *r_, *ir_;
    Bottom bottom_;
};

// The input of a forward transform of length n: the polynomial x^shift in[0, size), shift + size
// <= n, in canonical coefficients. Vector v holds coefficients [8v, 8v + 8); those outside
// [shift, shift + size) are zero and not read.
class Source {
public:
    Source(const std::uint32_t* in, std::size_t size, std::size_t shift)
        : in_(in), shift_(shift), end_(shift + size), first_((shift + 7) / 8), first_touched_(shift / 8) {
        count_ = end_ / 8 > first_ ? end_ / 8 - first_ : 0;
        touched_ = (end_ + 7) / 8 - first_touched_;
    }

    Vec operator()(std::size_t v) const {
        if (v - first_ < count_) return _mm256_loadu_si256(reinterpret_cast<const Vec*>(in_ + (8 * v - shift_)));
        if (v - first_touched_ >= touched_) return _mm256_setzero_si256();
        return edge(v);
    }

private:
    // A vector partly inside.
    [[gnu::noinline]] Vec edge(std::size_t v) const {
        alignas(32) std::uint32_t x[8] = {};
        for (std::size_t i = 8 * v; i < 8 * v + 8; ++i)
            if (i >= shift_ && i < end_) x[i - 8 * v] = in_[i - shift_];
        return load(x);
    }

    const std::uint32_t* in_;
    std::size_t shift_, end_;
    std::size_t first_, count_;          // vectors [first_, first_ + count_) are inside
    std::size_t first_touched_, touched_;  // vectors outside [first_touched_, first_touched_ + touched_) are zero
};

// Top levels. Forward: canonical inputs, outputs < 4P. Inverse: inputs < 2P, canonical outputs
// times s. Quarters (radix 4) or halves (radix 2) of h vectors each. Written for any input, which
// may also be out itself.

// Radix-4 identity group on the quarters a, b, c, d (a polynomial mod X^4 - 1, X = x^(n/4)):
// outputs (a + c) + (b + d), (a + c) - (b + d), (a - c) + z (b - d), (a - c) - z (b - d), z = r[1].
inline void forward_top4(const Source& in, Vec* out, std::size_t h, const std::uint32_t* roots) {
    const Factor z(roots[1], roots[9]);
    const Vec p = broadcast(kP);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec a = in(j), b = in(j + h), c = in(j + 2 * h), d = in(j + 3 * h);
        const Vec ac = add(a, c), amc = _mm256_sub_epi32(add(a, p), c);  // < 2P
        const Vec bd = add(b, d), zbmd = times(_mm256_sub_epi32(add(b, p), d), z);
        out[j] = add(ac, bd), out[j + h] = diff(ac, bd);
        out[j + 2 * h] = add(amc, zbmd), out[j + 3 * h] = diff(amc, zbmd);
    }
}

// Radix-2 on the halves u, v: outputs u + v, u - v.
inline void forward_top2(const Source& in, Vec* out, std::size_t h) {
    const Vec p = broadcast(kP);
    for (std::size_t j = 0; j < h; ++j) {
        const Vec u = in(j), v = in(j + h);
        out[j] = add(u, v), out[j + h] = _mm256_sub_epi32(add(u, p), v);
    }
}

inline void inverse_top4(Vec* f, std::size_t h, Half output, const std::uint32_t* inverse_roots, const Factor& s) {
    if (output == Half::kBoth) return ntt::detail::inverse_radix4(f, h, inverse_roots, s);
    const Factor z(inverse_roots[1], inverse_roots[9]);
    const auto scale = [&s](Vec x) { return reduce(times(x, s), kP); };
    for (std::size_t j = 0; j < h; ++j) {
        const Vec p0 = f[j], p1 = f[j + h], p2 = f[j + 2 * h], p3 = f[j + 3 * h];
        const Vec ab = low(add(p0, p1)), cd = low(add(p2, p3));
        const Vec amb = low(diff(p0, p1)), cmd = times(diff(p2, p3), z);
        if (output == Half::kLower) {
            f[j] = scale(add(ab, cd)), f[j + h] = scale(add(amb, cmd));
        } else {
            f[j + 2 * h] = scale(diff(ab, cd)), f[j + 3 * h] = scale(diff(amb, cmd));
        }
    }
}

inline void inverse_top2(Vec* f, std::size_t h, Half output, std::uint32_t scale) {
    if (output == Half::kBoth) {
        alignas(32) std::uint32_t s[16] = {};  // table layout: s at entry 1
        s[1] = scale;
        s[9] = ntt::detail::quotient(scale);
        return ntt::kernels::scale_radix2(f, h, s);
    }
    const Factor s(scale);
    for (std::size_t j = 0; j < h; ++j) {
        if (output == Half::kLower) f[j] = reduce(times(add(f[j], f[j + h]), s), kP);
        else f[j + h] = reduce(times(diff(f[j], f[j + h]), s), kP);
    }
}

}  // namespace detail

class Transform {
public:
    static constexpr int kMinLog = 6, kMaxLog = ntt::kMaxLog;

    // Arena words the tables take.
    static constexpr std::size_t words(int lg_max) { return Arena::footprint(2 * ntt::detail::table_words(lg_max)); }

    // Tables for transforms of length up to 2^lg_max, kMinLog <= lg_max <= kMaxLog, from arena.
    Transform(Arena& arena, int lg_max) : lg_max_(lg_max) {
        if (lg_max < kMinLog || lg_max > kMaxLog) std::abort();
        const std::size_t table = ntt::detail::table_words(lg_max), entries = (std::size_t(1) << lg_max) / 16;
        roots_ = arena.take(2 * table).data();
        inverse_roots_ = roots_ + table;
        ntt::detail::build_table(roots_, entries, ntt::detail::kRoots[0]);
        ntt::detail::build_table(inverse_roots_, entries, ntt::detail::kRoots[1]);
    }

    int lg_max() const { return lg_max_; }

    // out = the transform of x^shift in, of length n = out.size(): coefficients in [0, P), those
    // outside [shift, shift + in.size()) zero (shift + in.size() <= n). in may lie inside out at
    // offset shift (in place); otherwise the two must not overlap.
    void forward(std::span<const std::uint32_t> in, std::size_t shift, std::span<std::uint32_t> out) const {
        run(out, source(in, shift, out.size()), detail::ForwardBottom{roots_}, 0, Half::kBoth);
    }

    // In place: coefficients in [0, P) -> transform.
    void forward(std::span<std::uint32_t> a) const { forward(a, 0, a); }

    // Doubling: out = the upper half of the transform of length 2n of x^shift in, n = out.size(),
    // in as for forward() (needs lg_max >= lg + 1). Its leaves are n/8 .. n/4 - 1: the transform of
    // x^shift in mod (x^n + 1). With forward(in, shift, lower) it gives the transform of length 2n.
    void forward_upper(std::span<const std::uint32_t> in, std::size_t shift, std::span<std::uint32_t> out) const {
        using namespace detail;
        check_length(2 * out.size());
        const Source src = source(in, shift, out.size());
        const detail::Recursion recursion(roots_, inverse_roots_, ForwardBottom{roots_});
        const std::size_t nv = out.size() / 8;
        auto* v = reinterpret_cast<Vec*>(out.data());
        if (std::countr_zero(nv) % 2 == 0) {  // 2n / 8 = 2 * 4^j: group 1 below a radix-2 level
            for (std::size_t j = 0; j < nv; ++j) v[j] = src(j);
            return recursion.visit(out.data(), nv, 1);
        }
        // 2n / 8 = 4^j: groups 2 and 3 below the radix-4 identity group, whose upper quarters are 0.
        const std::size_t h = nv / 2;
        const Factor z(roots_[1], roots_[9]);
        for (std::size_t j = 0; j < h; ++j) {
            const Vec a = src(j), zb = times(src(j + h), z);
            v[j] = add(a, zb), v[j + h] = diff(a, zb);
        }
        recursion.visit(out.data(), h, 2);
        recursion.visit(out.data() + 8 * h, h, 3);
    }

    // out = the coefficients in [0, P) of the transform in, both of length n = out.size(). in may
    // be out; otherwise the two must not overlap.
    void inverse(std::span<const std::uint32_t> in, std::span<std::uint32_t> out, Half output = Half::kBoth) const {
        using namespace ntt::detail;
        const std::uint32_t scale = power(std::uint32_t(out.size() / 8), kP - 2);  // undoes the factor n / 8
        run(out, detail::Source(nullptr, 0, 0), detail::InverseBottom{inverse_roots_, in.data()}, scale, output);
    }

    // In place: transform -> coefficients in [0, P).
    void inverse(std::span<std::uint32_t> a, Half output = Half::kBoth) const { inverse(a, a, output); }

    // out = (x^shift in) b mod (x^n - 1) for b a transform of length n = out.size(); in as for
    // forward(). Only the output half of out is computed.
    void cyclic_product(std::span<const std::uint32_t> in, std::size_t shift, std::span<std::uint32_t> out,
                        std::span<const std::uint32_t> b, Half output = Half::kBoth) const {
        using namespace ntt::detail;
        const std::uint32_t scale = multiply_mod(power(std::uint32_t(out.size() / 8), kP - 2), kR);  // and 2^-32
        run(out, source(in, shift, out.size()), detail::ProductBottom{roots_, inverse_roots_, b.data()}, scale, output);
    }

    // In place: a = a b mod (x^n - 1).
    void cyclic_product(std::span<std::uint32_t> a, std::span<const std::uint32_t> b) const { cyclic_product(a, 0, a, b); }

    // out = the coefficients of a b mod (x^n - 1) for transforms a and b of length n = out.size().
    // Only the output half of out is computed. out may be a or b; otherwise none may overlap.
    void inverse_product(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b, std::span<std::uint32_t> out,
                         Half output = Half::kBoth) const {
        using namespace ntt::detail;
        const std::uint32_t scale = multiply_mod(power(std::uint32_t(out.size() / 8), kP - 2), kR);  // and 2^-32
        const detail::ProductBottom product{roots_, inverse_roots_, b.data()};
        run(out, detail::Source(nullptr, 0, 0), detail::InverseProductBottom{product, a.data()}, scale, output);
    }

    // Two transforms of the same length.
    struct Pair {
        std::span<const std::uint32_t> a, b;
    };

    // out = the coefficients of the sum of a b mod (x^n - 1) over 1 to 3 pairs of transforms of
    // length n = out.size(). Only the output half of out is computed. out may be an operand.
    void inverse_product_sum(std::span<const Pair> pairs, std::span<std::uint32_t> out, Half output = Half::kBoth) const {
        switch (pairs.size()) {
            case 1: return inverse_product(pairs[0].a, pairs[0].b, out, output);
            case 2: return inverse_products<2>(pairs, out, output);
            case 3: return inverse_products<3>(pairs, out, output);
            default: std::abort();
        }
    }

    // out = the transform of (x^shift in) b mod (x^n - 1) for b a transform of length
    // n = out.size(); in as for forward(). out and b must not overlap.
    void forward_product(std::span<const std::uint32_t> in, std::size_t shift, std::span<std::uint32_t> out,
                         std::span<const std::uint32_t> b) const {
        const detail::ProductBottom product{roots_, inverse_roots_, b.data()};
        run(out, source(in, shift, out.size()), detail::ForwardProductBottom{product}, 0, Half::kBoth);
    }

    // a = a b, both transforms of the same length.
    void multiply(std::span<std::uint32_t> a, std::span<const std::uint32_t> b) const {
        products(a.data(), a.data(), b.data(), a.size(), [](detail::Vec, detail::Vec p) { return p; });
    }

    // c = c + a b, all transforms of the same length.
    void multiply_add(std::span<std::uint32_t> c, std::span<const std::uint32_t> a,
                      std::span<const std::uint32_t> b) const {
        products(c.data(), a.data(), b.data(), a.size(), [](detail::Vec c, detail::Vec p) { return detail::add(c, p); });
    }

private:
    void check_length(std::size_t n) const {
        const int lg = std::countr_zero(n);
        if (!std::has_single_bit(n) || lg < kMinLog || lg > lg_max_) std::abort();
    }

    template <std::size_t K>
    void inverse_products(std::span<const Pair> pairs, std::span<std::uint32_t> out, Half output) const {
        using namespace ntt::detail;
        const std::uint32_t scale = multiply_mod(power(std::uint32_t(out.size() / 8), kP - 2), kR);  // and 2^-32
        detail::InverseProductSumBottom<K> bottom{};
        for (std::size_t k = 0; k < K; ++k) bottom.terms[k] = {{roots_, inverse_roots_, pairs[k].b.data()}, pairs[k].a.data()};
        run(out, detail::Source(nullptr, 0, 0), bottom, scale, output);
    }

    static detail::Source source(std::span<const std::uint32_t> in, std::size_t shift, std::size_t n) {
        if (shift > n || in.size() > n - shift) std::abort();
        return detail::Source(in.data(), in.size(), shift);
    }

    // Top level, the subtrees, the top level's inverse with the scale (canonical output).
    // n / 8 = 4^j: one radix-4 group; n / 8 = 2 * 4^j: a radix-2 level.
    template <class Bottom>
    void run(std::span<std::uint32_t> a, const detail::Source& in, const Bottom& bottom, std::uint32_t scale,
             Half output) const {
        using namespace detail;
        check_length(a.size());
        const detail::Recursion recursion(roots_, inverse_roots_, bottom);
        const std::size_t nv = a.size() / 8;
        auto* v = reinterpret_cast<Vec*>(a.data());
        if (std::countr_zero(nv) % 2 == 0) {
            const std::size_t h = nv / 4;
            if constexpr (Bottom::kForward) forward_top4(in, v, h, roots_);
            for (std::size_t t = 0; t < 4; ++t) recursion.visit(a.data() + 8 * t * h, h, t);
            if constexpr (Bottom::kInverse) inverse_top4(v, h, output, inverse_roots_, Factor(scale));
        } else {
            const std::size_t h = nv / 2;
            if constexpr (Bottom::kForward) forward_top2(in, v, h);
            recursion.visit(a.data(), h, 0);
            recursion.visit(a.data() + 8 * h, h, 1);
            if constexpr (Bottom::kInverse) inverse_top2(v, h, output, scale);
        }
    }

    // out = combine(out, a b mod P) leaf by leaf, canonical.
    template <class Combine>
    void products(std::uint32_t* out, const std::uint32_t* a, const std::uint32_t* b, std::size_t n, Combine combine) const {
        using namespace detail;
        check_length(n);
        const Factor undo_montgomery(kR);
        Window window[2];
        fill_window(window[0], load(a), leaf_weight(roots_, 0));
        for (std::size_t p = 0; p < n / 8; ++p) {
            if (p + 1 < n / 8) fill_window(window[(p + 1) & 1], load(a + 8 * p + 8), leaf_weight(roots_, p + 1));
            const Vec product = reduce(times(leaf_product(window[p & 1], b + 8 * p), undo_montgomery), kP);
            store(out + 8 * p, canonical(combine(load(out + 8 * p), product)));
        }
    }

    int lg_max_;
    std::uint32_t *roots_, *inverse_roots_;
};

}  // namespace poly
