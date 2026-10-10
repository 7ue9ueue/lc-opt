// f(a r^i) mod 998244353 for i < M, f of N coefficients (N, M <= 2^19), by the chirp z-transform.
// With t(k) = k (k - 1) / 2 and any g, i j = t(i + j - g) - t(i) - t(j - g) + g i, so
//   f(a r^i) = r^(g i - t(i)) sum_j A_j K_(i + j),  A_j = c_j a^j r^-t(j - g),  K_u = r^t(u - g):
// a middle product. One cyclic convolution of length L = 2^lg >= N + M - 1 of A with the reversed
// kernel B_v = K_(L - 1 - v) gives f(a r^i) = r^(g i - t(i)) (A B)[L - 1 - i]. As t(1 - x) = t(x),
// g = L / 2 - 1 makes B a palindrome (B = K): B is generated for v < L / 2 only, and its transform
// at the conjugate of a node is a mirror of the node's (Subtrees), so b's forward transform runs
// for about half of the nodes.
// The convolution is ntt::Product's (lib/ntt/product.hpp: radix-8 top, Subtrees, bottom kernels)
// with these changes: the radix-8 pass of B reads its lower half twice; nodes come in conjugate
// pairs; the last inverse level computes only the outputs, times r^(g i - t(i)) and in reverse
// order. L >= 2 max(N, M), so A and the outputs each lie in one half.
// Output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
#include <sys/mman.h>

#include <algorithm>
#include <bit>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/ntt/product.hpp"
#include "lib/poly/chirp.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

using ntt::detail::add;
using ntt::detail::broadcast;
using ntt::detail::diff;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::multiply_mod;
using ntt::detail::power;
using ntt::detail::reduce;
using ntt::detail::slot;
using ntt::detail::Vec;
using poly::detail::Chirp;
using poly::detail::load;
using poly::detail::montgomery;
using poly::detail::store;

// Shoup product, < 2P for any x < 2^32.
Vec times(Vec x, const Factor& f) { return ntt::detail::multiply(x, f); }

// x mod 2P for x < 4P.
Vec low(Vec x) { return reduce(x, 2 * kP); }

// x - y + P for y < P.
Vec diff_canonical(Vec x, Vec y) { return _mm256_sub_epi32(_mm256_add_epi32(x, broadcast(kP)), y); }

Vec reverse(Vec x) { return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

std::uint32_t inverse(std::uint32_t x) { return power(x, kP - 2); }

// r^e for r != 0 and any e >= 0.
std::uint32_t power_of(std::uint32_t r, std::uint64_t e) { return power(r, std::uint32_t(e % (kP - 1))); }

// ntt::detail::forward_radix8 for a palindrome f[0, 8q), f[8q - 1 - i] = f[i] word by word, of
// which f[0, 4q) is given: block f_(t+4) = f[(t + 4) q, (t + 5) q) is f_(3-t) reversed. The groups
// act on u = f mod (x^(n/2) - 1) and v = f mod (x^(n/2) + 1), u_t = f_t + f_(t+4) and
// v_t = f_t - f_(t+4). Writes blocks 0, 1, 2, 4 and 5 only: blocks 3, 6 and 7 are their conjugates
// (Subtrees::mirror). Columns j and q - 1 - j read each other's vectors, so they go together.
// Inputs canonical; outputs < 4P.
void forward_radix8_palindrome(Vec* f, std::size_t q, const std::uint32_t* roots) {
    const Factor i(roots[1], roots[9]), y(roots[2], roots[10]);
    const auto column = [f, q, i, y](std::size_t j, const Vec* lo, const Vec* mirror) {
        Vec u[4], v[4];  // < 2P
#pragma GCC unroll 4
        for (std::size_t t = 0; t < 4; ++t) {
            const Vec hi = reverse(mirror[3 - t]);
            u[t] = add(lo[t], hi), v[t] = diff_canonical(lo[t], hi);
        }
        const Vec g0 = low(add(u[0], u[2])), g1 = low(add(u[1], u[3]));
        const Vec h0 = low(diff(u[0], u[2])), ih1 = times(diff(u[1], u[3]), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1), f[j + 2 * q] = add(h0, ih1);
        const Vec iv2 = times(v[2], i), iv3 = times(v[3], i);
        const Vec u0 = low(add(v[0], iv2)), yu1 = times(add(v[1], iv3), y);
        f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1);
    };
    for (std::size_t j = 0, k = q - 1; j < k; ++j, --k) {
        Vec at_j[4], at_k[4];
#pragma GCC unroll 4
        for (std::size_t t = 0; t < 4; ++t) at_j[t] = f[j + t * q], at_k[t] = f[k + t * q];
        column(j, at_j, at_k);
        column(k, at_k, at_j);
    }
}

// ntt::detail::Subtrees (lib/ntt/product.hpp) for a palindromic b, whose nodes pair up. Node k of
// a level holds b mod x^s - z_k with z_k = r[k]^2 (Recursion's numbering: children 4k + t). For a
// palindrome, b mod x^s - 1/z = z rev(b mod x^s - z), rev reversing the s coefficients. With
// r[k] = w^bitrev(k), node k's conjugate is 3 2^e - 1 - k for 2^e <= k < 2^(e+1): the mirror of
// a node's groups (bottom-level nodes of 4 vectors) is its conjugate's groups in reverse order.
// So b's forward levels run for one node of each pair; mirror() derives the other's groups and
// visit<false> skips its b levels. Only nodes 0 and 1 are their own conjugates (z = 1, -1); their
// children pair up as 2 and 3 (node 0) or 4 and 7, 5 and 6 (node 1). Subtrees of at least 16
// vectors, so tiles start at even group indices.
class Subtrees {
public:
    Subtrees(const std::uint32_t* roots, const std::uint32_t* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // The 8 blocks of q vectors after forward_radix8 (a) and forward_radix8_palindrome (b): forward
    // transforms, leaf products into a, inverse transforms, then the top inverse group of each half.
    void visit_top(Vec* a, Vec* b, std::size_t q) const {
        visit_own_conjugate(a, b, q, 0);
        visit_own_conjugate(a + q, b + q, q, 1);
        visit_pair(a, b, q, 0, 2, 3);
        ntt::kernels::inverse_identity(a, q, ir_);
        visit_pair(a, b, q, 0, 4, 7);
        visit_pair(a, b, q, 0, 5, 6);
        ntt::kernels::inverse(a + 4 * q, q, ir_ + slot(1), ir_ + slot(2));
    }

private:
    // Node k in {0, 1} of nv vectors at a, b.
    void visit_own_conjugate(Vec* a, Vec* b, std::size_t nv, std::size_t k) const {
        if (nv <= 256) return visit<true>(a, b, nv, k);
        const std::size_t h = nv / 4;
        forward<true>(a, b, h, k);
        if (k == 0) {
            visit_own_conjugate(a, b, h, 0);
            visit_own_conjugate(a + h, b + h, h, 1);
            visit_pair(a, b, h, 0, 2, 3);
        } else {
            visit_pair(a, b, h, 4, 4, 7);
            visit_pair(a, b, h, 4, 5, 6);
        }
        inverse(a, h, k);
    }

    // Node k and its conjugate c in a row of nodes first, first + 1, ... of nv vectors at a, b.
    void visit_pair(Vec* a, Vec* b, std::size_t nv, std::size_t first, std::size_t k, std::size_t c) const {
        const std::size_t at = (k - first) * nv, to = (c - first) * nv;
        visit<true>(a + at, b + at, nv, k);
        mirror(b + at, b + to, nv, k * (nv / 4));
        visit<false>(a + to, b + to, nv, c);
    }

    // to[nv - 1 - v] = reverse(from[v]) z_(first + v / 4) for v < nv: from holds nv / 4 groups
    // from first (even), each b mod x^32 - z_k with z_k = r[k]^2 = (-1)^k r[k / 2]. Outputs < 2P.
    void mirror(const Vec* from, Vec* to, std::size_t nv, std::size_t first) const {
        for (std::size_t v = 0; v < nv; v += 8) {
            const std::uint32_t* z = r_ + slot((first + v / 4) / 2);
            const Factor plus(z[0], z[8]), minus(kP - z[0], ~z[8]);  // quotient(P - z) = ~quotient(z)
#pragma GCC unroll 4
            for (std::size_t t = 0; t < 4; ++t) {
                to[nv - 1 - v - t] = times(reverse(from[v + t]), plus);
                to[nv - 5 - v - t] = times(reverse(from[v + 4 + t]), minus);
            }
        }
    }

    // Forward transforms, leaf products into a, inverse transform; nv = 4^j >= 16. b keeps its
    // transform down to the level above the bottom stage, which reads b and leaves it unchanged.
    // Without kForwardB, b already holds that transform (from mirror()).
    template <bool kForwardB>
    void visit(Vec* a, Vec* b, std::size_t nv, std::size_t k) const {
        switch (nv) {
        case 16: return tile<kForwardB, 16>(a, b, k);
        case 64: return tile<kForwardB, 64>(a, b, k);
        case 256: return tile<kForwardB, 256>(a, b, k);
        }
        const std::size_t h = nv / 4;
        forward<kForwardB>(a, b, h, k);
        for (std::size_t t = 0; t < 4; ++t) visit<kForwardB>(a + t * h, b + t * h, h, 4 * k + t);
        inverse(a, h, k);
    }

    template <bool kForwardB, std::size_t NV>
    [[gnu::noinline]] void tile(Vec* a, Vec* b, std::size_t k) const {
        for (std::size_t h = NV / 4; h >= 4; h /= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < NV; j += 4 * h, ++g) forward<kForwardB>(a + j, b + j, h, g);
        bottom(a, b, NV, k * (NV / 4));
        for (std::size_t h = 4; h < NV; h *= 4)
            for (std::size_t j = 0, g = k * (NV / (4 * h)); j < NV; j += 4 * h, ++g) inverse(a + j, h, g);
    }

    template <bool kForwardB>
    void forward(Vec* a, Vec* b, std::size_t h, std::size_t k) const {
        if (k == 0) {
            ntt::kernels::forward_identity(a, h, r_);
            if (kForwardB) ntt::kernels::forward_identity(b, h, r_);
            return;
        }
        const std::uint32_t *x = r_ + slot(k), *y = r_ + slot(2 * k);
        if (!kForwardB) return ntt::kernels::forward(a, h, x, y);
        if (h == 4) return ntt::kernels::forward_pair(a, b, h, x, y);
        ntt::kernels::forward(a, h, x, y);
        ntt::kernels::forward(b, h, x, y);
    }

    void inverse(Vec* a, std::size_t h, std::size_t k) const {
        if (k == 0) return ntt::kernels::inverse_identity(a, h, ir_);
        ntt::kernels::inverse(a, h, ir_ + slot(k), ir_ + slot(2 * k));
    }

    struct alignas(64) Leaves {
        std::uint32_t window[4][16];  // [w A_t, A_t]: x^i A_t mod x^8 - w is a sliding window
        std::uint32_t coefficients[4][8];
    };

    // Groups [first, first + nv / 4) with h = 1 and their leaves, two per kernel; first is even.
    // The next two groups' forward half overlaps the current two's products and inverse.
    void bottom(Vec* a, Vec* b, std::size_t nv, std::size_t first) const {
        Leaves leaves[2][2];
        ntt::product_kernels::bottom_first(a, b, leaves[0], r_ + slot(first), r_ + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const std::uint32_t *ix = ir_ + slot(k), *iy = ir_ + slot(2 * k);
            if (j + 8 == nv) return ntt::product_kernels::bottom_last(a + j, leaves[cur], ix, iy);
            ntt::product_kernels::bottom_both(a + j + 8, b + j + 8, leaves[next], r_ + slot(k + 2),
                                              r_ + slot(2 * k + 4), a + j, leaves[cur], ix, iy);
        }
    }

    const std::uint32_t *r_, *ir_;
};

// The last level and the output: with u = a[0, L/2) and w = a[L/2, L) after the halves' top
// inverse groups (< 2P), coefficient L - 1 - i of the product is (u - w)[L/2 - 1 - i] times the
// transform's scale. y[i] = (u - w)[L/2 - 1 - i] x_i / 2^32 for the terms x_i of chirp, which
// carry that scale, for i < m <= L/2 rounded up to 32.
void output(const std::uint32_t* a, std::size_t half, std::uint32_t* y, std::size_t m, Chirp chirp) {
    const std::uint32_t *u = a, *w = a + half;
    for (std::size_t i = 0; i < m; i += 32, chirp.next())
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) {
            const std::size_t at = half - 8 - i - 8 * v;
            const Vec x = reduce(diff(load(u + at), load(w + at)), 2 * kP);
            store(y + i + 8 * v, reduce(montgomery(reverse(x), chirp[v]), kP));
        }
}

// The evaluations f(a r^i), i < m, of f = c[0, n), for a, r != 0. L = 2^lg >= 2 max(n, m) with
// lg even and >= 10 (L / 8 = 2 * 4^j vectors, subtrees of at least 16). One mapping in huge
// pages; single use.
class ChirpZ {
public:
    ChirpZ(std::size_t n, std::size_t m) : n_(n), m_(m) {
        lg_ = std::max(10, int(std::bit_width(2 * std::max(n, m) - 1)));
        lg_ += lg_ % 2;
        const std::size_t len = length(), words = 2 * (len + kPadding) + 2 * ntt::detail::table_words(lg_);
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes_ = (words * sizeof(std::uint32_t) + fields::kTextBytes + kHuge - 1) / kHuge * kHuge + kHuge;
        region_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) std::abort();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region_) + kHuge - 1) & ~(kHuge - 1);
        a_ = reinterpret_cast<std::uint32_t*>(aligned);
#ifdef MADV_HUGEPAGE
        ::madvise(a_, bytes_ - kHuge, MADV_HUGEPAGE);
#endif
        b_ = a_ + len + kPadding;  // a different cache set from a at equal offsets
        roots_ = b_ + len + kPadding;
        inverse_roots_ = roots_ + ntt::detail::table_words(lg_);
        text_ = reinterpret_cast<char*>(inverse_roots_ + ntt::detail::table_words(lg_));
    }

    ~ChirpZ() { ::munmap(region_, bytes_); }

    ChirpZ(const ChirpZ&) = delete;
    ChirpZ& operator=(const ChirpZ&) = delete;

    // Room for c: n coefficients < P.
    std::uint32_t* coefficients() { return a_; }
    char* text() { return text_; }  // fields::kTextBytes bytes, 16-byte aligned

    // The m evaluations, canonical.
    const std::uint32_t* evaluate(std::uint32_t a, std::uint32_t r) {
        using namespace ntt::detail;
        const std::size_t len = length(), q = len / 64;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        // With g = L / 2 - 1 and t(u - g) = t(u) + t(-g) - g u, t(-g) = g (g + 1) / 2:
        // B_u = K_u = r^t(-g) (r^-g)^u r^t(u) for u < L / 2, A_k = c_k r^-t(-g) (a r^g)^k r^-t(k).
        const std::uint64_t g = len / 2 - 1, tg = g * (g + 1) / 2;
        const std::uint32_t r_inverse = inverse(r), r_g = power(r, std::uint32_t(g));
        poly::chirp(power_of(r, tg), power(r_inverse, std::uint32_t(g)), r, {b_, len / 2});
        poly::multiply_chirp(power_of(r_inverse, tg), multiply_mod(a, r_g), r_inverse, {a_, n_});
        // The outputs' factors r^(g i - t(i)) carry the transform's scale (L / 8)^-1 2^32 and 2^32
        // for montgomery().
        const std::uint32_t scale = multiply_mod(multiply_mod(inverse(std::uint32_t(len / 8)), kR), kR);
        auto* av = reinterpret_cast<Vec*>(a_);
        auto* bv = reinterpret_cast<Vec*>(b_);
        forward_radix8(av, q, roots_);
        forward_radix8_palindrome(bv, q, roots_);
        Subtrees(roots_, inverse_roots_).visit_top(av, bv, q);
        output(a_, len / 2, b_, m_, Chirp(scale, r_g, r_inverse));
        return b_;
    }

private:
    static constexpr std::size_t kPadding = 16;  // words after each factor: the kernels read 4 bytes past

    std::size_t length() const { return std::size_t(1) << lg_; }

    std::size_t n_, m_;
    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
    char* text_;
};

// f(x) for f = c[0, n), c zero up to the next multiple of 32. Lane l of chain w sums
// c[32 v + 8 w + l] x^(32 v) by Horner's rule; the 32 sums are then weighted by x^(8 w + l).
std::uint32_t evaluate(const std::uint32_t* c, std::size_t n, std::uint32_t x) {
    const Vec x32 = broadcast(multiply_mod(power(x, 32), kR));
    Vec sum[4] = {};
    for (std::size_t v = (n + 31) / 32; v-- > 0;)
        for (int w = 0; w < 4; ++w) sum[w] = low(_mm256_add_epi32(montgomery(sum[w], x32), load(c + 32 * v + 8 * w)));
    alignas(32) std::uint32_t lanes[32];
    for (int w = 0; w < 4; ++w) store(lanes + 8 * w, sum[w]);
    std::uint32_t total = 0, term = 1;
    for (int k = 0; k < 32; ++k, term = multiply_mod(term, x)) total = (total + multiply_mod(lanes[k] % kP, term)) % kP;
    return total;
}

void write_same(std::uint32_t first, std::uint32_t rest, std::size_t count) {
    std::vector<std::uint32_t> values(count, rest);
    values[0] = first;
    alignas(64) static char text[fields::kTextBytes];
    io::Writer out;
    fields::write(out, values.data(), count, text);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    const std::uint32_t a = in.read<std::uint32_t>(), r = in.read<std::uint32_t>();
    if (a == 0 || n == 1) {  // f(0) = c_0 everywhere
        const std::uint32_t c0 = in.read<std::uint32_t>();
        return write_same(c0, c0, m);
    }
    if (r == 0 || r == 1 || m == 1) {  // f(a), then f(0) or f(a)
        std::vector<std::uint32_t> c((n + 31) / 32 * 32);
        io::read_bulk(in, c.data(), n);
        const std::uint32_t at_a = evaluate(c.data(), n, a);
        return write_same(at_a, r == 0 ? c[0] : at_a, m);
    }
    ChirpZ chirp_z(n, m);
    io::read_bulk(in, chirp_z.coefficients(), n);
    const std::uint32_t* y = chirp_z.evaluate(a, r);
    io::Writer out;
    fields::write(out, y, m, chirp_z.text());
}

}  // namespace

RUN_EARLY(solve)
