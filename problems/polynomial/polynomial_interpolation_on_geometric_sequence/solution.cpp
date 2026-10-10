// The f of degree < N with f(a r^i) = y_i mod 998244353 (N <= 2^19, the a r^i distinct), in
// Newton's form. With t(k) = k (k - 1) / 2 and Q_k = (1 - r) (1 - r^2) ... (1 - r^k), the divided
// differences are d_k = a^-k r^-t(k) E_k, E = Y K mod x^N, Y_i = (-1)^i y_i / Q_i,
// K_j = r^t(j) / Q_j; the q-binomial theorem expands the Newton basis prod_(j<k) (x - a r^j), so
//   c_l = (-a)^-l / Q_l sum_(k >= l) D_k K_(k-l),  D_k = (-1)^k Q_k r^-t(k) E_k.
// Both sums are the lower half of a product with K, one cyclic convolution each of length
// L = 2^lg >= 2N in ntt::Product's layout (lib/ntt/product.hpp). The second reuses K's
// transform down to its leaves (bottom.hpp): five transforms in all. Q, 1/Q and the chirps come
// from scans in 32 lanes.
// Output in fixed-width fields (problems/convolution/convolution_mod/fields.hpp).
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/ntt/product.hpp"
#include "lib/poly/calculus.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"
#include "bottom.hpp"

namespace {

using ntt::detail::add;
using ntt::detail::broadcast;
using ntt::detail::diff;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::multiply;
using ntt::detail::multiply_mod;
using ntt::detail::power;
using ntt::detail::reduce;
using ntt::detail::slot;
using ntt::detail::Vec;
using poly::detail::load;
using poly::detail::load_unaligned;
using poly::detail::low;
using poly::detail::low_difference;
using poly::detail::montgomery;
using poly::detail::store;
using poly::detail::store_unaligned;
using poly::detail::times;

std::uint32_t inverse(std::uint32_t x) { return power(x, kP - 2); }

// x^e for x != 0 and any e >= 0.
std::uint32_t power_of(std::uint32_t x, std::uint64_t e) { return power(x, std::uint32_t(e % (kP - 1))); }

std::uint64_t triangle(std::uint64_t k) { return k * (k - 1) / 2; }  // t(k) = k (k - 1) / 2

std::uint32_t negate(std::uint32_t x) { return x ? kP - x : 0; }

std::uint32_t mont(std::uint32_t x) { return multiply_mod(x, kR); }  // x 2^32 mod P

Vec reverse(Vec x) { return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

// Rows become columns: lane l of r[t] moves to lane t of r[l], t, l < 8.
// Unrolled, so that the arrays stay in registers.
[[gnu::always_inline]] inline void transpose(Vec* r) {
    Vec t[8], u[8];
#pragma GCC unroll 4
    for (int i = 0; i < 8; i += 2) {
        t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
    }
#pragma GCC unroll 2
    for (int i = 0; i < 8; i += 4) {
        u[i] = _mm256_unpacklo_epi64(t[i], t[i + 2]);
        u[i + 1] = _mm256_unpackhi_epi64(t[i], t[i + 2]);
        u[i + 2] = _mm256_unpacklo_epi64(t[i + 1], t[i + 3]);
        u[i + 3] = _mm256_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
        r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
    }
}

// A term x and its multiplier term g of a scan (below), at one position of one lane.
struct State {
    std::uint32_t x, g;
};

// Scans over the positions of a sequence of n >= 3 terms, in 32 lanes: lane s = 8 v + l (vector
// v, lane l) covers the positions k = s C + j, j < C. C is a multiple of 16 with C / 16 odd, so
// the 32 streams of a pass fall into different cache sets. Positions k >= n are padding; lane
// partial() is the one that reaches n, at step first_padding() (32 if none does).
class Lanes {
public:
    static constexpr int kCount = 32;

    explicit Lanes(std::size_t n) {
        chunk_ = ((n + kCount - 1) / kCount + 15) / 16 * 16;
        if (chunk_ / 16 % 2 == 0) chunk_ += 16;
        partial_ = n / chunk_;
        first_padding_ = n % chunk_;
    }

    std::size_t chunk() const { return chunk_; }
    std::size_t start(int s) const { return s * chunk_; }
    std::size_t end(int s) const { return s * chunk_ + chunk_ - 1; }
    int partial() const { return int(partial_); }
    std::size_t first_padding() const { return first_padding_; }

    // x[v] = f(8 v + l) in lane l.
    template <class F>
    static void fill(Vec (&x)[4], F f) {
        alignas(32) std::uint32_t values[kCount];
        for (int s = 0; s < kCount; ++s) values[s] = f(s);
        for (int v = 0; v < 4; ++v) x[v] = load(values + 8 * v);
    }

    // Lane partial() of x; partial() < 32.
    std::uint32_t lane(const Vec (&x)[4]) const {
        alignas(32) std::uint32_t values[kCount];
        for (int v = 0; v < 4; ++v) store(values + 8 * v, x[v]);
        return values[partial_];
    }

    // The scan x_(k+1) = x_k (g_k + o) / 2^32 (forward; backward x_(k-1) = x_k (g_k + o) / 2^32)
    // with g_(k+1) = g_k w (backward g_(k-1) = g_k w), from initial(s), the State at the lane's
    // first position: start(s) forward, end(s) backward. A backward scan restarts lane partial()
    // at position n - 1 from last, so padding terms need not be defined. put(k, x) gets
    // x_k, ..., x_(k+7) of one lane, k = start(s) + 8 i. o < 2P; State values and x_k in [0, 2P).
    template <bool kForward, class Initial, class Put>
    void scan(Initial initial, State last, std::uint32_t o, std::uint32_t w, Put put) const {
        Vec x[4], g[4];
        {
            alignas(32) std::uint32_t xs[kCount], gs[kCount];
            for (int s = 0; s < kCount; ++s) {
                const State state = initial(s);
                xs[s] = state.x, gs[s] = state.g;
            }
            for (int v = 0; v < 4; ++v) x[v] = load(xs + 8 * v), g[v] = load(gs + 8 * v);
        }
        const Vec offset = broadcast(o);
        const Factor ratio(w);
        const std::size_t restart = kForward || first_padding_ == 0 ? chunk_ : first_padding_ - 1;
        Vec block[4][8];  // block[v][t]: step j0 + t
        const auto step = [&](int t) {
#pragma GCC unroll 4
            for (int v = 0; v < 4; ++v) {
                block[v][t] = x[v];
                x[v] = montgomery(x[v], low(add(g[v], offset)));
                g[v] = times(g[v], ratio);
            }
        };
        const std::size_t chunk = chunk_;
        for (std::size_t b = 0; b < chunk; b += 8) {
            const std::size_t j0 = kForward ? b : chunk - 8 - b;
            // A branch per step would keep x and g in memory: the block of the restart runs apart.
            if (restart - j0 < 8) {
#pragma GCC unroll 1
                for (int i = 0; i < 8; ++i) {
                    const int t = kForward ? i : 7 - i;
                    if (j0 + t == restart) set_partial(x, last.x), set_partial(g, last.g);
                    step(t);
                }
            } else {
#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) step(kForward ? i : 7 - i);
            }
#pragma GCC unroll 4
            for (int v = 0; v < 4; ++v) {
                Vec rows[8];
#pragma GCC unroll 8
                for (int t = 0; t < 8; ++t) rows[t] = block[v][t];
                transpose(rows);
#pragma GCC unroll 1
                for (int l = 0; l < 8; ++l) put((8 * v + l) * chunk + j0, rows[l]);
            }
        }
    }

private:
    // Constant indices only, so that x stays in registers.
    [[gnu::always_inline]] void set_partial(Vec (&x)[4], std::uint32_t value) const {
        const Vec lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        for (int v = 0; v < 4; ++v) {
            const Vec mask = _mm256_cmpeq_epi32(_mm256_add_epi32(lanes, broadcast(8 * v)), broadcast(partial_));
            x[v] = _mm256_blendv_epi8(x[v], broadcast(value), mask);
        }
    }

    std::size_t chunk_, partial_, first_padding_;
};

// ntt::detail::Subtrees (lib/ntt/product.hpp) with an option to reuse b's transform: with
// kForwardB, the bottom stage leaves b's leaves (its last forward level) in b (bottom.hpp: keep);
// without, b holds them from such a visit and only a is transformed (reuse). Subtrees of at least
// 16 vectors, so tiles start at even group indices.
template <bool kForwardB>
class Subtrees {
public:
    Subtrees(const std::uint32_t* roots, const std::uint32_t* inverse_roots) : r_(roots), ir_(inverse_roots) {}

    // Forward transforms, leaf products into a, inverse transform; nv = 4^j >= 16.
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
        if (kForwardB) bottom::first_keep(a, b, leaves[0], r_ + slot(first), r_ + slot(2 * first));
        else bottom::first_reuse(a, b, leaves[0], r_ + slot(first), r_ + slot(2 * first));
        for (std::size_t j = 0, k = first;; j += 8, k += 2) {
            const std::size_t cur = j / 8 % 2, next = cur ^ 1;
            const std::uint32_t *ix = ir_ + slot(k), *iy = ir_ + slot(2 * k), *x = r_ + slot(k + 2),
                                *y = r_ + slot(2 * k + 4);
            if (j + 8 == nv) return ntt::product_kernels::bottom_last(a + j, leaves[cur], ix, iy);
            if (kForwardB) bottom::both_keep(a + j + 8, b + j + 8, leaves[next], x, y, a + j, leaves[cur], ix, iy);
            else bottom::both_reuse(a + j + 8, b + j + 8, leaves[next], x, y, a + j, leaves[cur], ix, iy);
        }
    }

    const std::uint32_t *r_, *ir_;
};

// Interpolation on n >= 3 points (so a, r != 0 and r^k != 1 for 0 < k < n). L = 2^lg >= 2n with
// lg even and >= 10 (L / 8 = 2 * 4^j vectors, subtrees of at least 16). Buffers a and b of L
// words. In a: y, Y's transform and the first product, with E in a[L/2, L); then D in a[0, L/2),
// its transform and the second product, with G in a[L/2, L); then c in a[0, n). In b: K's
// transform. One mapping in huge pages; single use.
class Interpolation {
public:
    explicit Interpolation(std::size_t n) : n_(n), lanes_(n) {
        lg_ = std::max(10, int(std::bit_width(2 * n - 1)));
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

    ~Interpolation() { ::munmap(region_, bytes_); }

    Interpolation(const Interpolation&) = delete;
    Interpolation& operator=(const Interpolation&) = delete;

    std::uint32_t* values() { return a_; }  // room for y: n values < P
    char* text() { return text_; }           // fields::kTextBytes bytes, 16-byte aligned

    // The n coefficients, canonical.
    const std::uint32_t* interpolate(std::uint32_t a, std::uint32_t r) {
        using namespace ntt::detail;
        const std::size_t len = length(), q = len / 64;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        const std::uint32_t r_inverse = inverse(r);
        totals(r);
        weights(r, r_inverse);
        kernel(r, r_inverse);
        auto* av = reinterpret_cast<Vec*>(a_);
        auto* bv = reinterpret_cast<Vec*>(b_);
        ntt::detail::forward_radix8(av, q, roots_);
        ntt::detail::forward_radix8(bv, q, roots_);
        product<true>(av, bv, q);
        differences(r, r_inverse);
        ntt::detail::forward_radix8(av, q, roots_);
        product<false>(av, bv, q);
        coefficients(a, r, r_inverse);
        return a_;
    }

private:
    static constexpr std::size_t kPadding = 16;  // words after each factor: the kernels read 4 bytes past

    std::size_t length() const { return std::size_t(1) << lg_; }

    // u_k = 1 - r^k for 0 < k < n, 1 otherwise; Q_k = u_0 u_1 ... u_k.
    std::uint32_t factor(std::uint32_t r, std::size_t k) const {
        return k == 0 || k >= n_ ? 1 : (1 + kP - power_of(r, k)) % kP;
    }

    // The product of the leaves carries 2^-32 and the transform a factor L / 8. The last scan of
    // each product undoes both by its factor (L / 8)^-1 2^64, for montgomery() with the raw output.
    std::uint32_t output_scale() const {
        return multiply_mod(mont(inverse(std::uint32_t(length() / 8))), kR);
    }

    // Q at the lanes' ends and its inverse: q_end_[s] = Q_min(end(s), n - 1). Scan: x_k = Q_k
    // 2^32 / Q_(start(s) - 1), x_(k+1) = x_k (u_(k+1) 2^32) / 2^32 with g_k = -r^(k+1) 2^32.
    void totals(std::uint32_t r) {
        const Lanes& lanes = lanes_;
        Vec x[4], g[4];
        Lanes::fill(x, [&](int s) { return mont(factor(r, lanes.start(s))); });
        Lanes::fill(g, [&](int s) { return negate(mont(power_of(r, lanes.start(s) + 1))); });
        const Vec offset = broadcast(kR);
        const Factor ratio(r);
        const std::size_t last = lanes.first_padding() - 1;  // step at position n - 1 of lane partial()
        std::uint32_t partial = lanes.first_padding() == 0 ? kR : 0;
        if (last == 0) partial = lanes.lane(x);
        for (std::size_t j = 1; j < lanes.chunk(); ++j) {
#pragma GCC unroll 4
            for (int v = 0; v < 4; ++v) {
                x[v] = montgomery(x[v], low(add(g[v], offset)));
                g[v] = times(g[v], ratio);
            }
            if (j == last) partial = lanes.lane(x);
        }
        alignas(32) std::uint32_t lane_x[Lanes::kCount], total[Lanes::kCount];
        for (int v = 0; v < 4; ++v) store(lane_x + 8 * v, x[v]);
        const std::uint32_t unit = inverse(kR);
        std::uint32_t q = 1;
        for (int s = 0; s < Lanes::kCount; ++s) {
            const std::uint32_t x_s = s < lanes.partial() ? lane_x[s] : s == lanes.partial() ? partial : kR;
            total[s] = multiply_mod(x_s % kP, unit);
            q_end_[s] = q = multiply_mod(q, total[s]);
        }
        q = inverse(q);
        for (int s = Lanes::kCount; s-- > 0;) {
            q_end_inverse_[s] = q;
            q = multiply_mod(q, total[s]);
        }
    }

    // Y_k = (-1)^k y_k / Q_k in a, k < n, zero up to L / 2. Backward scan: x_k = 2^32 / Q_k,
    // x_(k-1) = x_k u_k with g_k = -r^k 2^32.
    void weights(std::uint32_t r, std::uint32_t r_inverse) {
        const Lanes& lanes = lanes_;
        const auto initial = [&](std::size_t k, int s) {
            return State{mont(q_end_inverse_[s]), negate(mont(power_of(r, k)))};
        };
        const Vec two_p = broadcast(2 * kP);
        std::uint32_t* const y = a_;
        lanes.scan<false>([&](int s) { return initial(lanes.end(s), s); }, initial(n_ - 1, Lanes::kCount - 1), kR,
                          r_inverse, [&](std::size_t k, Vec x) {
                              const Vec sign = _mm256_blend_epi32(x, _mm256_sub_epi32(two_p, x), 0xAA);  // k even
                              store(y + k, reduce(montgomery(load(y + k), sign), kP));
                          });
        clear(a_);
    }

    // K_k = r^t(k) / Q_k in b, k < n, zero up to L / 2. Backward scan: x_(k-1) = x_k (r^(1-k) - r)
    // with g_k = r^(1-k) 2^32.
    void kernel(std::uint32_t r, std::uint32_t r_inverse) {
        const Lanes& lanes = lanes_;
        const auto initial = [&](std::size_t k, int s) {
            return State{multiply_mod(power_of(r, triangle(k)), q_end_inverse_[s]), mont(power_of(r_inverse, k - 1))};
        };
        std::uint32_t* const b = b_;
        lanes.scan<false>([&](int s) { return initial(lanes.end(s), s); }, initial(n_ - 1, Lanes::kCount - 1),
                          kP - mont(r), r, [b](std::size_t k, Vec x) { store(b + k, reduce(x, kP)); });
        clear(b_);
    }

    // Zeros f[n, L / 2) after a scan wrote padding terms.
    void clear(std::uint32_t* f) const {
        const std::size_t end = std::min(length() / 2, lanes_.chunk() * Lanes::kCount);
        if (n_ < end) std::memset(f + n_, 0, (end - n_) * sizeof(std::uint32_t));
    }

    // D_(n-1-k) = (-1)^k Q_k r^-t(k) E_k in a, k < n, zero up to L/2, from E in a[L/2, L). Forward
    // scan: x_(k+1) = x_k (r - r^-k) with g_k = -r^-k 2^32, x_k = (-1)^k Q_k r^-t(k) (L / 8)^-1 2^64.
    void differences(std::uint32_t r, std::uint32_t r_inverse) {
        const Lanes& lanes = lanes_;
        const std::uint32_t scale = output_scale();
        const auto initial = [&](int s) {
            const std::size_t k = lanes.start(s);  // even
            const std::uint32_t q = s == 0 ? 1 : multiply_mod(q_end_[s - 1], factor(r, k));
            return State{multiply_mod(multiply_mod(q, power_of(r_inverse, triangle(k))), scale),
                         negate(mont(power_of(r_inverse, k)))};
        };
        const std::uint32_t* const u = a_ + length() / 2;
        std::uint32_t* const d = a_;
        const std::size_t n = n_;
        lanes.scan<true>(initial, {}, mont(r), r_inverse, [=](std::size_t k, Vec x) {
            if (k >= n) return;
            const Vec e = reduce(montgomery(load(u + k), x), kP);
            if (k + 8 <= n) return store_unaligned(d + n - 8 - k, reverse(e));
            alignas(32) std::uint32_t lanes_e[8];
            store(lanes_e, e);
            for (std::size_t i = 0; k + i < n; ++i) d[n - 1 - k - i] = lanes_e[i];
        });
        std::memset(a_ + n_, 0, (length() / 2 - n_) * sizeof(std::uint32_t));
    }

    // c_l = (-a)^-l / Q_l G_(n-1-l) in a, l < n, from G in a[L/2, L). Backward scan:
    // x_(l-1) = x_l (a r^l - a) with g_l = a r^l 2^32, x_l = (-a)^-l / Q_l (L / 8)^-1 2^64.
    void coefficients(std::uint32_t a, std::uint32_t r, std::uint32_t r_inverse) {
        const Lanes& lanes = lanes_;
        const std::uint32_t scale = output_scale(), a_inverse = inverse(a);
        const auto initial = [&](std::size_t l, int s) {
            std::uint32_t x = multiply_mod(multiply_mod(power_of(a_inverse, l), q_end_inverse_[s]), scale);
            if (l % 2) x = negate(x);
            return State{x, mont(multiply_mod(a, power_of(r, l)))};
        };
        const std::uint32_t* const u = a_ + length() / 2;
        std::uint32_t* const c = a_;
        const std::ptrdiff_t n = std::ptrdiff_t(n_);
        lanes.scan<false>([&](int s) { return initial(lanes.end(s), s); }, initial(n_ - 1, Lanes::kCount - 1),
                          kP - mont(a), r_inverse, [=](std::size_t l, Vec x) {
                              if (std::ptrdiff_t(l) >= n) return;
                              const std::ptrdiff_t at = n - 8 - std::ptrdiff_t(l);  // >= -7: in a
                              const Vec g = reverse(load_unaligned(u + at));
                              store(c + l, reduce(montgomery(g, x), kP));
                          });
    }

    // The lower half of a b in a[L/2, L), < 2P, before the scale (L / 8)^-1 2^32; ntt::Product's
    // layout.
    template <bool kForwardB>
    void product(Vec* a, Vec* b, std::size_t q) const {
        const Subtrees<kForwardB> subtrees(roots_, inverse_roots_);
        for (std::size_t c = 0; c < 8; ++c) subtrees.visit(a + c * q, b + c * q, q, c);
        top_lower(a, q);
    }

    // The top inverse levels for the lower half only: blocks a[t q, (t + 1) q) hold the identity
    // group (t < 4) and group 1 (t >= 4), < 2P; block 4 + m gets quarter m of the lower half, < 2P:
    // kernels.hpp's inverse_identity and inverse, then the sum of the last radix-2 level.
    void top_lower(Vec* a, std::size_t q) const {
        const std::uint32_t* const ir = inverse_roots_;
        const Factor x(ir[1], ir[9]), y(ir[2], ir[10]), z(ir[3], ir[11]);  // identity group: z = x
        for (Vec* f = a; f != a + q; ++f) {
            const Vec ab = low(add(f[0], f[q])), cd = low(add(f[2 * q], f[3 * q]));
            const Vec amb = low_difference(f[0], f[q]), cmd = multiply(diff(f[2 * q], f[3 * q]), x);
            const Vec gab = low(add(f[4 * q], f[5 * q])), gcd = low(add(f[6 * q], f[7 * q]));
            const Vec gamb = multiply(diff(f[4 * q], f[5 * q]), y), gcmd = multiply(diff(f[6 * q], f[7 * q]), z);
            f[4 * q] = low(add(low(add(ab, cd)), low(add(gab, gcd))));
            f[5 * q] = low(add(low(add(amb, cmd)), low(add(gamb, gcmd))));
            f[6 * q] = low(add(low_difference(ab, cd), multiply(diff(gab, gcd), x)));
            f[7 * q] = low(add(low_difference(amb, cmd), multiply(diff(gamb, gcmd), x)));
        }
    }

    std::size_t n_;
    Lanes lanes_;
    int lg_;
    void* region_;
    std::size_t bytes_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
    char* text_;
    std::array<std::uint32_t, Lanes::kCount> q_end_, q_end_inverse_;
};

// Lagrange interpolation in O(n^2) for small n: f = sum_i y_i / M'(x_i) M(x) / (x - x_i).
std::vector<std::uint32_t> interpolate_small(std::uint32_t a, std::uint32_t r, const std::vector<std::uint32_t>& y) {
    const std::size_t n = y.size();
    std::vector<std::uint32_t> x(n), m(n + 1, 0), c(n, 0), q(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = i ? multiply_mod(x[i - 1], r) : a;
    m[0] = 1;  // M = prod (x - x_i), low degree first
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = i + 2; k-- > 0;) m[k] = ((k ? m[k - 1] : 0) + kP - multiply_mod(m[k], x[i])) % kP;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t derivative = 1;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) derivative = multiply_mod(derivative, (x[i] + kP - x[j]) % kP);
        const std::uint32_t weight = multiply_mod(y[i], inverse(derivative));
        std::uint32_t carry = 0;  // q = M / (x - x_i), from the top
        for (std::size_t k = n; k-- > 0;) q[k] = carry = (m[k + 1] + multiply_mod(carry, x[i])) % kP;
        for (std::size_t k = 0; k < n; ++k) c[k] = (c[k] + multiply_mod(weight, q[k])) % kP;
    }
    return c;
}

constexpr std::size_t kSmall = 32;  // largest n for interpolate_small

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const std::uint32_t a = in.read<std::uint32_t>(), r = in.read<std::uint32_t>();
    io::Writer out;
    if (n == 0) return out.write('\n');
    if (n <= kSmall) {
        std::vector<std::uint32_t> y(n);
        for (auto& v : y) v = in.read<std::uint32_t>();
        const std::vector<std::uint32_t> c = interpolate_small(a, r, y);
        alignas(64) static char text[fields::kTextBytes];
        return fields::write(out, c.data(), n, text);
    }
    Interpolation interpolation(n);
    io::read_bulk(in, interpolation.values(), n);
    fields::write(out, interpolation.interpolate(a, r), n, interpolation.text());
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here). _exit skips their teardown too.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
