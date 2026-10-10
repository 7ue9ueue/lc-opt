// f(c + k) mod 998244353 for k < M, from the samples f(0), ..., f(N - 1) of f, deg f < N
// (N, M <= 2^19). For c + k outside {0, ..., N - 1} mod P, Lagrange's formula gives
//   f(c + k) = (-1)^k P_k sum_i g_i h_(k + N - 1 - i),
//   g_i = f(i) / (i! (N - 1 - i)!),  h_t = (-1)^t / (d + t),  d = c - N + 1,
//   P_k = (d + k) (d + k + 1) ... (d + k + N - 1):
// a middle product of g (N terms) and h (T = N + M - 1 terms), one cyclic convolution of length
// L = 2^lg >= 2 max(N, M) with g at [0, N) and h_t at L/2 - N + 1 + t, so the sums sit at L/2 + k:
// the upper half in natural order. Outputs at sample points (c + k = i mod P, i < N) are f(i);
// the others form at most two ranges whose d + t are nonzero, each one product.
// The transform is ntt::Product's layout (lib/ntt/product.hpp) as in
// multipoint_evaluation_on_geometric_sequence. Factorials, the inverses 1 / (d + t) and P_k come
// from product chains in 32 lanes (lib/poly/factorials.hpp). Output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp).
#include <algorithm>
#include <array>
#include <bit>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "lib/ntt/product.hpp"
#include "lib/poly/factorials.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

using ntt::detail::add;
using ntt::detail::broadcast;
using ntt::detail::diff;
using ntt::detail::diff_canonical;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::multiply_mod;
using ntt::detail::power;
using ntt::detail::reduce;
using ntt::detail::slot;
using ntt::detail::Vec;
using poly::Lanes;
using poly::detail::Chain;
using poly::detail::load;
using poly::detail::load_unaligned;
using poly::detail::low;
using poly::detail::montgomery;
using poly::detail::odd_lanes;
using poly::detail::scan;
using poly::detail::store_unaligned;

std::uint32_t inverse(std::uint32_t x) { return power(x, kP - 2); }
std::uint32_t mont(std::uint32_t x) { return multiply_mod(x, kR); }  // x 2^32 mod P
std::uint32_t negate(std::uint32_t x) { return x ? kP - x : 0; }

// x mod P for any integer x.
std::uint32_t residue(std::int64_t x) { return std::uint32_t((x % std::int64_t(kP) + kP) % kP); }

// a b / 2^32 mod P in [0, 2P) for a b < 2^32 P.
[[gnu::always_inline]] inline Vec times(Vec a, Vec b) { return montgomery(a, odd_lanes(a), b, odd_lanes(b)); }

Vec reverse(Vec x) { return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

// Stores the lanes l of x with 0 <= k + l < end at p[k + l]. Inlined: a call would spill the
// chains' registers in the scans.
[[gnu::always_inline]] inline void store_clipped(std::uint32_t* p, std::int64_t k, std::int64_t end, Vec x) {
    if (k >= 0 && k + 8 <= end) return store_unaligned(p + k, x);
    if (k + 8 <= 0 || k >= end) return;
    const Vec index = add(broadcast(std::uint32_t(k)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    const Vec inside = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), index),
                                           _mm256_cmpgt_epi32(broadcast(std::uint32_t(end)), index));
    _mm256_maskstore_epi32(reinterpret_cast<int*>(p + k), inside, x);
}

// ntt::detail::forward_radix8 for a factor that fills f[0, 8q): the groups act on the halves
// u = f mod (x^(n/2) - 1) and v = f mod (x^(n/2) + 1), u_t = f_t + f_(t+4) and
// v_t = f_t - f_(t+4) for the blocks f_t = f[t q, (t + 1) q). Inputs canonical; outputs < 4P.
void forward_radix8_full(Vec* f, std::size_t q, const std::uint32_t* roots) {
    using ntt::detail::multiply;
    const Factor i(roots[1], roots[9]), y(roots[2], roots[10]), z(roots[3], roots[11]);
    for (std::size_t j = 0; j < q; ++j) {
        Vec u[4], v[4];  // < 2P
#pragma GCC unroll 4
        for (std::size_t t = 0; t < 4; ++t) {
            const Vec lo = f[j + t * q], hi = f[j + (t + 4) * q];
            u[t] = add(lo, hi), v[t] = diff_canonical(lo, hi);
        }
        const Vec g0 = low(add(u[0], u[2])), g1 = low(add(u[1], u[3]));
        const Vec h0 = low(diff(u[0], u[2])), ih1 = multiply(diff(u[1], u[3]), i);
        f[j] = add(g0, g1), f[j + q] = diff(g0, g1);
        f[j + 2 * q] = add(h0, ih1), f[j + 3 * q] = diff(h0, ih1);
        const Vec iv2 = multiply(v[2], i), iv3 = multiply(v[3], i);
        const Vec u0 = low(add(v[0], iv2)), v0 = low(diff(v[0], iv2));
        const Vec yu1 = multiply(add(v[1], iv3), y), zv1 = multiply(diff(v[1], iv3), z);
        f[j + 4 * q] = add(u0, yu1), f[j + 5 * q] = diff(u0, yu1);
        f[j + 6 * q] = add(v0, zv1), f[j + 7 * q] = diff(v0, zv1);
    }
}

// Ranges of outputs f(c + k) without sample points, from n >= 2 samples: d + t != 0 mod P for
// t < T = n + m - 1, m the range's length. Scans over [0, n) and over the positions t of h run in
// 32 lanes of C positions (lib/poly/factorials.hpp); lane s of h covers [o + s C, o + (s + 1) C),
// with the padding 32 C - T (< 1024) at the end (o = 0) or, if d + t = 0 there, at the front
// (o = T - 32 C). Buffers: a (f, then g, then the product), b (h), y (P_k with the output factors,
// then the outputs; a range writes only its own outputs). One mapping in huge pages, never freed.
class SampleShift {
public:
    // Ranges of at most m outputs, total outputs in all.
    SampleShift(std::size_t n, std::size_t m, std::size_t total) : n_(n), capacity_(std::size_t(1) << log_length(m)) {
        const std::size_t len = capacity_, tables = ntt::detail::table_words(log_length(m));
        mem::Arena arena((2 * len + total + 6 * kPadding + 2 * tables) * sizeof(std::uint32_t) + fields::kTextBytes + 1024);
        arena.take<std::uint32_t>(kPadding);
        b_ = arena.take<std::uint32_t>(len + kPadding);
        a_ = arena.take<std::uint32_t>(len + kPadding);  // a different cache set from b at equal offsets
        arena.take<std::uint32_t>(kPadding);
        outputs_ = arena.take<std::uint32_t>(total + kPadding);
        roots_ = arena.take<std::uint32_t>(tables);
        inverse_roots_ = arena.take<std::uint32_t>(tables);
        text_ = arena.take<char>(fields::kTextBytes);
    }

    // f(0), ..., f(n - 1), canonical, zero after them; shift() overwrites them.
    std::uint32_t* samples() { return a_; }
    std::uint32_t* outputs() { return outputs_; }  // canonical
    char* text() { return text_; }                 // fields::kTextBytes bytes, 64-byte aligned

    // outputs[first + k] = f(c + k) for k < m.
    void shift(std::uint32_t c, std::size_t first, std::size_t m) {
        using namespace ntt::detail;
        m_ = m, terms_ = n_ + m - 1, lg_ = log_length(m), y_ = outputs_ + first;
        chunk_ = poly::detail::scan_chunk(terms_);
        const std::size_t len = length(), q = len / 64;
        d_ = residue(std::int64_t(c) - std::int64_t(n_) + 1);
        delta_ = len / 2 - n_ + 1;
        const std::uint64_t zero_at = kP - d_;  // d + t = 0 for t = zero_at
        origin_ = zero_at < 32 * chunk_ ? std::int64_t(terms_) - std::int64_t(32 * chunk_) : 0;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        weights();
        prefix_products();
        constants();
        inverses();
        auto* av = reinterpret_cast<Vec*>(a_);
        auto* bv = reinterpret_cast<Vec*>(b_);
        forward_radix8(av, q, roots_);
        forward_radix8_full(bv, q, roots_);
        const Subtrees subtrees(roots_, inverse_roots_);
        for (std::size_t t = 0; t < 4; ++t) subtrees.visit(av + t * q, bv + t * q, q, t);
        ntt::kernels::inverse_identity(av, q, inverse_roots_);
        for (std::size_t t = 4; t < 8; ++t) subtrees.visit(av + t * q, bv + t * q, q, t);
        ntt::kernels::inverse(av + 4 * q, q, inverse_roots_ + slot(1), inverse_roots_ + slot(2));
        output();
    }

    // Prepares another range: f in a again, the rest of a and b zero.
    void reset(const std::uint32_t* samples) {
        std::copy(samples, samples + n_, a_);
        std::fill(a_ + n_, a_ + capacity_ + kPadding, 0);
        std::fill(b_ - kPadding, b_ + capacity_ + kPadding, 0);
    }

private:
    // Words before and after each buffer: scans run up to 1023 positions past their ends (and
    // before b with o < 0). 1040 words also put the buffers in different cache sets.
    static constexpr std::size_t kPadding = 1040;

    // L = 2^lg >= 2 max(n, m), lg even and >= 10 (L / 8 = 2 * 4^j vectors, subtrees of at least 16).
    int log_length(std::size_t m) const {
        const int lg = std::max(10, int(std::bit_width(2 * std::max(n_, m) - 1)));
        return lg + lg % 2;
    }
    std::size_t length() const { return std::size_t(1) << lg_; }
    std::int64_t lane_start(int s) const { return origin_ + std::int64_t(s) * std::int64_t(chunk_); }
    std::uint32_t* h(std::int64_t t) const { return b_ + delta_ + t; }  // h_t; o - 8 <= t < o + 32 C

    // g_i = f(i) w_i in place in a, w_i = 1 / (i! (n - 1 - i)!) = w_(n-1-i). A reversed chain of
    // 1 / i! 2^32 in lanes of C' positions over [0, 32 C'), from 1 / (top - 1)! at each lane's top,
    // stores s_i = 1 / i! 2^32 at b[delta + i] (overwritten by h later). Then one product
    // s_i s_(n-1-i) gives w_i 2^32 for both g_i and g_(n-1-i), 8 pairs per step from both ends.
    void weights() {
        const std::size_t chunk = poly::detail::scan_chunk(n_);
        Lanes top, base;
        for (int s = 0; s < 32; ++s) top[s] = std::uint32_t((s + 1) * chunk - 1), base[s] = mont(top[s]);
        Lanes start = poly::factorials(top);
        poly::invert(start);
        for (auto& x : start) x = mont(x);
        Chain<true> chain(start, base, kP - kR);
        scan(chunk, [&](std::size_t j, int s, Vec x) { store_unaligned(h(std::int64_t((s + 1) * chunk - 8 - j)), x); }, chain);
        const std::int64_t n = n_;
        std::int64_t i = 0;
        for (; 2 * i + 16 <= n; i += 8) {
            const std::int64_t back = n - 8 - i;
            const Vec w = times(load_unaligned(h(i)), reverse(load_unaligned(h(back))));
            poly::detail::store(a_ + i, reduce(times(load(a_ + i), w), kP));
            store_unaligned(a_ + back, reduce(times(load_unaligned(a_ + back), reverse(w)), kP));
        }
        const std::uint32_t unscale = inverse(multiply_mod(kR, kR));  // 2^-64
        for (std::int64_t j = i; j < n - i; ++j)
            a_[j] = multiply_mod(a_[j], multiply_mod(multiply_mod(*h(j) % kP, *h(n - 1 - j) % kP), unscale));
    }

    // G_t = prod (-(d + u)) over u in [o + s C, t), t in lane s, into h(t), and G_(n + k) also
    // into y[k] for k < m (G_T = 1 when T = o + 32 C starts a lane 32). The lane totals tau_s are G
    // at the lanes' tops: tau_s = (-1)^C times the plain product, C even.
    void prefix_products() {
        Lanes one, base;
        for (int s = 0; s < 32; ++s) one[s] = 1, base[s] = mont(negate(residue(d_ + lane_start(s))));
        Chain<> chain(one, base, kP - kR);
        const std::int64_t n = n_, m = m_;
        scan(chunk_, [&](std::size_t j, int s, Vec x) {
            const std::int64_t t = lane_start(s) + std::int64_t(j);
            store_unaligned(h(t), x);
            store_clipped(y_, t - n, m, x);
        }, chain);
        if (origin_ + std::int64_t(32 * chunk_) == std::int64_t(terms_)) y_[m_ - 1] = 1;
        for (int s = 0; s < 32; ++s) {
            const std::int64_t last = lane_start(s) + std::int64_t(chunk_) - 1;
            tau_[s] = multiply_mod(*h(last) % kP, negate(residue(d_ + last)));
        }
        g0_ = *h(0) % kP;
        gn_ = y_[0] % kP;
    }

    // With F_t = prod (d + u) over u in [o, t) and A_s = F at lane s's start (the product of the
    // totals below s), h_t = (-1)^(o + t) / (d + t) times the scale below, and the factor of the
    // output k is z_k = (-1)^(k + o) P_k (L / 8)^-1 2^64 (one convolution sum times P_k (-1)^k,
    // the transform's L / 8 and the leaf products' 2^-32 undone by montgomery() in output()).
    void constants() {
        std::array<std::uint32_t, 33> totals;  // 1 / tau_s, then 1 / G_0
        std::copy(tau_.begin(), tau_.end(), totals.begin());
        totals[32] = g0_;
        poly::invert(totals);
        std::array<std::uint32_t, 33> prefix, inverse_prefix;  // A_s, 1 / A_s
        prefix[0] = inverse_prefix[0] = 1;
        for (int s = 0; s < 32; ++s) {
            inverse_tau_[s] = totals[s];
            prefix[s + 1] = multiply_mod(prefix[s], tau_[s]);
            inverse_prefix[s + 1] = multiply_mod(inverse_prefix[s], totals[s]);
        }
        const std::uint32_t scale = multiply_mod(mont(inverse(std::uint32_t(length() / 8))), kR);  // (L/8)^-1 2^64
        const std::uint32_t sign_n = n_ % 2 ? kP - 1 : 1;
        // z_k = R_(k-1) G_(n+k) kappa / 2^64 with R_(k-1) = 2^32 / G_k: G_k in lane s, G_(n+k) in
        // lane s' (s + j for j = (n + 1) / C, or one more from k = kstar_s on), kappa =
        // (-1)^n (A_s' / A_s) (L / 8)^-1 2^96.
        const std::size_t first = (n_ + 1) / chunk_;
        const auto kappa = [&](int s, std::size_t lane) {
            return multiply_mod(multiply_mod(multiply_mod(prefix[std::min<std::size_t>(lane, 32)], inverse_prefix[s]), sign_n),
                                multiply_mod(scale, kR));
        };
        for (int s = 0; s < 32; ++s) {
            kappa_low_[s] = broadcast(kappa(s, s + first));
            kappa_high_[s] = broadcast(kappa(s, s + first + 1));
            const std::int64_t kstar = lane_start(int(s + first + 1)) - std::int64_t(n_);
            kappa_from_[s] = broadcast(std::uint32_t(std::clamp<std::int64_t>(kstar - 1, -(1 << 30), 1 << 30)));
        }
        // z_0 = (-1)^o P_0 (L/8)^-1 2^64, P_0 = F_n / F_0 = A_s(n) G_n / (A_s(0) G_0) with
        // G_t = (-1)^(t - o) times the stored value.
        const auto lane_of = [this](std::int64_t t) { return std::size_t((t - origin_) / std::int64_t(chunk_)); };
        std::uint32_t z0 = multiply_mod(multiply_mod(prefix[lane_of(n_)], gn_), multiply_mod(inverse_prefix[lane_of(0)], totals[32]));
        if ((std::int64_t(n_) - origin_) % 2 != 0) z0 = negate(z0);  // the signs of G_n, G_0 and (-1)^o
        z0_ = multiply_mod(z0, scale);
    }

    // h_t = G_t R_t / 2^32 with R_t = 2^32 / (G_(t+1) (-1)^C): a reversed chain from 2^32 / tau_s at
    // each lane's top, multipliers d + t. Also z_k for 0 < k < m in place of y[k] = G_(n+k):
    // z_k = R_(k-1) y[k] kappa / 2^64. Then the padding of h is cleared and z_0 set.
    void inverses() {
        Lanes start, base;
        for (int s = 0; s < 32; ++s) {
            start[s] = mont(inverse_tau_[s]);
            base[s] = mont(residue(d_ + lane_start(s) + std::int64_t(chunk_) - 1));
        }
        Chain<true> chain(start, base, kP - kR);
        const std::int64_t m = m_;
        const Vec lanes = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        scan(chunk_, [&](std::size_t j, int s, Vec r) {
            const std::int64_t p = lane_start(s) + std::int64_t(chunk_) - 8 - std::int64_t(j);
            store_unaligned(h(p), reduce(times(load_unaligned(h(p)), r), kP));
            if (p + 8 > 0 && p + 1 < m) {
                const Vec k = add(broadcast(std::uint32_t(p)), lanes);
                const Vec high = _mm256_cmpgt_epi32(k, kappa_from_[s]);  // k >= kstar_s
                const Vec kappa = _mm256_blendv_epi8(kappa_low_[s], kappa_high_[s], high);
                store_clipped(y_, p + 1, m, times(times(r, load_unaligned(y_ + p + 1)), kappa));
            }
        }, chain);
        const std::size_t len = length();
        const std::int64_t front = std::max<std::int64_t>(0, std::int64_t(delta_) + origin_);
        std::fill(b_ + front, b_ + delta_, 0);
        const std::size_t back = std::min<std::size_t>(len, delta_ + std::max<std::size_t>(32 * chunk_, n_ + 1023));
        std::fill(b_ + delta_ + terms_, b_ + std::max(back, delta_ + terms_), 0);
        y_[0] = z0_;
    }

    // y_k = f(c + k) = (u - w)[k] z_k / 2^32 for the halves u, w of a after their top inverse
    // groups: (A H)[L/2 + k] = (u - w)[k] times the transform's scale.
    void output() {
        const std::size_t half = length() / 2;
        const std::int64_t m = m_;
        for (std::size_t k = 0; k < m_; k += 8) {
            const Vec g = low(diff(load(a_ + k), load(a_ + half + k)));
            store_clipped(y_, std::int64_t(k), m, reduce(times(g, load_unaligned(y_ + k)), kP));
        }
    }

    std::size_t n_, capacity_, m_ = 0, terms_ = 0, chunk_ = 0, delta_ = 0;
    int lg_ = 0;
    std::int64_t origin_ = 0;
    std::uint32_t d_ = 0, g0_ = 0, gn_ = 0, z0_ = 0;
    std::uint32_t *a_, *b_, *outputs_, *y_ = nullptr, *roots_, *inverse_roots_;
    char* text_;
    std::array<std::uint32_t, 32> tau_, inverse_tau_;
    Vec kappa_low_[32], kappa_high_[32], kappa_from_[32];  // kappa_from_: kstar_s - 1
};

// Outputs [first, first + count): the samples from source on (direct), or shifted from c = source.
struct Range {
    std::size_t first, count;
    bool direct;
    std::uint32_t source;
};

// The ranges of f(c + k), k < m, for n >= 2: c + k is the sample c + k for k < n - c (c < n), or
// the sample k - (P - c) for 0 <= k - (P - c) < n. At most two shifted ranges, which start at c or
// at n.
std::vector<Range> ranges(std::size_t n, std::size_t m, std::uint32_t c) {
    std::vector<Range> list;
    const auto add_range = [&](std::size_t first, std::size_t count, bool direct, std::uint32_t source) {
        if (count) list.push_back({first, count, direct, source});
    };
    if (c < n) {
        const std::size_t direct = std::min(m, n - c);
        add_range(0, direct, true, c);
        add_range(direct, m - direct, false, std::uint32_t(n));
        return list;
    }
    const std::size_t wrap = std::min<std::size_t>(m, kP - c), direct = std::min(m - wrap, n);
    add_range(0, wrap, false, c);
    add_range(wrap, direct, true, 0);
    add_range(wrap + direct, m - wrap - direct, false, std::uint32_t(n));
    return list;
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    const std::uint32_t c = in.read<std::uint32_t>();
    const std::vector<Range> list = n == 1 ? std::vector<Range>{} : ranges(n, m, c);
    std::size_t largest = 0, shifted = 0;
    for (const Range& r : list)
        if (!r.direct) largest = std::max(largest, r.count), ++shifted;
    SampleShift shift(n, largest, m);
    std::uint32_t* const f = shift.samples();
    std::uint32_t* const y = shift.outputs();
    io::read_bulk(in, f, n);
    if (n == 1) std::fill(y, y + m, f[0]);
    for (const Range& r : list)
        if (r.direct) std::copy(f + r.source, f + r.source + r.count, y + r.first);
    std::vector<std::uint32_t> samples;
    if (shifted > 1) samples.assign(f, f + n);
    bool fresh = true;
    for (const Range& r : list) {
        if (r.direct) continue;
        if (!fresh) shift.reset(samples.data());
        shift.shift(r.source, r.first, r.count);
        fresh = false;
    }
    io::Writer out;
    fields::write(out, y, m, shift.text());
}

}  // namespace

RUN_EARLY(solve)
