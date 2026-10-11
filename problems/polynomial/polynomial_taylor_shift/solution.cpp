// f(x + c) mod 998244353 for f of N <= 2^19 coefficients a_i. With A_i = a_i i! and
// E_j = c^j / j!,
//   b_k = (1 / k!) sum_(i >= k) A_i E_(i-k).
// In one cyclic convolution of length L = 2^lg >= 2N, A sits at [0, N) and E reversed, E_j at
// L/2 - j, so b_k k! is coefficient L/2 + k: the upper half in natural order. The transform is
// ntt::Product's layout (lib/ntt/product.hpp); its top inverse levels run for the upper half only.
// Factorials and c^j / j! come from product chains in 8 lanes whose multipliers are arithmetic
// progressions, started from lib/poly/factorials.hpp. Output in fixed-width fields
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
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::multiply;
using ntt::detail::multiply_mod;
using ntt::detail::power;
using ntt::detail::reduce;
using ntt::detail::Subtrees;
using ntt::detail::Vec;
using poly::detail::load;
using poly::detail::load_unaligned;
using poly::detail::low;
using poly::detail::low_difference;
using poly::detail::montgomery;
using poly::detail::odd_lanes;
using poly::detail::store;
using poly::detail::transpose_steps;

std::uint32_t inverse(std::uint32_t x) { return power(x, kP - 2); }

std::uint32_t mont(std::uint32_t x) { return multiply_mod(x, kR); }  // x 2^32 mod P

constexpr int kLanes = 8;

// One value per lane of a scan.
using Lanes = std::array<std::uint32_t, kLanes>;

// 8 steps of the lanes: block[t] holds step t of lane l in lane l.
using Block = Vec[8];

// A product chain in 8 lanes: at its step j, lane s multiplies x by m = base_s + j step mod P,
// x <- x m / 2^32 in [0, 2P), after recording x. x starts below 2P; base and step below P.
class Chain {
public:
    Chain(const Lanes& x, const Lanes& base, std::uint32_t step)
        : x_(load_unaligned(x.data())), m_(load_unaligned(base.data())), step_(broadcast(step)) {}

    // Records the terms as step t of block, then advances.
    [[gnu::always_inline]] void advance(Block& block, int t) {
        block[t] = x_;
        x_ = montgomery(x_, odd_lanes(x_), m_, odd_lanes(m_));
        m_ = reduce(add(m_, step_), kP);
    }

private:
    Vec x_, m_, step_;
};

// A chain as Chain with two steps per product on its critical path: from x_j, x_(j+1) = x_j m_j and
// x_(j+2) = x_j M_j with M_j = m_j m_(j+1) / 2^32. M_(j+2) = M_j + D_j, D_j = (4 step m_j +
// 6 step^2) / 2^32 and D_(j+2) = D_j + 8 step^2 / 2^32, all mod P in [0, P).
class PairChain {
public:
    PairChain(const Lanes& x, const Lanes& base, std::uint32_t step) {
        const std::uint32_t unit = inverse(kR);  // 2^-32
        const std::uint32_t square = multiply_mod(multiply_mod(step, step), unit);
        Lanes pair, difference;
        for (int s = 0; s < kLanes; ++s) {
            pair[s] = multiply_mod(multiply_mod(base[s], (base[s] + step) % kP), unit);
            difference[s] = (multiply_mod(multiply_mod(std::uint32_t(4 * std::uint64_t(step) % kP), base[s]), unit) + 6 * std::uint64_t(square)) % kP;
        }
        x_ = load_unaligned(x.data()), m_ = load_unaligned(base.data());
        pair_ = load_unaligned(pair.data()), difference_ = load_unaligned(difference.data());
        step_ = broadcast(std::uint32_t(2 * std::uint64_t(step) % kP));
        increment_ = broadcast(std::uint32_t(8 * std::uint64_t(square) % kP));
    }

    // Records steps t and t + 1 as steps 7 - t and 6 - t of block, then advances two steps.
    [[gnu::always_inline]] void advance_reversed(Block& block, int t) {
        const Vec x_odd = odd_lanes(x_);
        block[7 - t] = x_;
        block[6 - t] = montgomery(x_, x_odd, m_, odd_lanes(m_));
        x_ = montgomery(x_, x_odd, pair_, odd_lanes(pair_));
        m_ = reduce(add(m_, step_), kP);
        pair_ = reduce(add(pair_, difference_), kP);
        difference_ = reduce(add(difference_, increment_), kP);
    }

private:
    Vec x_, m_, pair_, difference_, step_, increment_;
};

// f(x + c) for n > 64 coefficients and c != 0. L = 2^lg >= 2n with lg even and >= 10 (L / 8 =
// 2 * 4^j vectors, subtrees of at least 16). Scans run over 8 lanes of C positions each, C a
// multiple of 64 with C / 64 odd, so the 8 streams of a pass fall into different cache sets;
// n <= 8 C <= n + 1023. Lanes run in lockstep, 8 steps per block; while the chains compute
// block j + 1, the lanes of block j are transposed into position order and stored.
// Buffers of L words: A, then the product, in a; E, then the output, in b. One mapping in huge
// pages, never freed; single use.
class TaylorShift {
public:
    explicit TaylorShift(std::size_t n) {
        lg_ = std::max(10, int(std::bit_width(2 * n - 1)));
        lg_ += lg_ % 2;
        chunk_ = ((n + kLanes - 1) / kLanes + 63) / 64 * 64;
        if (chunk_ / 64 % 2 == 0) chunk_ += 64;
        const std::size_t len = length(), tables = ntt::detail::table_words(lg_);
        mem::Arena arena((2 * len + 3 * kPadding + 2 * tables) * sizeof(std::uint32_t) + fields::kTextBytes);
        arena.take<std::uint32_t>(kPadding);
        b_ = arena.take<std::uint32_t>(len + kPadding);
        a_ = arena.take<std::uint32_t>(len + kPadding);  // a different cache set from b at equal offsets
        roots_ = arena.take<std::uint32_t>(tables);
        inverse_roots_ = arena.take<std::uint32_t>(tables);
        text_ = arena.take<char>(fields::kTextBytes);
    }

    std::uint32_t* coefficients() { return a_; }  // room for f: n values < P
    char* text() { return text_; }                // fields::kTextBytes bytes, 16-byte aligned

    // The n coefficients of f(x + c), canonical.
    const std::uint32_t* shift(std::uint32_t c) {
        using namespace ntt::detail;
        const std::size_t len = length(), q = len / 64;
        build_table(roots_, len / 16, kRoots[0]);
        build_table(inverse_roots_, len / 16, kRoots[1]);
        lane_factorials();
        weights(c);
        auto* av = reinterpret_cast<Vec*>(a_);
        auto* bv = reinterpret_cast<Vec*>(b_);
        forward_radix8(av, q, roots_);
        forward_radix8(bv, q, roots_);
        // E_0 = 1 sits at L / 2, which forward_radix8 does not read. Modulo x^(L/2) - 1 and
        // x^(L/2) + 1 it is +1 and -1 at x^0, which enters the first word of every output block
        // of the half's radix-4 group with factor 1.
        for (std::size_t t = 0; t < 8; ++t) {
            std::uint32_t& x = b_[8 * q * t];
            x = (x % kP + (t < 4 ? 1 : kP - 1)) % kP;
        }
        const Subtrees subtrees(roots_, inverse_roots_);
        for (std::size_t t = 0; t < 8; ++t) subtrees.visit(av + t * q, bv + t * q, q, t);
        top_upper(av, q);
        output();
        return b_;
    }

private:
    // Words before b and after each buffer: E's padding terms reach 1023 words below b. 1040
    // words (a multiple of 64 bytes) also put a and b 64 bytes apart in cache-set terms.
    static constexpr std::size_t kPadding = 1040;

    std::size_t length() const { return std::size_t(1) << lg_; }
    std::size_t start(int s) const { return s * chunk_; }

    // factorial_[s] = start(s)!, inverse_[s] = 1 / start(s)!, s <= 8.
    void lane_factorials() {
        poly::Lanes points{};
        for (int s = 0; s <= kLanes; ++s) points[s] = std::uint32_t(start(s));
        const poly::Lanes f = poly::factorials(points);
        std::copy_n(f.begin(), kLanes + 1, factorial_.begin());
        inverse_ = factorial_;
        poly::invert(inverse_);
    }

    // A_i = a_i i! in place in a, i < 8 C (the input is zero from n on), and E_j = c^j / j! at
    // L/2 - j in b for 0 < j <= 8 C. Lane s of the input chain starts at i = start(s): x = i! 2^32,
    // multipliers (i + 1) 2^32. Lane s of the E chain starts at j = start(s + 1) and runs down:
    // y = E_j, multipliers j c^-1 2^32. Terms E_j with j >= n are harmless: they meet A only in
    // coefficients below L/2.
    void weights(std::uint32_t c) {
        const std::size_t chunk = chunk_;
        const std::uint32_t c_chunk = power(c, std::uint32_t(chunk)), c_step = mont(inverse(c));
        Lanes x, up, y, down;
        std::uint32_t c_power = 1;  // c^start(s + 1)
        for (int s = 0; s < kLanes; ++s) {
            c_power = multiply_mod(c_power, c_chunk);
            x[s] = mont(factorial_[s]), up[s] = mont(std::uint32_t(start(s) + 1));
            y[s] = multiply_mod(c_power, inverse_[s + 1]), down[s] = multiply_mod(std::uint32_t(start(s + 1)), c_step);
        }
        Chain input(x, up, kR), kernel(y, down, kP - c_step);
        Block forward[2], backward[2];
#pragma GCC unroll 8
        for (int t = 0; t < 8; ++t) input.advance(forward[0], t), kernel.advance(backward[0], t);
        for (std::size_t j = 0, cur = 0; j < chunk; j += 8, cur ^= 1) {
            std::uint32_t* const a = a_ + j;                        // lane s at a + s C
            std::uint32_t* const e = b_ + length() / 2 - chunk + j;  // lane s at e - s C
#pragma GCC unroll 8
            for (int t = 0; t < 8; ++t) {
                input.advance(forward[cur ^ 1], t);
                kernel.advance(backward[cur ^ 1], t);
                if (t == 0) transpose_steps(forward[cur]), transpose_steps(backward[cur]);
                const std::size_t at = t * chunk;  // lane t
                const Vec f = load(a + at), w = forward[cur][t];
                store(a + at, reduce(montgomery(f, odd_lanes(f), w, odd_lanes(w)), kP));
                store(e - at, reduce(backward[cur][t], kP));
            }
        }
    }

    // The top inverse levels for the upper half only: blocks a[t q, (t + 1) q) hold the identity
    // group (t < 4) and group 1 (t >= 4) of the halves u and w, < 2P; block m < 4 gets quarter m
    // of u - w, < 2P: kernels.hpp's inverse_identity and inverse, then the difference of the last
    // radix-2 level.
    void top_upper(Vec* a, std::size_t q) const {
        const std::uint32_t* const ir = inverse_roots_;
        const Factor x(ir[1], ir[9]), y(ir[2], ir[10]), z(ir[3], ir[11]);  // identity group: z = x
        for (Vec* f = a; f != a + q; ++f) {
            const Vec ab = low(add(f[0], f[q])), cd = low(add(f[2 * q], f[3 * q]));
            const Vec amb = low_difference(f[0], f[q]), cmd = multiply(diff(f[2 * q], f[3 * q]), x);
            const Vec gab = low(add(f[4 * q], f[5 * q])), gcd = low(add(f[6 * q], f[7 * q]));
            const Vec gamb = multiply(diff(f[4 * q], f[5 * q]), y), gcmd = multiply(diff(f[6 * q], f[7 * q]), z);
            f[0] = low_difference(low(add(ab, cd)), low(add(gab, gcd)));
            f[q] = low_difference(low(add(amb, cmd)), low(add(gamb, gcmd)));
            f[2 * q] = low(diff(low_difference(ab, cd), multiply(diff(gab, gcd), x)));
            f[3 * q] = low(diff(low_difference(amb, cmd), multiply(diff(gamb, gcmd), x)));
        }
    }

    // b_k = (A E)[L/2 + k] / k! into b, from u - w in a (top_upper): (A E)[L/2 + k] = (u - w)[k]
    // times the transform's scale. Lane s of the chain starts at k = start(s + 1) - 1 and runs
    // down: z = (L / 8)^-1 2^64 / k!, multipliers k 2^32. The leaf products carry 2^-32 and the
    // transform a factor L / 8; montgomery() with z undoes both. Outputs k >= n are padding.
    void output() {
        const std::size_t chunk = chunk_;
        const std::uint32_t scale = multiply_mod(mont(inverse(std::uint32_t(length() / 8))), kR);
        Lanes z, down;
        for (int s = 0; s < kLanes; ++s) {
            z[s] = multiply_mod(multiply_mod(inverse_[s + 1], std::uint32_t(start(s + 1))), scale);
            down[s] = mont(std::uint32_t(start(s + 1) - 1));
        }
        PairChain chain(z, down, kP - kR);
        Block block[2];
#pragma GCC unroll 4
        for (int t = 0; t < 8; t += 2) chain.advance_reversed(block[0], t);
        for (std::size_t j = 0, cur = 0; j < chunk; j += 8, cur ^= 1) {
            const std::uint32_t* const u = a_ + chunk - 8 - j;  // lane s at u + s C
            std::uint32_t* const out = b_ + chunk - 8 - j;
#pragma GCC unroll 8
            for (int t = 0; t < 8; ++t) {
                if (t % 2 == 0) chain.advance_reversed(block[cur ^ 1], t);
                if (t == 0) transpose_steps(block[cur]);
                const std::size_t at = t * chunk;  // lane t
                const Vec g = load(u + at), w = block[cur][t];
                store(out + at, reduce(montgomery(g, odd_lanes(g), w, odd_lanes(w)), kP));
            }
        }
    }

    std::size_t chunk_;
    int lg_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
    char* text_;
    std::array<std::uint32_t, kLanes + 1> factorial_, inverse_;
};

// f(x + c) by Horner's rule in O(n^2): g <- g (x + c) + a_i from the top.
std::vector<std::uint32_t> shift_small(const std::vector<std::uint32_t>& a, std::uint32_t c) {
    const std::size_t n = a.size();
    std::vector<std::uint32_t> g(n, 0);
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t k = n - 1; k > 0; --k) g[k] = (g[k - 1] + multiply_mod(g[k], c)) % kP;
        g[0] = (multiply_mod(g[0], c) + a[i]) % kP;
    }
    return g;
}

constexpr std::size_t kSmall = 64;  // largest n for shift_small

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const std::uint32_t c = in.read<std::uint32_t>();
    io::Writer out;
    if (n <= kSmall || c == 0) {
        std::vector<std::uint32_t> a(n);
        io::read_bulk(in, a.data(), n);
        const std::vector<std::uint32_t> b = c == 0 ? a : shift_small(a, c);
        alignas(64) static char text[fields::kTextBytes];
        return fields::write(out, b.data(), n, text);
    }
    TaylorShift taylor(n);
    io::read_bulk(in, taylor.coefficients(), n);
    fields::write(out, taylor.shift(c), n, taylor.text());
}

}  // namespace

RUN_EARLY(solve)
