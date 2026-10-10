// f(x + c) mod 998244353 for f of N <= 2^19 coefficients a_i. With A_i = a_i i! and
// E_j = c^j / j!,
//   b_k = (1 / k!) sum_(i >= k) A_i E_(i-k).
// In one cyclic convolution of length L = 2^lg >= 2N, A sits at [0, N) and E reversed, E_j at
// L/2 - j, so b_k k! is coefficient L/2 + k: the upper half in natural order. The transform is
// ntt::Product's layout (lib/ntt/product.hpp) with the last radix-2 level folded into the output
// pass. Factorials and c^j / j! come from product chains in 32 lanes whose multipliers are
// arithmetic progressions (lib/poly/factorials.hpp). Output in fixed-width fields
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

using ntt::detail::broadcast;
using ntt::detail::diff;
using ntt::detail::kP;
using ntt::detail::kR;
using ntt::detail::multiply_mod;
using ntt::detail::power;
using ntt::detail::reduce;
using ntt::detail::slot;
using ntt::detail::Subtrees;
using ntt::detail::Vec;
using poly::Lanes;
using poly::detail::Chain;
using poly::detail::load;
using poly::detail::low;
using poly::detail::montgomery;
using poly::detail::odd_lanes;
using poly::detail::scan;
using poly::detail::store;

std::uint32_t inverse(std::uint32_t x) { return power(x, kP - 2); }

std::uint32_t mont(std::uint32_t x) { return multiply_mod(x, kR); }  // x 2^32 mod P

constexpr int kLanes = 32;

// f(x + c) for n > 64 coefficients and c != 0. L = 2^lg >= 2n with lg even and >= 10 (L / 8 =
// 2 * 4^j vectors, subtrees of at least 16). Scans (poly::detail::scan) run over 32 lanes of
// C = poly::detail::scan_chunk(n) positions each, n <= 32 C <= n + 1023.
// Buffers of L words: A, then the product, in a; E, then the output, in b. One mapping in huge
// pages, never freed; single use.
class TaylorShift {
public:
    explicit TaylorShift(std::size_t n) : n_(n) {
        lg_ = std::max(10, int(std::bit_width(2 * n - 1)));
        lg_ += lg_ % 2;
        chunk_ = poly::detail::scan_chunk(n);
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
        for (std::size_t t = 0; t < 4; ++t) subtrees.visit(av + t * q, bv + t * q, q, t);
        ntt::kernels::inverse_identity(av, q, inverse_roots_);
        for (std::size_t t = 4; t < 8; ++t) subtrees.visit(av + t * q, bv + t * q, q, t);
        ntt::kernels::inverse(av + 4 * q, q, inverse_roots_ + slot(1), inverse_roots_ + slot(2));
        output();
        return b_;
    }

private:
    // Words before b and after each buffer: E's padding terms reach 1023 words below b, the
    // output pass reads up to 1023 words past a. 1040 words (a multiple of 64 bytes) also put a
    // and b 64 bytes apart in cache-set terms.
    static constexpr std::size_t kPadding = 1040;

    std::size_t length() const { return std::size_t(1) << lg_; }
    std::size_t start(int s) const { return s * chunk_; }

    // factorial_[s] = start(s)!, inverse_start_[s] = 1 / start(s)!, s <= kLanes.
    void lane_factorials() {
        Lanes ends;
        for (int s = 0; s < kLanes; ++s) ends[s] = std::uint32_t(start(s + 1));
        const Lanes f = poly::factorials(ends);
        factorial_[0] = 1;
        std::copy(f.begin(), f.end(), factorial_.begin() + 1);
        inverse_start_ = factorial_;
        poly::invert(inverse_start_);
    }

    // Lane values from s -> value(s).
    template <class F>
    static Lanes lanes(F value) {
        Lanes x;
        for (int s = 0; s < kLanes; ++s) x[s] = value(s);
        return x;
    }

    // A_i = a_i i! in place in a, i < 32 C (the input is zero from n on), and E_j = c^j / j! at
    // L/2 - j in b for 0 < j <= 32 C. Lane s of the input chain starts at i = start(s): x = i! 2^32,
    // multipliers (i + 1) 2^32. Lane s of the E chain starts at j = start(s + 1) and runs down:
    // y = E_j, multipliers j c^-1 2^32. Terms E_j with j >= n are harmless: they meet A only in
    // coefficients below L/2.
    void weights(std::uint32_t c) {
        const std::uint32_t c_chunk = power(c, std::uint32_t(chunk_)), c_step = mont(inverse(c));
        std::array<std::uint32_t, kLanes> c_power;  // c^start(s + 1)
        for (int s = 0; s < kLanes; ++s) c_power[s] = s ? multiply_mod(c_power[s - 1], c_chunk) : c_chunk;
        Chain<> input(lanes([this](int s) { return mont(factorial_[s]); }),
                      lanes([this](int s) { return mont(std::uint32_t(start(s) + 1)); }), kR);
        Chain<> kernel(lanes([&](int s) { return multiply_mod(c_power[s], inverse_start_[s + 1]); }),
                       lanes([&](int s) { return multiply_mod(std::uint32_t(start(s + 1)), c_step); }), kP - c_step);
        // Captures by value: the vector stores may alias anything, so values reached through
        // memory (members, captured references) would be reloaded after each of them.
        const std::size_t stride = chunk_;
        std::uint32_t* const a = a_;
        std::uint32_t* const e = b_ + length() / 2 - stride;  // lane s at e - s C: E_j at L/2 - j
        scan(stride, [=](std::size_t j, int s, Vec x, Vec y) {
            const std::size_t at = s * stride;
            const Vec f = load(a + j + at);
            store(a + j + at, reduce(montgomery(f, odd_lanes(f), x, odd_lanes(x)), kP));
            store(e + j - at, reduce(y, kP));
        }, input, kernel);
    }

    // b_k = (A E)[L/2 + k] / k! into b, from the halves u, w of a after their top inverse groups:
    // (A E)[L/2 + k] = (u - w)[k] times the transform's scale. Lane s of the chain starts at
    // k = start(s + 1) - 1 and runs down: z = (L / 8)^-1 2^64 / k!, multipliers k 2^32. The leaf
    // products carry 2^-32 and the transform a factor L / 8; montgomery() with z undoes both.
    // Outputs k >= n are padding.
    void output() {
        const std::uint32_t scale = multiply_mod(mont(inverse(std::uint32_t(length() / 8))), kR);
        Chain<true> chain(lanes([&](int s) { return multiply_mod(multiply_mod(inverse_start_[s + 1], std::uint32_t(start(s + 1))), scale); }),
                          lanes([this](int s) { return mont(std::uint32_t(start(s + 1) - 1)); }), kP - kR);
        const std::size_t stride = chunk_, half = length() / 2;
        const std::uint32_t* const u = a_;
        std::uint32_t* const out = b_;
        scan(stride, [=](std::size_t j, int s, Vec z) {  // by value, as in weights()
            const std::size_t at = s * stride + stride - 8 - j;  // lane s from the top down
            const Vec g = low(diff(load(u + at), load(u + at + half)));
            store(out + at, reduce(montgomery(g, odd_lanes(g), z, odd_lanes(z)), kP));
        }, chain);
    }

    std::size_t n_, chunk_;
    int lg_;
    std::uint32_t *a_, *b_, *roots_, *inverse_roots_;
    char* text_;
    std::array<std::uint32_t, kLanes + 1> factorial_, inverse_start_;
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
