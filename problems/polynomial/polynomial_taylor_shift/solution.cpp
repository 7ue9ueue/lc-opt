// f(x + c) mod 998244353 for f of N <= 2^19 coefficients a_i. With A_i = a_i i! and
// E_j = c^j / j!,
//   b_k = (1 / k!) sum_(i >= k) A_i E_(i-k).
// In one cyclic convolution of length L = 2^lg >= 2N, A sits at [0, N) and E reversed, E_j at
// L/2 - j, so b_k k! is coefficient L/2 + k: the upper half in natural order. The transform is
// ntt::Product's layout (lib/ntt/product.hpp) with the last radix-2 level folded into the output
// pass. Factorials and c^j / j! come from product chains in 32 lanes whose multipliers are
// arithmetic progressions. Output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp).
#include <algorithm>
#include <array>
#include <bit>
#include <vector>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/huge.hpp"
#include "lib/ntt/product.hpp"
#include "lib/poly/calculus.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"
#include "problems/polynomial/polynomial_taylor_shift/factorials.hpp"

namespace {

using ntt::detail::add;
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
using poly::detail::load;
using poly::detail::low;
using poly::detail::store;

std::uint32_t inverse(std::uint32_t x) { return power(x, kP - 2); }

std::uint32_t mont(std::uint32_t x) { return multiply_mod(x, kR); }  // x 2^32 mod P

// Lanes 1, 3, 5, 7 of x in lanes 0, 2, 4, 6: the operands of _mm256_mul_epu32.
Vec odd_lanes(Vec x) { return _mm256_srli_epi64(x, 32); }

// a b / 2^32 mod P in [0, 2P) for a b < 2^32 P (for example a, b < 2P), with the odd lanes of a
// and b in the even lanes of a_odd and b_odd.
[[gnu::always_inline]] inline Vec montgomery(Vec a, Vec a_odd, Vec b, Vec b_odd) {
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    Vec even = _mm256_mul_epu32(a, b), odd = _mm256_mul_epu32(a_odd, b_odd);
    even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
    odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
    return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
}

// Rows become columns: lane l of r[t] moves to lane t of r[l], t, l < 8.
[[gnu::always_inline]] inline void transpose(Vec (&r)[8]) {
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

constexpr int kVectors = 4, kLanes = 8 * kVectors;

// x[v] = f(8 v + l) in lane l.
template <class F>
void fill(Vec (&x)[kVectors], F f) {
    alignas(32) std::uint32_t values[kLanes];
    for (int s = 0; s < kLanes; ++s) values[s] = f(s);
    for (int v = 0; v < kVectors; ++v) x[v] = load(values + 8 * v);
}

// 8 steps of the lanes: block[v][t] holds step t of lanes 8 v .. 8 v + 7.
using Block = Vec[kVectors][8];

// A product chain in kLanes lanes: x <- x (base + u) / 2^32, then u <- u + step mod P, so lane s
// multiplies by base_s + j step at its step j. x in [0, 2P); base and u in [0, P), so their sum
// is a valid montgomery() operand without a reduction.
struct Chain {
    Vec x[kVectors], base[kVectors], base_odd[kVectors], u, step;

    template <class X, class Base>
    Chain(X initial, Base base_of, std::uint32_t step_value) : u(_mm256_setzero_si256()), step(broadcast(step_value)) {
        fill(x, initial);
        fill(base, base_of);
        for (int v = 0; v < kVectors; ++v) base_odd[v] = odd_lanes(base[v]);
    }

    // Records the terms in block[v][t], then advances.
    [[gnu::always_inline]] void advance(Block& block, int t) {
#pragma GCC unroll 4
        for (int v = 0; v < kVectors; ++v) {
            block[v][t] = x[v];
            x[v] = montgomery(x[v], odd_lanes(x[v]), add(base[v], u), add(base_odd[v], u));
        }
        u = reduce(add(u, step), kP);
    }
};

// f(x + c) for n > 64 coefficients and c != 0. L = 2^lg >= 2n with lg even and >= 10 (L / 8 =
// 2 * 4^j vectors, subtrees of at least 16). Scans run over 32 lanes of C positions each, C a
// multiple of 16 with C / 16 odd, so the 32 streams of a pass fall into different cache sets;
// n <= 32 C <= n + 1023. Lanes run in lockstep, 8 steps per block; while the chains compute
// block j + 1, the lanes of block j are transposed into position order and stored.
// Buffers of L words: A, then the product, in a; E, then the output, in b. One mapping in huge
// pages, never freed; single use.
class TaylorShift {
public:
    explicit TaylorShift(std::size_t n) : n_(n) {
        lg_ = std::max(10, int(std::bit_width(2 * n - 1)));
        lg_ += lg_ % 2;
        chunk_ = ((n + kLanes - 1) / kLanes + 15) / 16 * 16;
        if (chunk_ / 16 % 2 == 0) chunk_ += 16;
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

    // factorial_[s] = start(s)!, inverse_start_[s] = 1 / start(s)!, s <= kLanes. Lane s computes
    // start(s + 1)!: from y = start(s + 1) rounded down to a multiple of 1024, x = y! 2^32 (the
    // table), multipliers (y + 1) 2^32, (y + 2) 2^32, ... for start(s + 1) - y < 1024 steps,
    // then 2^32.
    void lane_factorials() {
        using factorials::kStep;
        const auto below = [this](int s) { return std::uint32_t(start(s + 1) / kStep * kStep); };
        Vec x[kVectors], m[kVectors], count[kVectors];
        fill(x, [&](int s) { return mont(factorials::kTable[below(s) / kStep]); });
        fill(m, [&](int s) { return mont(below(s) + 1); });
        fill(count, [&](int s) { return std::uint32_t(start(s + 1) - below(s)); });
        const Vec unit = broadcast(kR);
        for (std::uint32_t j = 0; j < kStep - 1; ++j) {
            const Vec step = broadcast(j);
#pragma GCC unroll 4
            for (int v = 0; v < kVectors; ++v) {
                const Vec factor = _mm256_blendv_epi8(unit, m[v], _mm256_cmpgt_epi32(count[v], step));
                x[v] = montgomery(x[v], odd_lanes(x[v]), factor, odd_lanes(factor));
                m[v] = reduce(add(m[v], unit), kP);
            }
        }
        alignas(32) std::uint32_t lane_x[kLanes];
        for (int v = 0; v < kVectors; ++v) store(lane_x + 8 * v, x[v]);
        const std::uint32_t unit_inverse = inverse(kR);
        factorial_[0] = 1;
        for (int s = 0; s < kLanes; ++s) factorial_[s + 1] = multiply_mod(lane_x[s] % kP, unit_inverse);
        std::uint32_t f = 1;  // batch inversion
        for (int s = 0; s <= kLanes; ++s) inverse_start_[s] = f, f = multiply_mod(f, factorial_[s]);
        f = inverse(f);
        for (int s = kLanes; s >= 0; --s)
            inverse_start_[s] = multiply_mod(inverse_start_[s], f), f = multiply_mod(f, factorial_[s]);
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
        Chain input([this](int s) { return mont(factorial_[s]); },
                    [this](int s) { return mont(std::uint32_t(start(s) + 1)); }, kR);
        Chain kernel([&](int s) { return multiply_mod(c_power[s], inverse_start_[s + 1]); },
                     [&](int s) { return multiply_mod(std::uint32_t(start(s + 1)), c_step); }, kP - c_step);
        const std::size_t stride = chunk_;
        Block forward[2], backward[2];
#pragma GCC unroll 8
        for (int t = 0; t < 8; ++t) input.advance(forward[0], t), kernel.advance(backward[0], t);
        for (std::size_t j = 0, cur = 0; j < stride; j += 8, cur ^= 1) {
            std::uint32_t* const a = a_ + j;                         // lane s at a + s C
            std::uint32_t* const e = b_ + length() / 2 - stride + j;  // lane s at e - s C
#pragma GCC unroll 8
            for (int t = 0; t < 8; ++t) {
                input.advance(forward[cur ^ 1], t);
                kernel.advance(backward[cur ^ 1], t);
                const int v = kVectors * t / 8, first = kVectors * t % 8;
                if (first == 0) transpose(forward[cur][v]), transpose(backward[cur][v]);
#pragma GCC unroll 4
                for (int l = first; l < first + kVectors; ++l) {
                    const std::size_t at = (8 * v + l) * stride;
                    const Vec f = load(a + at), x = forward[cur][v][l];
                    store(a + at, reduce(montgomery(f, odd_lanes(f), x, odd_lanes(x)), kP));
                    store(e - at, reduce(backward[cur][v][l], kP));
                }
            }
        }
    }

    // b_k = (A E)[L/2 + k] / k! into b, from the halves u, w of a after their top inverse groups:
    // (A E)[L/2 + k] = (u - w)[k] times the transform's scale. Lane s of the chain starts at
    // k = start(s + 1) - 1 and runs down: z = (L / 8)^-1 2^64 / k!, multipliers k 2^32. The leaf
    // products carry 2^-32 and the transform a factor L / 8; montgomery() with z undoes both.
    // Outputs k >= n are padding.
    void output() {
        const std::uint32_t scale = multiply_mod(mont(inverse(std::uint32_t(length() / 8))), kR);
        Chain chain([&](int s) { return multiply_mod(multiply_mod(inverse_start_[s + 1], std::uint32_t(start(s + 1))), scale); },
                    [this](int s) { return mont(std::uint32_t(start(s + 1) - 1)); }, kP - kR);
        const std::size_t stride = chunk_, half = length() / 2;
        Block block[2];
#pragma GCC unroll 8
        for (int t = 0; t < 8; ++t) chain.advance(block[0], 7 - t);
        for (std::size_t j = 0, cur = 0; j < stride; j += 8, cur ^= 1) {
            const std::uint32_t* const u = a_ + stride - 8 - j;  // lane s at u + s C
            std::uint32_t* const out = b_ + stride - 8 - j;
#pragma GCC unroll 8
            for (int t = 0; t < 8; ++t) {
                chain.advance(block[cur ^ 1], 7 - t);
                const int v = kVectors * t / 8, first = kVectors * t % 8;
                if (first == 0) transpose(block[cur][v]);
#pragma GCC unroll 4
                for (int l = first; l < first + kVectors; ++l) {
                    const std::size_t at = (8 * v + l) * stride;
                    const Vec g = low(diff(load(u + at), load(u + at + half))), z = block[cur][v][l];
                    store(out + at, reduce(montgomery(g, odd_lanes(g), z, odd_lanes(z)), kP));
                }
            }
        }
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
