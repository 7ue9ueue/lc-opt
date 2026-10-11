// ABC222 H Beautiful Binary Tree, N <= 10^7, 3 s. A(x) = x (1 + 3A + A^2)^2 (editorial), so with
// S = A^2, D = 1 + 3A + S, T = D^2: a_n = t_(n-1). The coefficients come online: a_n needs t up
// to n - 1, d_n = 3 a_n + s_n needs a up to n - 1. This solution runs the two self-convolutions S
// and T as relaxed (online) squares by B-ary divide and conquer, O(N log^2 N); the editorial's
// solution is O(N), and it says even the O(N log N) power series route misses the time limit.
#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/poly.hpp"

namespace {

#ifndef BRANCH
#define BRANCH 16
#endif

using Span = std::span<std::uint32_t>;
using poly::detail::Vec;
constexpr std::uint32_t kP = easy::kMod;
constexpr std::uint32_t kR = std::uint32_t((std::uint64_t(1) << 32) % kP);  // Montgomery factor 2^32 mod P
constexpr std::size_t kLeaf = 32;        // block size solved by schoolbook (dot32); transforms >= 2 kLeaf = 64
constexpr std::size_t kBranch = BRANCH;  // children per node below the root
constexpr std::size_t kMaxLength = std::size_t(1) << 23;  // longest split transform: P - 1 = 119 2^23
constexpr std::size_t kMaxBranch = 64;

constexpr std::uint32_t add_mod(std::uint32_t x, std::uint32_t y) { return x + y >= kP ? x + y - kP : x + y; }
constexpr std::uint32_t sub_mod(std::uint32_t x, std::uint32_t y) { return x >= y ? x - y : x + kP - y; }

// sum_(i in [lo, hi]) f_i f_(n-i) mod P (each term < 2^60; 16 sum below 2^64).
std::uint32_t dot(const std::uint32_t* f, std::size_t lo, std::size_t hi, std::size_t n) {
    std::uint64_t sum = 0, part = 0;
    for (std::size_t i = lo; i <= hi; ++i) {
        part += std::uint64_t(f[i]) * f[n - i];
        if ((i - lo) % 16 == 15) sum += part % kP, part = 0;
    }
    return std::uint32_t((sum + part % kP) % kP);
}

// x y / 2^32 mod P in [0, 2P), x < 2^32, y < P: Montgomery.
inline Vec montgomery(Vec x, Vec y) {
    using namespace poly::detail;
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    Vec even = _mm256_mul_epu32(x, y), odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32));
    even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
    odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
    return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
}

// lib/poly's transform stops at leaves a mod (x^8 - w_p), multiplied by 8 x 8 products. Split
// takes each leaf on to its values at the 8 roots of x^8 - w_p, rho_p zeta^i (rho_p^8 = w_p,
// zeta of order 8), so products of transforms become pointwise. Layout: 8 leaves per 64 words,
// word 8 i + t is value i (bit-reversed order) of leaf 8 g + t.
//
// w_p = r[p / 2] (even p) or -r[p / 2] (odd p), r[m] the product of root[j] over the set bits j
// of m, root[j] of order 2^(j + 2) (lib/poly/transform.hpp). So rho_p is the same product of
// root[j + 3], times root[2] (an 8th root of -1) for odd p, and rho_(8g + t) = s_g rho_t with
// s_g the product of root[j + 5] over the set bits j of g.
class Split {
public:
    // Tables for leaves [0, 8 groups), groups <= 2^16.
    explicit Split(std::size_t groups) : power_(8 * groups), inverse_power_(8 * groups) {
        std::uint32_t root[22], inverse_root[22];
        for (int i = 0; i < 22; ++i) {
            root[i] = easy::power(3, (kP - 1) >> (i + 2));
            inverse_root[i] = easy::power(root[i], kP - 2);
        }
        for (int k = 0; k < 4; ++k) {
            zeta_[k] = poly::detail::Factor(easy::power(root[1], k));
            inverse_zeta_[k] = poly::detail::Factor(easy::power(inverse_root[1], k));
        }
        const std::uint32_t inverse_eight = easy::power(8, kP - 2);
        for (int t = 0; t < 8; ++t) {
            std::uint32_t rho = 1, inverse_rho = 1;
            for (int j = 0; j < 2; ++j)
                if (t >> (j + 1) & 1) rho = easy::mul(rho, root[j + 3]), inverse_rho = easy::mul(inverse_rho, inverse_root[j + 3]);
            if (t & 1) rho = easy::mul(rho, root[2]), inverse_rho = easy::mul(inverse_rho, inverse_root[2]);
            std::uint32_t power = 1, inverse_power = inverse_eight;
            for (int k = 0; k < 8; ++k) {
                set(lane_[k], t, power);
                set(inverse_lane_[k], t, inverse_power);
                power = easy::mul(power, rho), inverse_power = easy::mul(inverse_power, inverse_rho);
            }
        }
        std::vector<std::uint32_t> s(groups, 1), inverse_s(groups, 1);
        for (std::size_t g = 1; g < groups; ++g) {
            const int j = std::countr_zero(g);
            s[g] = easy::mul(s[g & (g - 1)], root[j + 5]);
            inverse_s[g] = easy::mul(inverse_s[g & (g - 1)], inverse_root[j + 5]);
        }
        for (std::size_t g = 0; g < groups; ++g) {
            std::uint32_t power = kR, inverse_power = kR;  // Montgomery form
            for (int k = 0; k < 8; ++k) {
                power_[8 * g + std::size_t(k)] = power, inverse_power_[8 * g + std::size_t(k)] = inverse_power;
                power = easy::mul(power, s[g]), inverse_power = easy::mul(inverse_power, inverse_s[g]);
            }
        }
    }

    // In place, canonical in and out.
    void forward(Span a) const {
        using namespace poly::detail;
        for (std::size_t g = 0; g < a.size() / 64; ++g) {
            std::uint32_t* x = a.data() + 64 * g;
            Vec r[8];
            for (int t = 0; t < 8; ++t) r[t] = load(x + 8 * t);
            transpose(r);
            for (int k = 1; k < 8; ++k)
                r[k] = reduce(montgomery(times(r[k], lane_[k]), broadcast(power_[8 * g + std::size_t(k)])), kP);
            for (int half = 4; half >= 1; half /= 2)
                for (int b = 0; b < 8; b += 2 * half)
                    for (int k = 0; k < half; ++k) {
                        const Vec u = r[b + k], v = r[b + k + half];
                        r[b + k] = reduce(add(u, v), kP);
                        r[b + k + half] = k == 0 ? reduce(reduce(diff(u, v), 2 * kP), kP)
                                                 : reduce(times(diff(u, v), zeta_[k * 4 / half]), kP);
                    }
            for (int t = 0; t < 8; ++t) store(x + 8 * t, r[t]);
        }
    }

    // The inverse of forward.
    void inverse(Span a) const {
        using namespace poly::detail;
        for (std::size_t g = 0; g < a.size() / 64; ++g) {
            std::uint32_t* x = a.data() + 64 * g;
            Vec r[8];
            for (int t = 0; t < 8; ++t) r[t] = load(x + 8 * t);
            for (int half = 1; half <= 4; half *= 2)
                for (int b = 0; b < 8; b += 2 * half)
                    for (int k = 0; k < half; ++k) {
                        const Vec u = r[b + k];
                        const Vec v = k == 0 ? r[b + k + half] : reduce(times(r[b + k + half], inverse_zeta_[k * 4 / half]), kP);
                        r[b + k] = reduce(add(u, v), kP);
                        r[b + k + half] = reduce(_mm256_sub_epi32(add(u, broadcast(kP)), v), kP);
                    }
            r[0] = reduce(times(r[0], inverse_lane_[0]), kP);
            for (int k = 1; k < 8; ++k)
                r[k] = reduce(montgomery(times(r[k], inverse_lane_[k]), broadcast(inverse_power_[8 * g + std::size_t(k)])), kP);
            transpose(r);
            for (int t = 0; t < 8; ++t) store(x + 8 * t, r[t]);
        }
    }

private:
    static void set(poly::detail::Factors& f, int lane, std::uint32_t w) {
        reinterpret_cast<std::uint32_t*>(&f.w)[lane] = w;
        reinterpret_cast<std::uint32_t*>(&f.q)[lane] = ntt::detail::quotient(w);
    }

    static void transpose(Vec (&r)[8]) {
        Vec t[8], u[8];
        for (int i = 0; i < 8; i += 2) t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]), t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
        for (int i = 0; i < 8; i += 4)
            for (int j = 0; j < 2; ++j) {
                u[i + 2 * j] = _mm256_unpacklo_epi64(t[i + j], t[i + j + 2]);
                u[i + 2 * j + 1] = _mm256_unpackhi_epi64(t[i + j], t[i + j + 2]);
            }
        for (int i = 0; i < 4; ++i) {
            r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
            r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
        }
    }

    poly::detail::Factors lane_[8], inverse_lane_[8];  // rho_t^k; rho_t^-k / 8
    poly::detail::Factor zeta_[4] = {poly::detail::Factor(1), poly::detail::Factor(1), poly::detail::Factor(1), poly::detail::Factor(1)};
    poly::detail::Factor inverse_zeta_[4] = {poly::detail::Factor(1), poly::detail::Factor(1), poly::detail::Factor(1), poly::detail::Factor(1)};
    std::vector<std::uint32_t> power_, inverse_power_;  // s_g^k 2^32, s_g^-k 2^32 at 8 g + k
};

// out = sum_j a_j b_j / 2^32 mod P, pointwise, canonical inputs. Groups of 8 products sum below
// 8 P^2, and with the Montgomery term below 2^64.
void sum_of_products(std::span<const Span> a, std::span<const Span> b, Span out) {
    using namespace poly::detail;
    const Vec ni = broadcast(ntt::kernels::kNI), p = broadcast(kP);
    for (std::size_t v = 0; v < out.size(); v += 8) {
        Vec total = _mm256_setzero_si256();
        for (std::size_t j0 = 0; j0 < a.size(); j0 += 8) {
            Vec even = _mm256_setzero_si256(), odd = _mm256_setzero_si256();
            for (std::size_t j = j0; j < std::min(a.size(), j0 + 8); ++j) {
                const Vec x = load(a[j].data() + v), y = load(b[j].data() + v);
                even = _mm256_add_epi64(even, _mm256_mul_epu32(x, y));
                odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32)));
            }
            even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_mul_epu32(even, ni), p));
            odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_mul_epu32(odd, ni), p));
            const Vec sum = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);  // < 3P
            total = reduce(add(total, reduce(reduce(sum, 2 * kP), kP)), kP);
        }
        store(out.data() + v, total);
    }
}

// sum = (sum + x^c lag) 2^32 in the split layout: x^c is 1 on leaves below c / 8, -1 above.
void combine(Span sum, Span lag, std::size_t c) {
    using namespace poly::detail;
    const Factor r(kR);
    const Vec lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (std::size_t g = 0; g < sum.size() / 64; ++g) {
        const int below = int(std::clamp<std::size_t>(c / 8, 8 * g, 8 * g + 8) - 8 * g);
        const Vec lower = _mm256_cmpgt_epi32(broadcast(std::uint32_t(below)), lane);
        for (std::size_t v = 64 * g; v < 64 * g + 64; v += 8) {
            const Vec u = load(sum.data() + v), w = load(lag.data() + v);
            const Vec x = _mm256_blendv_epi8(_mm256_sub_epi32(add(u, broadcast(kP)), w), add(u, w), lower);
            store(sum.data() + v, reduce(times(x, r), kP));
        }
    }
}

// sum_(j<32) u_j w_j / 2^32 mod P for canonical u (aligned) and w: each 64-bit lane sums 8
// products below 8 P^2, plus the Montgomery term below 2^64; the 4 lanes then sum below 12 P.
std::uint32_t dot32(const std::uint32_t* u, const std::uint32_t* w) {
    using namespace poly::detail;
    Vec sum = _mm256_setzero_si256();
    for (int v = 0; v < 4; ++v) {
        const Vec x = load(u + 8 * v), y = _mm256_loadu_si256(reinterpret_cast<const Vec*>(w + 8 * v));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(x, y));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32)));
    }
    const Vec q = _mm256_mul_epu32(sum, broadcast(ntt::kernels::kNI));
    sum = _mm256_srli_epi64(_mm256_add_epi64(sum, _mm256_mul_epu32(q, broadcast(kP))), 32);
    const __m128i half = _mm_add_epi64(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    return std::uint32_t(std::uint64_t(_mm_cvtsi128_si64(_mm_add_epi64(half, _mm_unpackhi_epi64(half, half)))) % kP);
}

// Series 0 is A with accumulator S = A^2, series 1 is D with T = D^2. Node [l, l + B c) of a level
// splits into B blocks of c. A node with l = 0 adds to block k the products of blocks a, b < k;
// a node with l > 0 (then l >= B c) adds 2 f[l + jc, l + (j + 1) c) f[(k - j - 1) c, (k - j + 1) c)
// for j < k: a middle product, exact in a cyclic transform of 2c at outputs [c, 2c). The second
// factor's transform G_(k-j) depends only on the level, so each level computes it once. All
// block transforms are split (Split), so the sums over j are pointwise.
class RelaxedSquares {
public:
    explicit RelaxedSquares(std::size_t n)
        : n_(n), arena_(words(n)), transform_(arena_, lg_max(n)), split_(2 * root_child(n) / 64) {
        for (int s = 0; s < 2; ++s) value_[s] = arena_.take(n + kLeaf), square_[s] = arena_.take(n);
        for (const std::size_t c : children(n)) {
            Level& level = levels_.emplace_back();
            level.child = c;
            const bool root = c == root_child(n);
            level.branch = root ? (n + c - 1) / c : kBranch;
            for (int s = 0; s < 2; ++s)
                for (std::size_t j = 0; j + 1 < level.branch; ++j) {
                    level.block[s].push_back(arena_.take(2 * c));
                    if (!root) level.segment[s].push_back(arena_.take(2 * c));
                }
            level.sum = arena_.take(2 * c);
            level.lag = arena_.take(2 * c);
        }
    }

    // t_(n-1).
    std::uint32_t run() {
        solve(levels_.size(), 0, true);
        return square_[1][n_ - 1];
    }

private:
    struct Level {
        std::size_t child = 0, branch = 0;
        std::vector<Span> block[2];    // transforms of the finished blocks of the current node, split
        std::vector<Span> segment[2];  // G_delta, delta = 1 .. branch - 1, split, times 2^32
        Span sum, lag;
        bool ready = false;  // segment computed
    };

    // The child sizes of the levels, bottom up. The root has ceil(n / c) children: more than kBranch
    // when its children would need transforms longer than kMaxLength.
    static std::vector<std::size_t> children(std::size_t n) {
        std::vector<std::size_t> sizes;
        for (std::size_t c = kLeaf; c < n; c *= kBranch) {
            sizes.push_back(c);
            if (c * kBranch >= n || 2 * c * kBranch > kMaxLength) break;
        }
        return sizes;
    }

    static std::size_t root_child(std::size_t n) { return n > kLeaf ? children(n).back() : kLeaf; }

    static int lg_max(std::size_t n) { return std::max(poly::Transform::kMinLog, int(std::bit_width(2 * root_child(n) - 1))); }

    static std::size_t words(std::size_t n) {
        std::size_t w = poly::Transform::words(lg_max(n)) + 4 * poly::Arena::footprint(n + kLeaf);
        for (const std::size_t c : children(n)) {
            const std::size_t root = c == root_child(n), branch = root ? (n + c - 1) / c : kBranch;
            w += ((root ? 2 : 4) * (branch - 1) + 2) * poly::Arena::footprint(2 * c);
        }
        return w;
    }

    void solve(std::size_t depth, std::size_t l, bool zero) {
        if (depth == 0) return zero ? first_leaf(std::min(kLeaf, n_)) : leaf(l, std::min(l + kLeaf, n_));
        Level& level = levels_[depth - 1];
        const std::size_t c = level.child;
        if (!zero && !level.ready) make_segments(level);
        for (std::size_t k = 0; k < level.branch && l + k * c < n_; ++k) {
            const std::size_t start = l + k * c;
            if (k > 0)
                for (int s = 0; s < 2; ++s) zero ? add_square_terms(level, s, k, start) : add_middle_terms(level, s, k, start);
            solve(depth - 1, start, zero && k == 0);
            if (k + 1 < level.branch && start + c < n_)
                for (int s = 0; s < 2; ++s) {
                    transform_.forward(value_[s].subspan(start, c), 0, level.block[s][k]);
                    split_.forward(level.block[s][k]);
                }
        }
    }

    // G_delta = the transform of 2^33 f[(delta - 1) c, (delta + 1) c), split.
    void make_segments(Level& level) {
        const std::size_t c = level.child;
        const std::uint32_t scale = add_mod(kR, kR);
        for (int s = 0; s < 2; ++s)
            for (std::size_t delta = 1; delta < level.branch; ++delta) {
                const Span g = level.segment[s][delta - 1];
                for (std::size_t i = 0; i < 2 * c; ++i) g[i] = easy::mul(value_[s][(delta - 1) * c + i], scale);
                transform_.forward(g);
                split_.forward(g);
            }
        level.ready = true;
    }

    // l > 0: block k += [x^(c..2c)] sum_(j<k) F_j G_(k-j).
    void add_middle_terms(Level& level, int s, std::size_t k, std::size_t start) {
        const std::size_t c = level.child;
        Span a[kMaxBranch], b[kMaxBranch];
        for (std::size_t j = 0; j < k; ++j) a[j] = level.block[s][j], b[j] = level.segment[s][k - j - 1];
        sum_of_products(std::span(a, k), std::span(b, k), level.sum);
        split_.inverse(level.sum);
        transform_.inverse(level.sum, poly::Half::kUpper);
        accumulate(s, start, level.sum.subspan(c, c));
    }

    // l = 0: block k += [x^(0..c)] (sum_(a+b=k) F_a F_b + x^c sum_(a+b=k-1) F_a F_b), a, b < k.
    void add_square_terms(Level& level, int s, std::size_t k, std::size_t start) {
        const std::size_t c = level.child;
        const auto& f = level.block[s];
        Span a[kMaxBranch], b[kMaxBranch];
        for (std::size_t i = 1; i < k; ++i) a[i - 1] = f[i], b[i - 1] = f[k - i];
        sum_of_products(std::span(a, k - 1), std::span(b, k - 1), level.sum);
        for (std::size_t i = 0; i < k; ++i) a[i] = f[i], b[i] = f[k - 1 - i];
        sum_of_products(std::span(a, k), std::span(b, k), level.lag);
        combine(level.sum, level.lag, c);
        split_.inverse(level.sum);
        transform_.inverse(level.sum, poly::Half::kLower);
        accumulate(s, start, level.sum.first(c));
    }

    void accumulate(int s, std::size_t start, Span terms) {
        const std::size_t count = std::min(terms.size(), n_ - start);
        for (std::size_t i = 0; i < count; ++i) square_[s][start + i] = add_mod(square_[s][start + i], terms[i]);
    }

    // [0, r): every pair by schoolbook. Then the reversed prefixes for leaf().
    void first_leaf(std::size_t r) {
        const Span a = value_[0], d = value_[1], sa = square_[0], td = square_[1];
        for (std::size_t n = 0; n < r; ++n) {
            a[n] = n == 0 ? 0 : td[n - 1];
            sa[n] = add_mod(sa[n], dot(a.data(), 0, n, n));
            d[n] = std::uint32_t((3 * std::uint64_t(a[n]) + sa[n] + (n == 0)) % kP);
            td[n] = add_mod(td[n], dot(d.data(), 0, n, n));
        }
        for (int s = 0; s < 2; ++s)
            for (std::size_t x = 0; x + 1 < std::min(kLeaf, r); ++x) reversed_[s][x] = easy::mul(value_[s][kLeaf - 1 - x], kR);
    }

    // l > 0 owns the pairs (i, n - i) with i >= l, twice: with m = n - l and u_j = f_(l+j), the sum
    // over j <= m of u_j f_(m-j), a 32-term dot product with the reversed prefix at offset 31 - m
    // (zero past m). The term j = m, f_n f_0, is added apart: f_n is not known yet.
    void leaf(std::size_t l, std::size_t r) {
        const Span a = value_[0], d = value_[1], sa = square_[0], td = square_[1];
        for (std::size_t n = l; n < r; ++n) {
            const std::size_t at = kLeaf - 1 - (n - l);
            const std::uint32_t own_a = dot32(&a[l], &reversed_[0][at]), own_d = dot32(&d[l], &reversed_[1][at]);
            a[n] = td[n - 1];
            sa[n] = add_mod(sa[n], add_mod(own_a, own_a));
            d[n] = std::uint32_t((3 * std::uint64_t(a[n]) + sa[n]) % kP);
            td[n] = std::uint32_t((td[n] + 2 * std::uint64_t(own_d) + 2 * std::uint64_t(d[n])) % kP);  // d_0 = 1
        }
    }

    std::size_t n_;
    poly::Arena arena_;
    poly::Transform transform_;
    Split split_;
    Span value_[2], square_[2];
    std::vector<Level> levels_;
    alignas(32) std::uint32_t reversed_[2][2 * kLeaf] = {};  // f_(31-x) 2^32 for x < 31, then zeros
};

}  // namespace

int main() {
    easy::Reader in;
    const auto n = in.read<std::uint32_t>();
    RelaxedSquares squares(n);
    easy::Writer out;
    out.write(squares.run(), '\n');
}
