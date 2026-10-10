// c_k = sum over i j = k (mod 2^N) of a_i b_j mod 998244353, N <= 20.
//
// Index i = 2^s u with u odd is in level s. The odd residues mod 2^M (M = N - s >= 2) are
// +-u with u = 4i + 1, i < L = 2^(M-2); the sign transform (x+ = x(u) + x(-u), x- = x(u) - x(-u))
// splits level s into two functions on G = <5> = {u = 1 mod 4}, a cyclic group of order L. A
// product of levels s and t lands in level s + t, as the product of the units mod 2^m,
// m = N - s - t: both factors folded to u mod 2^m, one convolution over G per sign. Products with
// s + t >= N land on 0.
//
// Transforms over G run in natural order of i, so splitting the input into levels is a
// deinterleave. Stage H (blocks of 2H positions, H >= 8) maps (x_i, x_i+H), i < H, to
// (x_i + x_i+H, (x_i - x_i+H) t_H(i)), t_H(i) = r_2H^(dlog_5(4i + 1) mod 2H) (r_2H of order 2H):
// the character twist of a decimation in frequency, which leaves every block a function on the
// quotient group. Blocks of 8 positions (u mod 32) are leaves: cyclic convolutions of length 8 in
// discrete-log order. The first 2^(m-2) words of a transform of length L are the transform of the
// input folded to u mod 2^m, so each level of a and b is transformed once (2^N words per factor),
// and output level m sums the leaf products of its pairs and runs one inverse transform. Levels
// m <= 7 are convolved directly.
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
#include "lib/run/early.hpp"
#include "../convolution_mod/fields.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using ntt::detail::add;
using ntt::detail::broadcast;
using ntt::detail::diff;
using ntt::detail::Factor;
using ntt::detail::kP;
using ntt::detail::multiply;
using ntt::detail::reduce;
using ntt::detail::Vec;

constexpr int kMaxLog = 20;
constexpr int kDirectLog = 7;                                      // output levels m <= 7: direct
constexpr std::size_t kFold = std::size_t(1) << (kDirectLog - 2);  // their longest sequence
constexpr std::size_t kPadding = 16;                               // words after each array: loads read past

u32 add_mod(u32 x, u32 y) {
    const u32 s = x + y;
    return std::min(s, s - kP);
}
u32 sub_mod(u32 x, u32 y) {
    const u32 d = x - y;
    return std::min(d, d + kP);
}
u32 mul_mod(u32 x, u32 y) { return u32(u64(x) * y % kP); }

constexpr u32 kHalf = (kP + 1) / 2;

Vec load(const u32* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
void store(u32* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }

// x < 4P -> x mod 2P (low), x mod P (canonical).
Vec low(Vec x) { return reduce(x, 2 * kP); }
Vec canonical(Vec x) { return reduce(low(x), kP); }
// x - y mod 2P in [0, 2P) for x, y < 2P.
Vec low_difference(Vec x, Vec y) {
    const Vec d = _mm256_sub_epi32(x, y);
    return _mm256_min_epu32(d, _mm256_add_epi32(d, broadcast(2 * kP)));
}

// x w mod P in [0, 2P) for any x < 2^32, per lane: q holds the Shoup quotients floor(w 2^32 / P),
// q_odd those of the odd lanes in the even lanes.
[[gnu::always_inline]] inline Vec shoup(Vec x, Vec w, Vec q, Vec q_odd) {
    const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(x, q), 32);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), q_odd);
    const Vec quotient = _mm256_blend_epi32(even, odd, 0xAA);
    return _mm256_sub_epi32(_mm256_mullo_epi32(x, w), _mm256_mullo_epi32(quotient, broadcast(kP)));
}

// By a table line [q(8), w(8)]: the odd quotients load 4 bytes further on, within the line.
[[gnu::always_inline]] inline Vec shoup(Vec x, const u32* line) {
    return shoup(x, load(line + 8), load(line), load(line + 1));
}

// By w with quotients q, in registers.
Vec shoup(Vec x, Vec w, Vec q) { return shoup(x, w, q, _mm256_srli_epi64(q, 32)); }

// Shoup quotients floor(w 2^32 / P) of canonical w: (w 2^32 mod P) (-1 / P) mod 2^32.
Vec quotients(Vec w) {
    const Factor to_r(ntt::detail::kR);
    return _mm256_mullo_epi32(reduce(multiply(w, to_r), kP), broadcast(ntt::kernels::kNI));
}

// Bump allocation from one mapping in 2 MiB pages where the kernel allows. Never freed.
class Arena {
public:
    explicit Arena(std::size_t bytes) {
        constexpr std::size_t kHuge = std::size_t(1) << 21;
        bytes = (bytes + kHuge - 1) / kHuge * kHuge;
        void* p = ::mmap(nullptr, bytes + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) std::abort();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(p) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
        ::madvise(reinterpret_cast<void*>(aligned), bytes, MADV_HUGEPAGE);
#endif
        next_ = reinterpret_cast<char*>(aligned);
        end_ = next_ + bytes;
    }

    // count words, 64-byte aligned, zero, then kPadding readable words.
    u32* words(std::size_t count) { return reinterpret_cast<u32*>(take(4 * (count + kPadding))); }
    char* bytes(std::size_t count) { return take(count); }

private:
    char* take(std::size_t count) {
        char* p = next_;
        next_ += (count + 63) & ~std::size_t(63);
        if (next_ > end_) std::abort();
        return p;
    }

    char *next_, *end_;
};

// Discrete log of u = 1 mod 4 to base 5 modulo 2^n: 5^k = u. 5^(2^j) = 1 + 2^(j+2) mod 2^(j+3).
u32 dlog(u32 u, int n) {
    const u32 mask = (u32(1) << n) - 1;
    u32 k = 0, w = 1, g = 5;  // w = 5^k = u mod 2^(j+2)
    for (int j = 0; j + 2 < n; ++j, g = g * g & mask)
        if ((u ^ w) & ((u32(8) << j) - 1)) w = w * g & mask, k |= u32(1) << j;
    return k;
}

// u^-1 mod 2^32 for odd u (Newton: each step doubles the correct low bits, from 3).
u32 inverse_odd(u32 u) {
    u32 x = u;
    for (int i = 0; i < 4; ++i) x *= 2 - u * x;
    return x;
}

// Twiddles of the transform (see the header) for levels of up to 2^(n-2) positions, n >= 8.
// A radix-4 step on blocks of 4H positions (H = 8h, h vectors) takes tau_i = t_2H(i), i < H:
// t_2H(i + H) = J tau_i (J = r_4^-1: 1 + 4H = 5^(3H) mod 16H) and t_H(i) = tau_i^2. Its table
// holds, per vector of 8 positions, three lines [q, w] of tau, tau^2, tau^3 (forward) or of their
// inverses (inverse). For even n, level 0 starts with a radix-2 step, H = 2^(n-3):
// t_H(i + H/2) = J t_H(i); its table holds one line [q, w] of t_H(i), i < H/2, per vector.
// The tables hold the forward twiddles until invert() replaces them by the inverse ones.
class Tables {
public:
    static constexpr std::size_t kTau = 0, kTau2 = 16, kTau3 = 32, kEntry = 48;

    struct Lines {
        const u32* base;
        std::size_t stride;
    };

    Tables(Arena& arena, int n) : n_(n) {
        if (n < kDirectLog + 1) return;
        h_max_ = std::size_t(1) << (n % 2 ? n - 7 : n - 8);
        pairs_ = arena.words(offset(4 * h_max_));
        if (n % 2 == 0) top_ = arena.words(std::size_t(16) << (n - 7));
        build(0);
        fold_entry(1);
    }

    void invert() { build(1); }

    // Radix-4 steps with quarter h vectors (h a power of 4).
    const u32* pair(std::size_t h) const { return pairs_ + offset(h); }

    // The radix-2 step of a level with nv vectors: lines of t_H(i), H = 4 nv, i < H/2.
    Lines radix2(std::size_t nv) const {
        if (nv == std::size_t(1) << (n_ - 5)) return {top_, 16};
        return {pair(nv / 2) + kTau2, kEntry};
    }

    // J, or J^-1 after invert().
    Factor j() const { return Factor(ntt::detail::kRoots[1 - direction_][0]); }

    // The inverse entry of pair(1), for folds while the tables are forward.
    const u32* inverse_bottom() const { return inverse_bottom_; }

private:
    static std::size_t offset(std::size_t h) { return 16 * (h - 1); }  // kEntry (1 + 4 + ... + h/4)

    // The tables of one direction.
    void build(int direction) {
        direction_ = direction;
        u32* tau = pairs_ + offset(h_max_) + kTau;
        characters(direction, n_ % 2 ? tau + 8 : top_ + 8, n_ % 2 ? kEntry : 16);
        if (n_ % 2 == 0) {
            for (std::size_t j = 0; j < std::size_t(1) << (n_ - 7); ++j) store(top_ + 16 * j, quotients(load(top_ + 16 * j + 8)));
            for (std::size_t j = 0; j < h_max_; ++j)  // t_2H = t_4H^2, H = 8 h_max
                store(tau + kEntry * j + 8, canonical(shoup(load(top_ + 16 * j + 8), top_ + 16 * j)));
        }
        for (std::size_t h = h_max_;; h /= 4) {
            u32* t = pairs_ + offset(h);
            for (std::size_t j = 0; j < h; ++j, t += kEntry) {
                const Vec w = load(t + kTau + 8), q = quotients(w);
                if (j < h / 4) {  // tau of pair(h / 4) is tau^4
                    const Vec w2 = canonical(shoup(w, w, q));
                    store(pairs_ + offset(h / 4) + kEntry * j + kTau + 8, canonical(shoup(w2, w2, quotients(w2))));
                }
                write_entry(t, w, q);
            }
            if (h == 1) break;
        }
    }

    // Lines of tau, tau^2, tau^3 at t from tau = w with quotients q.
    static void write_entry(u32* t, Vec w, Vec q) {
        const Vec w2 = canonical(shoup(w, w, q)), w3 = canonical(shoup(w2, w, q));
        store(t + kTau, q), store(t + kTau + 8, w);
        store(t + kTau2, quotients(w2)), store(t + kTau2 + 8, w2);
        store(t + kTau3, quotients(w3)), store(t + kTau3 + 8, w3);
    }

    // pair(1) of a direction into inverse_bottom_: tau_i = t_16(i) = r_32^(dlog(4i + 1) mod 32).
    void fold_entry(int direction) {
        alignas(32) u32 w[8];
        for (u32 i = 0; i < 8; ++i) w[i] = ntt::detail::power(ntt::detail::kRoots[direction][3], dlog(4 * i + 1, 7));
        const Vec v = load(w);
        write_entry(inverse_bottom_, v, quotients(v));
    }

    // chi(4i + 1) = r^dlog(4i + 1) for i < 2^(n-4), r = kRoots[direction][n - 4] (order 2^(n-2));
    // vector i / 8 at out + stride i / 8.
    // With u = u_lo + 2^b u_hi and 2b >= n, u = u_lo (1 + 2^b z), z = u_hi u_lo^-1 mod 2^(n-b), and
    // 1 + 2^b z = (1 + 2^b)^z: the value is chi(u_lo) theta^z, theta = chi(1 + 2^b). For eight
    // consecutive u_lo it is a geometric sequence in u_hi, ratios theta^(u_lo^-1); all sequences of
    // a row are independent.
    void characters(int direction, u32* out, std::size_t stride) const {
        constexpr int kMaxB = (kMaxLog + 1) / 2;
        const int n = n_, b = std::max(5, (n + 1) / 2);
        const std::size_t columns = std::size_t(1) << (b - 2), rows = std::size_t(1) << (n - 2 - b);
        const u32 z_mask = (u32(1) << (n - b)) - 1;
        u32 powers[kMaxLog];  // r^(2^j)
        powers[0] = ntt::detail::kRoots[direction][n - 4];
        for (int j = 1; j < n - 2; ++j) powers[j] = mul_mod(powers[j - 1], powers[j - 1]);
        const auto chi = [&](u32 u) {
            const u32 k = dlog(u, n);
            u32 r = 1;
            for (int j = 0; j < n - 2; ++j)
                if (k >> j & 1) r = mul_mod(r, powers[j]);
            return r;
        };
        std::array<u32, std::size_t(1) << (kMaxLog / 2)> thetas;  // theta^z, z < 2^(n-b)
        thetas[0] = 1;
        const u32 theta = chi(1 + (u32(1) << b));
        for (u32 z = 1; z <= z_mask; ++z) thetas[z] = mul_mod(thetas[z - 1], theta);
        alignas(32) u32 ratio[2][std::size_t(1) << (kMaxB - 2)];  // [w, q][u_lo / 4]
        for (std::size_t c = 0; c < columns; ++c) {
            const u32 u = u32(4 * c + 1);
            out[stride * (c / 8) + c % 8] = chi(u);
            ratio[0][c] = thetas[inverse_odd(u) & z_mask];
            ratio[1][c] = ntt::detail::quotient(ratio[0][c]);
        }
        for (std::size_t row = 1; row < rows; ++row)
            for (std::size_t g = 0; g < columns; g += 8) {
                u32* at = out + stride * ((g + row * columns) / 8);
                const Vec v = load(at - stride * (columns / 8));
                store(at, canonical(shoup(v, load(ratio[0] + g), load(ratio[1] + g))));
            }
    }

    int n_, direction_ = 0;
    std::size_t h_max_ = 0;
    u32* pairs_ = nullptr;
    u32* top_ = nullptr;
    alignas(32) u32 inverse_bottom_[kEntry] = {};
};

// Radix-4 step on f[0], f[h], f[2h], f[3h] with the table entry t (inputs < 2P, outputs < 2P;
// canonical if Last).
template <bool Last>
[[gnu::always_inline]] inline void forward4(Vec* f, std::size_t h, const u32* t, const Factor& rot) {
    const Vec x0 = f[0], x1 = f[h], x2 = f[2 * h], x3 = f[3 * h];
    const Vec s02 = low(add(x0, x2)), s13 = low(add(x1, x3));
    const Vec e = low(diff(x0, x2)), g = multiply(diff(x1, x3), rot);
    const Vec y0 = low(add(s02, s13)), y1 = shoup(diff(s02, s13), t + Tables::kTau2);
    const Vec y2 = shoup(add(e, g), t + Tables::kTau), y3 = shoup(diff(e, g), t + Tables::kTau3);
    if constexpr (Last) {
        f[0] = reduce(y0, kP), f[h] = reduce(y1, kP), f[2 * h] = reduce(y2, kP), f[3 * h] = reduce(y3, kP);
    } else {
        f[0] = y0, f[h] = y1, f[2 * h] = y2, f[3 * h] = y3;
    }
}

// Inverse of forward4 times 4 (inputs < 4P, outputs < 4P).
[[gnu::always_inline]] inline void inverse4(Vec* f, std::size_t h, const u32* t, const Factor& rot) {
    const Vec a = low(f[0]), b = shoup(f[h], t + Tables::kTau2);
    const Vec c = shoup(f[2 * h], t + Tables::kTau), d = shoup(f[3 * h], t + Tables::kTau3);
    const Vec ab = low(add(a, b)), amb = low_difference(a, b);
    const Vec cd = low(add(c, d)), j = multiply(diff(c, d), rot);
    f[0] = add(ab, cd), f[2 * h] = diff(ab, cd);
    f[h] = add(amb, j), f[3 * h] = diff(amb, j);
}

// Forward and inverse transforms of one level, both signs (x: +, y: -) at once.
class Transform {
public:
    explicit Transform(const Tables& tables) : tables_(tables) {}

    // [x, y] = [P, R] with P[i] = v(4i + 1), R[i] = v(-(4i + 1)), canonical -> leaves of x+, x-,
    // canonical.
    void forward(Vec* x, Vec* y, std::size_t nv) const {
        const Factor rot = tables_.j();
        for (std::size_t j = 0; j < nv; ++j) signs(x + j, y + j);
        const std::size_t nb = top_block(nv);
        for (Vec* f : {x, y}) {
            if (nb == nv / 2) {
                const Tables::Lines t = tables_.radix2(nv);
                for (std::size_t j = 0; j < nb / 2; ++j) top2(f + j, nb, t.base + t.stride * j);
                for (std::size_t j = 0; j < nb / 2; ++j) top2(f + nb / 2 + j, nb, t.base + t.stride * j, &rot);
            } else {
                const u32* t = tables_.pair(nb);
                for (std::size_t j = 0; j < nb; ++j) forward4<false>(f + j, nb, t + Tables::kEntry * j, rot);
            }
        }
        for (std::size_t o = 0; o < nv; o += nb) block(x + o, y + o, nb, rot);
    }

    // Leaves (< 4P) of c+, c- -> [P, R] of scale (c+ +- c-) / nv, canonical.
    void inverse(Vec* x, Vec* y, std::size_t nv, u32 scale_value) const {
        const Factor rot = tables_.j(), scale(scale_value);
        const std::size_t nb = top_block(nv);
        for (std::size_t o = 0; o < nv; o += nb) inverse_block(x + o, y + o, nb, rot);
        for (Vec* f : {x, y}) {
            if (nb == nv / 2) {
                const Tables::Lines t = tables_.radix2(nv);
                for (std::size_t j = 0; j < nb / 2; ++j) inverse_top2(f + j, nb, t.base + t.stride * j);
                for (std::size_t j = 0; j < nb / 2; ++j) inverse_top2(f + nb / 2 + j, nb, t.base + t.stride * j, &rot);
            } else {
                const u32* t = tables_.pair(nb);
                for (std::size_t j = 0; j < nb; ++j) inverse4(f + j, nb, t + Tables::kEntry * j, rot);
            }
        }
        for (std::size_t j = 0; j < nv; ++j) combine(x + j, y + j, scale);
    }

    // Canonical leaves 0..3 of a transform of nv >= 4 vectors -> its input folded to 32 words
    // (u mod 2^7, natural order), canonical: leaves 0..3 are the transform of the fold.
    void fold(const Vec* leaves, u32* out) const {
        Vec f[4] = {leaves[0], leaves[1], leaves[2], leaves[3]};
        inverse4(f, 1, tables_.inverse_bottom(), Factor(ntt::detail::kRoots[0][0]));  // 4 times the fold
        const Factor quarter(ntt::detail::power(4, kP - 2));
        for (int t = 0; t < 4; ++t) store(out + 8 * t, reduce(multiply(f[t], quarter), kP));
    }

private:
    static constexpr std::size_t kTile = 256;  // blocks of at most kTile vectors go stage by stage

    // Blocks after the top step: radix-2 (odd stage count) or radix-4.
    static std::size_t top_block(std::size_t nv) { return std::countr_zero(nv) % 2 ? nv / 2 : nv / 4; }

    // [P, R] -> [P + R, P - R] for canonical P, R: both < 2P. A pass of its own: the top step on
    // both at once reads and writes eight streams, which is slower than two passes.
    static void signs(Vec* x, Vec* y) {
        const Vec p = *x, r = *y;
        *x = add(p, r);
        *y = _mm256_add_epi32(_mm256_sub_epi32(p, r), broadcast(kP));
    }

    // Radix-2 top step (H = 8h positions) on vectors j, j + h: the twiddle t = t_H (a table line),
    // times J (rot) in the upper half of j (t_H(i + H/2) = J t_H(i)).
    static void top2(Vec* f, std::size_t h, const u32* t, const Factor* rot = nullptr) {
        const Vec a = f[0], b = f[h], d = diff(a, b);
        f[0] = low(add(a, b));
        f[h] = shoup(rot ? multiply(d, *rot) : d, t);
    }

    // Inverse of top2 times 2 (inputs < 4P, outputs < 4P); inverse twiddles.
    static void inverse_top2(Vec* f, std::size_t h, const u32* t, const Factor* rot = nullptr) {
        const Vec a = low(f[0]), c = shoup(f[h], t), d = rot ? multiply(c, *rot) : c;
        f[0] = add(a, d), f[h] = diff(a, d);
    }

    // c+, c- < 4P -> canonical scale (c+ + c-), scale (c+ - c-).
    static void combine(Vec* x, Vec* y, const Factor& scale) {
        const Vec p = low(*x), m = low(*y);
        *x = reduce(multiply(add(p, m), scale), kP);
        *y = reduce(multiply(diff(p, m), scale), kP);
    }

    // Stages of a block of nb = 4^k vectors after the top step; leaves canonical.
    void block(Vec* x, Vec* y, std::size_t nb, const Factor& rot) const {
        if (nb > kTile) {
            step(x, y, nb / 4, rot);
            for (std::size_t o = 0; o < nb; o += nb / 4) block(x + o, y + o, nb / 4, rot);
            return;
        }
        for (std::size_t h = nb / 4; h > 1; h /= 4)
            for (std::size_t o = 0; o < nb; o += 4 * h) step(x + o, y + o, h, rot);
        const u32* t = tables_.pair(1);
        for (std::size_t o = 0; o < nb; o += 4) {
            forward4<true>(x + o, 1, t, rot);
            forward4<true>(y + o, 1, t, rot);
        }
    }

    // One radix-4 step on a block of 4h vectors, one sign after the other (both at once lose on
    // large blocks: twice the streams).
    void step(Vec* x, Vec* y, std::size_t h, const Factor& rot) const {
        const u32* t = tables_.pair(h);
        for (Vec* f : {x, y})
            for (std::size_t j = 0; j < h; ++j) forward4<false>(f + j, h, t + Tables::kEntry * j, rot);
    }

    void inverse_block(Vec* x, Vec* y, std::size_t nb, const Factor& rot) const {
        if (nb > kTile) {
            for (std::size_t o = 0; o < nb; o += nb / 4) inverse_block(x + o, y + o, nb / 4, rot);
            inverse_step(x, y, nb / 4, rot);
            return;
        }
        const u32* t = tables_.pair(1);
        for (std::size_t o = 0; o < nb; o += 4) {
            inverse4(x + o, 1, t, rot);
            inverse4(y + o, 1, t, rot);
        }
        for (std::size_t h = 4; h < nb; h *= 4)
            for (std::size_t o = 0; o < nb; o += 4 * h) inverse_step(x + o, y + o, h, rot);
    }

    void inverse_step(Vec* x, Vec* y, std::size_t h, const Factor& rot) const {
        const u32* t = tables_.pair(h);
        for (Vec* f : {x, y})
            for (std::size_t j = 0; j < h; ++j) inverse4(f + j, h, t + Tables::kEntry * j, rot);
    }

    const Tables& tables_;
};

// 64-bit lanes t < 2^63 -> t 2^-32 mod P in the high dword, below t / 2^32 + P.
Vec montgomery(Vec t) {
    const Vec m = _mm256_mul_epu32(t, broadcast(ntt::kernels::kNI));
    return _mm256_add_epi64(t, _mm256_mul_epu32(m, broadcast(kP)));
}

// 64-bit lanes t < 4P 2^32 -> t mod 2P 2^32 in [0, 2P 2^32), by the high dword alone.
Vec reduce_high(Vec t) {
    const Vec high_2p = _mm256_set1_epi64x(static_cast<long long>(2 * kP) << 32);
    return _mm256_min_epu32(t, _mm256_sub_epi32(t, high_2p));
}

// Leaf position of 5^k mod 32 (k < 8): position i holds u = 4i + 1. An involution.
constexpr int kLeafLog[8] = {0, 1, 6, 7, 4, 5, 2, 3};
Vec leaf_order(Vec x) { return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(0, 1, 6, 7, 4, 5, 2, 3)); }

// Leaf products. A leaf x of a, in log order, is held as a window [x, x]: z^j x mod (z^8 - 1),
// z = 5, is window [8 - j, 16 - j). A product with a canonical leaf y of b (natural order) goes
// into 64-bit sums, coefficients 2i in even and 2i + 1 in odd: window vector [k, k + 8) meets
// y_(8-k) in even and y_(9-k) in odd. A product adds under 8 P^2 < 2P 2^32; reduce_high keeps the
// sums below 2P 2^32 between products. Several products go in one loop, so that their dependency
// chains interleave (one alone is latency-bound).
template <int N>
struct Chains {
    const u32* window[N];
    const u32* leaf[N];
};

[[gnu::always_inline]] inline Vec product(Vec x, const u32* y, int j) {
    return _mm256_mul_epu32(x, broadcast(y[kLeafLog[j]]));
}

// Two products: windows[0] y[0] into (e0, o0), windows[1] y[1] into (e1, o1).
[[gnu::always_inline]] inline void multiply_add(const Chains<2>& c, Vec& e0, Vec& o0, Vec& e1, Vec& o1) {
#pragma GCC unroll 9
    for (int k = 9; k >= 1; --k) {
        const Vec x0 = load(c.window[0] + k), x1 = load(c.window[1] + k);
        if (k <= 8) {
            e0 = _mm256_add_epi64(e0, product(x0, c.leaf[0], 8 - k));
            e1 = _mm256_add_epi64(e1, product(x1, c.leaf[1], 8 - k));
        }
        if (k >= 2) {
            o0 = _mm256_add_epi64(o0, product(x0, c.leaf[0], 9 - k));
            o1 = _mm256_add_epi64(o1, product(x1, c.leaf[1], 9 - k));
        }
        asm("" : "+x"(e0), "+x"(o0), "+x"(e1), "+x"(o1));  // keeps GCC from spilling the sums
    }
    e0 = reduce_high(e0), o0 = reduce_high(o0), e1 = reduce_high(e1), o1 = reduce_high(o1);
}

// Four products from zero, window[i] y[i] into (e_i, o_i), not reduced: below 8 P^2.
[[gnu::always_inline]] inline void products(const Chains<4>& c, Vec (&e)[4], Vec (&o)[4]) {
    Vec e0 = _mm256_setzero_si256(), e1 = e0, e2 = e0, e3 = e0, o0 = e0, o1 = e0, o2 = e0, o3 = e0;
#pragma GCC unroll 9
    for (int k = 9; k >= 1; --k) {
        const Vec x0 = load(c.window[0] + k), x1 = load(c.window[1] + k), x2 = load(c.window[2] + k), x3 = load(c.window[3] + k);
        if (k <= 8) {
            e0 = _mm256_add_epi64(e0, product(x0, c.leaf[0], 8 - k));
            e1 = _mm256_add_epi64(e1, product(x1, c.leaf[1], 8 - k));
            e2 = _mm256_add_epi64(e2, product(x2, c.leaf[2], 8 - k));
            e3 = _mm256_add_epi64(e3, product(x3, c.leaf[3], 8 - k));
        }
        if (k >= 2) {
            o0 = _mm256_add_epi64(o0, product(x0, c.leaf[0], 9 - k));
            o1 = _mm256_add_epi64(o1, product(x1, c.leaf[1], 9 - k));
            o2 = _mm256_add_epi64(o2, product(x2, c.leaf[2], 9 - k));
            o3 = _mm256_add_epi64(o3, product(x3, c.leaf[3], 9 - k));
        }
        asm("" : "+x"(e0), "+x"(o0), "+x"(e1), "+x"(o1), "+x"(e2), "+x"(o2), "+x"(e3), "+x"(o3));
    }
    e[0] = e0, e[1] = e1, e[2] = e2, e[3] = e3, o[0] = o0, o[1] = o1, o[2] = o2, o[3] = o3;
}

// Four products: window[i / 2] y[i] into (e_i, o_i).
[[gnu::always_inline]] inline void multiply_add(const Chains<2>& c, const u32* const (&y)[4], Vec (&e)[4], Vec (&o)[4]) {
    Vec e0 = e[0], e1 = e[1], e2 = e[2], e3 = e[3], o0 = o[0], o1 = o[1], o2 = o[2], o3 = o[3];
#pragma GCC unroll 9
    for (int k = 9; k >= 1; --k) {
        const Vec x0 = load(c.window[0] + k), x1 = load(c.window[1] + k);
        if (k <= 8) {
            e0 = _mm256_add_epi64(e0, product(x0, y[0], 8 - k));
            e1 = _mm256_add_epi64(e1, product(x0, y[1], 8 - k));
            e2 = _mm256_add_epi64(e2, product(x1, y[2], 8 - k));
            e3 = _mm256_add_epi64(e3, product(x1, y[3], 8 - k));
        }
        if (k >= 2) {
            o0 = _mm256_add_epi64(o0, product(x0, y[0], 9 - k));
            o1 = _mm256_add_epi64(o1, product(x0, y[1], 9 - k));
            o2 = _mm256_add_epi64(o2, product(x1, y[2], 9 - k));
            o3 = _mm256_add_epi64(o3, product(x1, y[3], 9 - k));
        }
        asm("" : "+x"(e0), "+x"(o0), "+x"(e1), "+x"(o1), "+x"(e2), "+x"(o2), "+x"(e3), "+x"(o3));
    }
    e[0] = reduce_high(e0), e[1] = reduce_high(e1), e[2] = reduce_high(e2), e[3] = reduce_high(e3);
    o[0] = reduce_high(o0), o[1] = reduce_high(o1), o[2] = reduce_high(o2), o[3] = reduce_high(o3);
}

// Sums below 2P 2^32 (or 8 P^2) -> their value times 2^-32 mod P, < 2P, in natural order.
Vec finish(Vec even, Vec odd) {
    const Vec r = _mm256_blend_epi32(_mm256_srli_epi64(montgomery(even), 32), montgomery(odd), 0xAA);  // < 3P
    return leaf_order(reduce(r, 2 * kP));
}

// One level of one factor: sign parts of length L = 2^(M-2), M = N - s.
struct Level {
    std::size_t length = 0;   // L, 0 for M < 2
    u32 sum = 0;              // sum of the level's values mod P
    u32 fold[2][kFold] = {};  // [+], [-] folded to min(L, kFold) words, in log order
    u32* part[2] = {};        // L > kFold: P and R, then the leaves of [+] and [-]; for b, then c's
};

bool transformed(const Level& level) { return level.length > kFold; }

// Which levels go where. Input and output pass through chunks of 2^chunk_log values; per chunk,
// levels s < vector_levels are split off in vectors, and the values at multiples of
// 2^vector_levels go to a short array ("rest").
struct Plan {
    int n, chunk_log, vector_levels, levels;  // levels: transformed ones, s < levels

    explicit Plan(int log)
        : n(log),
          chunk_log(std::min(log, 16)),
          vector_levels(std::max(0, std::min(chunk_log - 4, log - kDirectLog))),
          levels(std::max(0, log - kDirectLog)) {}

    std::size_t chunk() const { return std::size_t(1) << chunk_log; }
    std::size_t rest() const { return std::size_t(1) << (n - vector_levels); }
};

// Work memory.
struct Buffers {
    u32* chunk;     // one chunk of values
    u32* rest[2];   // values at multiples of 2^s, s >= vector_levels
    char* text;     // a chunk as text

    Buffers(Arena& arena, const Plan& plan) {
        chunk = arena.words(plan.chunk());
        for (u32*& p : rest) p = arena.words(plan.rest());
        text = arena.bytes(std::max(fields::kTextBytes, 10 * plan.chunk()));
    }
};

// Odd and even words of x[0, 16): [x0 x2 .. x14], [x1 x3 .. x15].
void unzip(Vec v0, Vec v1, Vec& even, Vec& odd) {
    const __m256 a = _mm256_castsi256_ps(v0), b = _mm256_castsi256_ps(v1);
    even = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), 0xD8);
    odd = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8);
}

// Interleaved words: [even0 odd0 even1 odd1 ...] as two vectors.
void zip(Vec even, Vec odd, Vec& v0, Vec& v1) {
    const Vec lo = _mm256_unpacklo_epi32(even, odd), hi = _mm256_unpackhi_epi32(even, odd);
    v0 = _mm256_permute2x128_si256(lo, hi, 0x20);
    v1 = _mm256_permute2x128_si256(lo, hi, 0x31);
}

Vec reversed(Vec x) { return _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)); }

// x[0, 32) -> p = x[4i + 1], r = x[4i + 3] in reverse order (r[7 - i]), even = x[2i] (16 words).
void deinterleave(const u32* x, u32* p, u32* r, u32* even) {
    Vec e0, o0, e1, o1;
    unzip(load(x), load(x + 8), e0, o0);
    unzip(load(x + 16), load(x + 24), e1, o1);
    const __m256 a = _mm256_castsi256_ps(o0), b = _mm256_castsi256_ps(o1);
    const Vec plus = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), 0xD8);
    const Vec minus = _mm256_permutevar8x32_epi32(_mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD)),
                                                  _mm256_setr_epi32(7, 6, 3, 2, 5, 4, 1, 0));
    store(even, e0), store(even + 8, e1);
    store(p, plus), store(r, minus);
}

// Inverse of deinterleave.
void interleave(const u32* p, const u32* r, const u32* even, u32* x) {
    Vec o0, o1, v0, v1;
    zip(load(p), reversed(load(r)), o0, o1);
    const Vec e0 = load(even), e1 = load(even + 8);
    zip(e0, o0, v0, v1);
    store(x, v0), store(x + 8, v1);
    zip(e1, o1, v0, v1);
    store(x + 16, v0), store(x + 24, v1);
}

// Sets up a transformed level (M = m).
void open_level(Level& level, int m, Arena& arena) {
    const std::size_t len = std::size_t(1) << (m - 2);
    level.length = len;
    for (u32*& p : level.part) p = arena.words(len);
}

// Sum of a level whose folds are written.
void close_level(Level& level) {
    u64 t = 0;
    for (std::size_t i = 0; i < std::min(level.length, kFold); ++i) t += level.fold[0][i];
    level.sum = u32(t % kP);
}

// Reads 2^n values and splits them into their levels: transformed levels get P and R, the others
// their folds and sums.
void split(io::Reader& in, const Plan& plan, Level* levels, Arena& arena, const Buffers& buffers) {
    const int n = plan.n, vs = plan.vector_levels;
    for (int s = 0; s < plan.levels; ++s) open_level(levels[s], n - s, arena);
    u32* cur = buffers.rest[0];  // cur[i] = x[i 2^s]
    for (std::size_t first = 0; first < std::size_t(1) << n; first += plan.chunk()) {
        u32* x = buffers.chunk;
        io::read_bulk(in, x, plan.chunk());
        for (int s = 0; s < vs; ++s) {  // x[i] = value at first + i 2^s
            Level& level = levels[s];
            const std::size_t at = first >> (s + 2), groups = plan.chunk() >> (s + 5);
            for (std::size_t q = 0; q < groups; ++q)
                deinterleave(x + 32 * q, level.part[0] + at + 8 * q, level.part[1] + level.length - at - 8 * (q + 1), x + 16 * q);
        }
        std::copy(x, x + (plan.chunk() >> vs), cur + (first >> vs));
    }
    for (int s = vs; s <= n; ++s) {
        const int m = n - s;
        Level& level = levels[s];
        if (m < 2) {
            level.sum = cur[m];
            continue;  // level s + 1 is cur[0]
        }
        const u32 mask = (u32(1) << m) - 1;
        if (transformed(level)) {
            for (std::size_t i = 0; i < level.length; ++i) {
                level.part[0][i] = cur[4 * i + 1];
                level.part[1][i] = cur[mask - 4 * i];
            }
        } else {
            u32 u = 1;
            for (std::size_t k = 0; k < std::size_t(1) << (m - 2); ++k, u = u * 5 & mask) {
                level.fold[0][k] = add_mod(cur[u], cur[mask + 1 - u]);
                level.fold[1][k] = sub_mod(cur[u], cur[mask + 1 - u]);
            }
            level.length = std::size_t(1) << (m - 2);
            close_level(level);
        }
        u32* other = cur == buffers.rest[0] ? buffers.rest[1] : buffers.rest[0];
        for (std::size_t i = 0; i < std::size_t(1) << (m - 1); ++i) other[i] = cur[2 * i];
        cur = other;
    }
}

// Natural-order folds (u mod 2^7, position i: u = 4i + 1) -> log order: k at 5^k.
void to_log_order(const u32* natural, u32* log) {
    u32 u = 1;
    for (std::size_t k = 0; k < kFold; ++k, u = u * 5 & 127) log[k] = natural[(u - 1) / 4];
}

// Writes c from its levels, top down: after the level of the units mod 2^m, c[i 2^(n-m)] for
// i < 2^m are known. Up to m = n - vector_levels in the rest buffers; then by chunks.
class Assembly {
public:
    Assembly(const Plan& plan, const Buffers& buffers, io::Writer& out, u32 zero)
        : plan_(plan), buffers_(buffers), out_(out), cur_(buffers.rest[0]), other_(buffers.rest[1]) {
        cur_[0] = zero;
        if (plan.n == 0) fields::write(out_, cur_, 1, buffers_.text);
    }

    // m = 1: the unit 1.
    void push_odd(u32 value) {
        cur_[1] = value;
        if (plan_.n == 1) fields::write(out_, cur_, 2, buffers_.text);
    }

    // m >= 2, log order: the values at +-5^k are plus[k] +- minus[k], k < 2^(m-2); canonical.
    void push(int m, const u32* plus, const u32* minus) {
        spread(m);
        const std::size_t len = std::size_t(1) << (m - 2);
        const u32 mask = (u32(1) << m) - 1;
        u32 u = 1;
        for (std::size_t k = 0; k < len; ++k, u = u * 5 & mask) {
            other_[u] = add_mod(plus[k], minus[k]);
            other_[mask + 1 - u] = sub_mod(plus[k], minus[k]);
        }
        done(m);
    }

    // m >= 2, natural order: p[i] at 4i + 1, r[i] at -(4i + 1); canonical.
    void push_natural(int m, const u32* p, const u32* r) {
        spread(m);
        const u32 mask = (u32(1) << m) - 1;
        for (std::size_t i = 0; i < std::size_t(1) << (m - 2); ++i) {
            other_[4 * i + 1] = p[i];
            other_[mask - 4 * i] = r[i];
        }
        done(m);
    }

    // The levels m > n - vector_levels (P and R of level s = n - m in levels[s].part), by chunks.
    void finish(const Level* levels) {
        const int vs = plan_.vector_levels;
        if (vs == 0) return;
        const std::size_t chunk = plan_.chunk(), size = std::size_t(1) << plan_.n;
        for (std::size_t first = 0; first < size; first += chunk) {
            u32* x = buffers_.chunk;
            std::copy(cur_ + (first >> vs), cur_ + ((first + chunk) >> vs), x);
            for (int s = vs - 1; s >= 0; --s) {
                const Level& level = levels[s];
                const std::size_t at = first >> (s + 2), groups = chunk >> (s + 5);
                for (std::size_t q = groups; q-- > 0;)
                    interleave(level.part[0] + at + 8 * q, level.part[1] + level.length - at - 8 * (q + 1), x + 16 * q, x + 32 * q);
            }
            fields::detail::format(buffers_.text, x, chunk, fields::detail::kConstants);
            if (first + chunk == size) buffers_.text[10 * chunk - 1] = '\n';
            out_.write(std::string_view(buffers_.text, 10 * chunk));
        }
    }

private:
    void spread(int m) {
        for (std::size_t i = 0; i < std::size_t(1) << (m - 1); ++i) other_[2 * i] = cur_[i];
    }

    void done(int m) {
        if (m == plan_.n) fields::write(out_, other_, std::size_t(1) << m, buffers_.text);
        std::swap(cur_, other_);
    }

    const Plan& plan_;
    const Buffers& buffers_;
    io::Writer& out_;
    u32 *cur_, *other_;
};

// Output level m <= kDirectLog: cyclic convolutions of the folds.
void direct_level(Assembly& c, int n, int m, const Level* a, const Level* b) {
    const std::size_t len = std::size_t(1) << (m - 2);
    u32 result[2][kFold];
    for (int e = 0; e < 2; ++e) {
        u64 acc[kFold] = {};
        for (int s = 0; s <= n - m; ++s) {
            const Level &x = a[s], &y = b[n - m - s];
            u32 fx[kFold] = {}, fy[kFold] = {};
            for (std::size_t k = 0; k < std::min(x.length, kFold); ++k) fx[k % len] = add_mod(fx[k % len], x.fold[e][k]);
            for (std::size_t k = 0; k < std::min(y.length, kFold); ++k) fy[k % len] = add_mod(fy[k % len], y.fold[e][k]);
            for (std::size_t i = 0; i < len; ++i)
                for (std::size_t k = 0; k < len; ++k) acc[(i + k) % len] = (acc[(i + k) % len] + u64(fx[i]) * fy[k]) % kP;
        }
        for (std::size_t k = 0; k < len; ++k) result[e][k] = u32(acc[k]);
    }
    for (auto& r : result)
        for (std::size_t k = 0; k < len; ++k) r[k] = mul_mod(r[k], kHalf);
    c.push(m, result[0], result[1]);
}

// [x, x] in log order for leaf v of the levels s <= k of a (sign e).
using Windows = std::array<std::array<u32, 16>, kMaxLog + 1>;

void make_windows(const Level* a, int e, int k, std::size_t v, Windows& windows) {
    for (int s = 0; s <= k; ++s) {
        const Vec x = leaf_order(load(a[s].part[e] + 8 * v));
        store(windows[s].data(), x);
        store(windows[s].data() + 8, x);
    }
}

// Band 0: level 0 alone, one product per leaf; four leaves at a time.
void band_zero(const Level* a, Level* b, int e, std::size_t first, std::size_t end) {
    alignas(64) std::array<u32, 16> windows[8 + 1];  // [x, x] of leaf v at v % 8; loads read a word past
    const auto make_window = [&](std::size_t v) {
        const Vec x = leaf_order(load(a[0].part[e] + 8 * v));
        store(windows[v % 8].data(), x);
        store(windows[v % 8].data() + 8, x);
    };
    for (std::size_t v = first; v < first + 4; ++v) make_window(v);
    for (std::size_t v = first; v < end; v += 4) {  // end - first: a multiple of 4
        if (v + 4 < end)
            for (std::size_t u = v + 4; u < v + 8; ++u) make_window(u);
        u32* const c = b[0].part[e] + 8 * v;
        Vec even[4], odd[4];
        const Chains<4> chains = {{windows[v % 8].data(), windows[(v + 1) % 8].data(), windows[(v + 2) % 8].data(),
                                   windows[(v + 3) % 8].data()},
                                  {c, c + 8, c + 16, c + 24}};
        products(chains, even, odd);
        for (int i = 0; i < 4; ++i) store(c + 8 * i, finish(even[i], odd[i]));
    }
}

// Leaves [first, end) of the levels s <= k, sign e: level t of c gets the sum over s + t' = t of
// a_s b_t', in place of b_t. Leaves go in pairs, levels t and t - 1 together: up to eight
// independent sums. The windows of the next pair are stored before the products of this one load
// theirs: unaligned loads from recent stores stall.
void band_products(const Level* a, Level* b, int e, int k, std::size_t first, std::size_t end) {
    if (k == 0) return band_zero(a, b, e, first, end);
    alignas(64) Windows windows[4];  // leaf v at v % 4
    make_windows(a, e, k, first, windows[first % 4]);
    make_windows(a, e, k, first + 1, windows[(first + 1) % 4]);
    for (std::size_t v = first; v < end; v += 2) {
        if (v + 2 < end) {
            make_windows(a, e, k, v + 2, windows[(v + 2) % 4]);
            make_windows(a, e, k, v + 3, windows[(v + 3) % 4]);
        }
        const Windows &x = windows[v % 4], &w = windows[(v + 1) % 4];
        const auto leaf = [&](int t, std::size_t u) { return b[t].part[e] + 8 * u; };
        for (int t = k; t >= 0; t -= 2) {  // levels t and t - 1; b_t is last read for level t
            Vec even[4] = {}, odd[4] = {};  // level t of v, t - 1 of v, t of v + 1, t - 1 of v + 1
            for (int s = 0; s < t; ++s) {
                const u32* const y[4] = {leaf(t - s, v), leaf(t - 1 - s, v), leaf(t - s, v + 1), leaf(t - 1 - s, v + 1)};
                multiply_add({{x[s].data(), w[s].data()}, {}}, y, even, odd);
            }
            multiply_add({{x[t].data(), w[t].data()}, {leaf(0, v), leaf(0, v + 1)}}, even[0], odd[0], even[2], odd[2]);
            store(leaf(t, v), finish(even[0], odd[0]));
            store(leaf(t, v + 1), finish(even[2], odd[2]));
            if (t > 0) {
                store(leaf(t - 1, v), finish(even[1], odd[1]));
                store(leaf(t - 1, v + 1), finish(even[3], odd[3]));
            }
        }
    }
}

// Marks a mapped input as read once: the kernel then skips marking each page accessed when the
// mapping goes (as ../gcd_convolution). The mapping starts at the page of the first token.
void advise_sequential(const io::Reader& in) {
    struct stat st;
    if (::fstat(0, &st) != 0 || !S_ISREG(st.st_mode) || std::size_t(st.st_size) <= io::detail::kMapAbove) return;
    const auto start = reinterpret_cast<std::uintptr_t>(in.scan().cur) & ~std::uintptr_t(4095);
    ::madvise(reinterpret_cast<void*>(start), std::size_t(st.st_size), MADV_SEQUENTIAL);
}

void solve() {
    io::Reader in;
    const int n = in.read<int>();
    advise_sequential(in);
    const std::size_t size = std::size_t(1) << n;
    const Plan plan(n);
    // Two factors of 2^n words, tables under 2^n words, the chunk text and 64 KiB of padding.
    Arena arena(4 * 4 * size + 10 * plan.chunk() + fields::kTextBytes + (64 << 10));
    const Buffers buffers(arena, plan);
    Tables tables(arena, n);
    const Transform transform(tables);

    std::array<Level, kMaxLog + 1> a, b;
    for (auto* x : {&a, &b}) {
        split(in, plan, x->data(), arena, buffers);
        for (int s = 0; s < plan.levels; ++s) {
            Level& level = (*x)[s];
            auto* p = reinterpret_cast<Vec*>(level.part[0]);
            auto* r = reinterpret_cast<Vec*>(level.part[1]);
            transform.forward(p, r, level.length / 8);
            for (int e = 0; e < 2; ++e) {
                alignas(32) u32 natural[kFold];
                transform.fold(reinterpret_cast<const Vec*>(level.part[e]), natural);
                to_log_order(natural, level.fold[e]);
            }
            close_level(level);
        }
    }

    io::Writer out;
    u32 zero = 0;
    for (int s = 0; s <= n; ++s)
        for (int t = n - s; t <= n; ++t) zero = add_mod(zero, mul_mod(a[s].sum, b[t].sum));
    Assembly c(plan, buffers, out, zero);
    if (n >= 1) {
        u32 odd = 0;
        for (int s = 0; s <= n - 1; ++s) odd = add_mod(odd, mul_mod(a[s].sum, b[n - 1 - s].sum));
        c.push_odd(odd);
    }
    for (int m = 2; m <= std::min(n, kDirectLog); ++m) direct_level(c, n, m, a.data(), b.data());
    if (n <= kDirectLog) return;

    const int levels = plan.levels;
    for (int e = 0; e < 2; ++e)  // leaves [nv of level k + 1, nv of level k) are in levels 0..k
        for (int k = levels - 1; k >= 0; --k)
            band_products(a.data(), b.data(), e, k, k + 1 < levels ? a[k + 1].length / 8 : 0, a[k].length / 8);
    tables.invert();
    for (int m = kDirectLog + 1; m <= n; ++m) {
        Level& level = b[n - m];
        const std::size_t nv = level.length / 8;
        const u32 scale = mul_mod(mul_mod(ntt::detail::power(u32(nv), kP - 2), ntt::detail::kR), kHalf);
        transform.inverse(reinterpret_cast<Vec*>(level.part[0]), reinterpret_cast<Vec*>(level.part[1]), nv, scale);
        if (m <= n - plan.vector_levels) c.push_natural(m, level.part[0], level.part[1]);
    }
    c.finish(b.data());
}

}  // namespace

RUN_EARLY(solve)
