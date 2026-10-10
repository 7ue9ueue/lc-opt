// c_k = sum over i j = k (mod 2^N) of a_i b_j mod 998244353, N <= 20.
//
// Index i = 2^s u with u odd is in level s. The odd residues mod 2^M (M = N - s >= 2) are
// +-5^k, k < L = 2^(M-2), so level s is a function on {+-1} x Z/L; the sign transform
// (x+ = x(1) + x(-1), x- = x(1) - x(-1)) splits it into two cyclic sequences. A product of levels
// s and t lands in level s + t, as the product of the units mod 2^m, m = N - s - t: both factors
// folded to length 2^(m-2), one cyclic convolution per sign. Products with s + t >= N land on 0.
//
// Cyclic transforms use lib/ntt's tree (leaves x^8 - w). The first 2^(m-2) words of the transform
// of length 2^(M-2) are the transform of the input folded to length 2^(m-2), so each level of a and
// b is transformed once (2^N words per factor in all), and output level m sums the leaf products
// of its pairs and runs one inverse transform. Levels m <= 7 are convolved directly.
#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

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
using ntt::detail::slot;
using ntt::detail::Vec;

constexpr int kMaxLog = 20;
constexpr int kDirectLog = 7;                                // output levels m <= 7: direct
constexpr std::size_t kFold = std::size_t(1) << (kDirectLog - 2);  // their longest sequence
constexpr std::size_t kPadding = 16;                         // words after each array: kernels read past

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

// x < 4P -> x mod P.
Vec canonical(Vec x) { return reduce(reduce(x, 2 * kP), kP); }

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

// Forward and inverse transforms of lib/ntt's tree with the leaves exposed. A transform of nv
// vectors (nv >= 8) splits x^(8 nv) - 1 into leaves x^8 - w; leaf v has the weight
// w = +-r[2k] (v = 4k, 4k + 1) or +-r[2k + 1] (v = 4k + 2, 4k + 3), the same at every length.
class Transform {
public:
    // Tables for transforms of up to max_nv vectors.
    Transform(Arena& arena, std::size_t max_nv) {
        const int lg = std::max(6, std::countr_zero(max_nv) + 3);
        roots_ = arena.words(ntt::detail::table_words(lg));
        inverse_roots_ = arena.words(ntt::detail::table_words(lg));
        ntt::detail::build_table(roots_, (std::size_t(1) << lg) / 16, ntt::detail::kRoots[0]);
        ntt::detail::build_table(inverse_roots_, (std::size_t(1) << lg) / 16, ntt::detail::kRoots[1]);
    }

    // Canonical f -> its leaves, canonical. top_done: f is canonical after the top layer
    // (radix 2 if nv is 2 4^j, else radix 4; see top_layer).
    void forward(Vec* f, std::size_t nv, bool top_done) const {
        if (std::countr_zero(nv) % 2) {
            const std::size_t h = nv / 2;
            if (!top_done) ntt::detail::forward_radix2(f, h, false);
            visit(f, h, 0);
            visit(f + h, h, 1);
        } else if (top_done) {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) visit(f + t * h, h, t);
        } else {
            visit(f, nv, 0);
        }
    }

    // sqrt(-1) of the radix-4 top layer.
    Factor top_root() const { return Factor(roots_[1], roots_[9]); }

    // Canonical leaves of a transform of nv >= 4 vectors -> its input folded to 32 words,
    // canonical: the first four leaves are the transform of the fold.
    void fold(const Vec* leaves, u32* out) const {
        Vec f[4] = {leaves[0], leaves[1], leaves[2], leaves[3]};
        last_inverse(f, 0);  // 4 times the fold
        const Factor quarter(ntt::detail::power(4, kP - 2));
        for (int t = 0; t < 4; ++t) _mm256_storeu_si256(reinterpret_cast<Vec*>(out + 8 * t), reduce(multiply(f[t], quarter), kP));
    }

    // w of leaf v as a factor.
    Factor leaf_weight(std::size_t v) const {
        const u32* y = roots_ + slot(v / 4 * 2) + v / 2 % 2;
        return v % 2 ? Factor(kP - y[0], ~y[8]) : Factor(y[0], y[8]);  // q(P - w) = ~q(w)
    }

    // Leaves < 2P -> scale times the inverse transform, canonical. The inverse alone multiplies
    // by nv.
    void inverse(Vec* f, std::size_t nv, u32 scale) const {
        if (std::countr_zero(nv) % 2) {
            const std::size_t h = nv / 2;
            inverse_visit(f, h, 0);
            inverse_visit(f + h, h, 1);
            alignas(32) u32 s[16] = {};  // table layout: s at entry 1
            s[1] = scale;
            s[9] = ntt::detail::quotient(scale);
            ntt::kernels::scale_radix2(f, h, s);
        } else {
            const std::size_t h = nv / 4;
            for (std::size_t t = 0; t < 4; ++t) inverse_visit(f + t * h, h, t);
            ntt::detail::inverse_radix4(f, h, inverse_roots_, Factor(scale));
        }
    }

private:
    // Group k at stride h = nv / 4, then its children 4k + t.
    void visit(Vec* f, std::size_t nv, std::size_t k) const {
        if (nv == 4) return last_forward(f, k);
        const std::size_t h = nv / 4;
        if (k == 0)
            ntt::kernels::forward_identity(f, h, roots_);
        else
            ntt::kernels::forward(f, h, roots_ + slot(k), roots_ + slot(2 * k));
        for (std::size_t t = 0; t < 4; ++t) visit(f + t * h, h, 4 * k + t);
    }

    void inverse_visit(Vec* f, std::size_t nv, std::size_t k) const {
        if (nv == 4) return last_inverse(f, k);
        const std::size_t h = nv / 4;
        for (std::size_t t = 0; t < 4; ++t) inverse_visit(f + t * h, h, 4 * k + t);
        if (k == 0)
            ntt::kernels::inverse_identity(f, h, inverse_roots_);
        else
            ntt::kernels::inverse(f, h, inverse_roots_ + slot(k), inverse_roots_ + slot(2 * k));
    }

    // Group k with h = 1 (vectors 4k..4k + 3), inputs < 4P: the kernels need h even.
    void last_forward(Vec* f, std::size_t k) const {
        const u32 *x = roots_ + slot(k), *y = roots_ + slot(2 * k);
        const Factor fx(x[0], x[8]), fy(y[0], y[8]), fz(y[1], y[9]);
        const Vec f0 = reduce(f[0], 2 * kP), f1 = reduce(f[1], 2 * kP);
        const Vec xf2 = multiply(f[2], fx), xf3 = multiply(f[3], fx);  // < 2P
        const Vec a = reduce(add(f0, xf2), 2 * kP), c = reduce(diff(f0, xf2), 2 * kP);
        const Vec yb = multiply(add(f1, xf3), fy), zd = multiply(diff(f1, xf3), fz);
        f[0] = canonical(add(a, yb));
        f[1] = canonical(diff(a, yb));
        f[2] = canonical(add(c, zd));
        f[3] = canonical(diff(c, zd));
    }

    // Inverse of group k with h = 1, inputs and outputs < 2P.
    void last_inverse(Vec* f, std::size_t k) const {
        const u32 *x = inverse_roots_ + slot(k), *y = inverse_roots_ + slot(2 * k);
        const Factor fx(x[0], x[8]), fy(y[0], y[8]), fz(y[1], y[9]);
        const Vec ab = reduce(add(f[0], f[1]), 2 * kP), cd = reduce(add(f[2], f[3]), 2 * kP);
        const Vec amb = multiply(diff(f[0], f[1]), fy), cmd = multiply(diff(f[2], f[3]), fz);
        f[0] = reduce(add(ab, cd), 2 * kP);
        f[1] = reduce(add(amb, cmd), 2 * kP);
        f[2] = multiply(diff(ab, cd), fx);
        f[3] = multiply(diff(amb, cmd), fx);
    }

    u32 *roots_, *inverse_roots_;
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

// Sums of 64-bit products of x y mod (z^8 - w): coefficients 2i in even, 2i + 1 in odd. Both
// below 2P 2^32 on entry and exit: a product adds under 8 P^2 < 2P 2^32.
// window = [w x, x], y: canonical leaves.
void multiply_add(const u32* window, const u32* y, Vec& even, Vec& odd) {  // z^j x = window [8 - j, 16 - j)
#pragma GCC unroll 8
    for (int j = 0; j < 8; ++j) {
        const Vec yj = broadcast(y[j]);
        even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_loadu_si256(reinterpret_cast<const Vec*>(window + 8 - j)), yj));
        odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_loadu_si256(reinterpret_cast<const Vec*>(window + 9 - j)), yj));
    }
    even = reduce_high(even);
    odd = reduce_high(odd);
}

// Sums of multiply_add -> their value times 2^-32 mod P, < 2P.
Vec finish(Vec even, Vec odd) {
    const Vec r = _mm256_blend_epi32(_mm256_srli_epi64(montgomery(even), 32), montgomery(odd), 0xAA);  // < 3P
    return reduce(r, 2 * kP);
}

// One level of one factor: sign parts of length L = 2^(M-2), M = N - s.
struct Level {
    std::size_t length = 0;     // L, 0 for M < 2
    u32 sum = 0;                // sum of the level's values mod P
    u32 fold[2][kFold] = {};    // [+], [-] folded to min(L, kFold) words
    u32* part[2] = {};          // L > kFold: [+], [-], then their leaves; for b, then c's leaves
};

bool transformed(const Level& level) { return level.length > kFold; }

// Level s of x (2^n values) by sign and discrete log.
// For n >= kBlockedLog, x is held transposed: grid[c][h] = x[c + 2^B h], c < 2^B, h < kHeight,
// B = n - kRowLog, rows kPitch words apart. Level s (M = n - s >= kBlockedLog, b = M - kRowLog)
// has y[u] = x[2^s u] in row 2^s (u mod 2^b), column u >> b. With k = k_lo + 2^(b-2) k_hi,
// 5^k = 5^k_lo G^k_hi, G = 5^(2^(b-2)) = 1 mod 2^b: the row of +-5^k depends on k_lo alone, so
// eight consecutive k_lo use 16 rows (17 KiB) for all k_hi.
constexpr int kBlockedLog = 13;
constexpr int kRowLog = 8;
constexpr std::size_t kHeight = std::size_t(1) << kRowLog;
constexpr std::size_t kPitch = kHeight + 16;  // rows 2^s apart start in different L1 sets

Vec load(const u32* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
void store(u32* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }

Vec add_canonical(Vec x, Vec y) {
    const Vec s = _mm256_add_epi32(x, y);
    return _mm256_min_epu32(s, _mm256_sub_epi32(s, broadcast(kP)));
}
Vec sub_canonical(Vec x, Vec y) {
    const Vec d = _mm256_sub_epi32(x, y);
    return _mm256_min_epu32(d, _mm256_add_epi32(d, broadcast(kP)));
}

void transpose(Vec (&r)[8]) {
    Vec t[8], u[8];
    for (int i = 0; i < 8; i += 2) {
        t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
    }
    for (int i = 0; i < 8; i += 4) {
        u[i] = _mm256_unpacklo_epi64(t[i], t[i + 2]);
        u[i + 1] = _mm256_unpackhi_epi64(t[i], t[i + 2]);
        u[i + 2] = _mm256_unpacklo_epi64(t[i + 1], t[i + 3]);
        u[i + 3] = _mm256_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int i = 0; i < 4; ++i) {
        r[i] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x20);
        r[i + 4] = _mm256_permute2x128_si256(u[i], u[i + 4], 0x31);
    }
}

constexpr std::size_t kBand = 16;  // rows of x per pass: one cache line of each grid row

// Rows [first, first + kBand) of x (band, width = 2^B words each) -> their grid columns.
void to_grid(const u32* band, std::size_t width, std::size_t first, u32* grid) {
    for (std::size_t c = 0; c < width; c += 8)
        for (std::size_t h = 0; h < kBand; h += 8) {
            Vec r[8];
            for (std::size_t i = 0; i < 8; ++i) r[i] = load(band + (h + i) * width + c);
            transpose(r);
            for (std::size_t i = 0; i < 8; ++i) store(grid + (c + i) * kPitch + first + h, r[i]);
        }
}

// Inverse of to_grid.
void from_grid(const u32* grid, std::size_t width, std::size_t first, u32* band) {
    for (std::size_t c = 0; c < width; c += 8)
        for (std::size_t h = 0; h < kBand; h += 8) {
            Vec r[8];
            for (std::size_t i = 0; i < 8; ++i) r[i] = load(grid + (c + i) * kPitch + first + h);
            transpose(r);
            for (std::size_t i = 0; i < 8; ++i) store(band + (h + i) * width + c, r[i]);
        }
}

// Grid offsets of +-5^k (level s, M = m) for eight consecutive k_lo, stepping k_hi.
class UnitWalk {
public:
    UnitWalk(int m, int s, std::size_t k_lo) : b_(m - kRowLog), mask_(broadcast((u32(1) << m) - 1)) {
        const u32 mask = (u32(1) << m) - 1;
        u32 g = 5, p = 1;
        for (int i = 0; i < b_ - 2; ++i) g = g * g & mask;
        for (std::size_t i = 0; i < k_lo; ++i) p = p * 5 & mask;
        alignas(32) u32 lanes[8];
        for (u32& lane : lanes) lane = p, p = p * 5 & mask;
        unit_ = _mm256_load_si256(reinterpret_cast<const Vec*>(lanes));
        step_ = broadcast(g);
        const Vec low = _mm256_and_si256(unit_, broadcast((u32(1) << b_) - 1));  // odd: -u has 2^b - low
        row_pos_ = _mm256_mullo_epi32(_mm256_slli_epi32(low, s), broadcast(kPitch));
        row_neg_ = _mm256_mullo_epi32(_mm256_slli_epi32(_mm256_sub_epi32(broadcast(u32(1) << b_), low), s), broadcast(kPitch));
        row_neg_ = _mm256_add_epi32(row_neg_, broadcast(kHeight - 1));
    }

    // 5^k for the current k_hi; c times it, for c = 1 mod 2^b; the next k_hi.
    Vec unit() const { return unit_; }
    Vec times(u32 c) const { return _mm256_and_si256(_mm256_mullo_epi32(unit_, broadcast(c)), mask_); }
    void next() { unit_ = _mm256_and_si256(_mm256_mullo_epi32(unit_, step_), mask_); }

    // G^e mod 2^m, G = 5^(2^(b-2)) the step.
    u32 step_power(std::size_t e) const {
        const u32 mask = u32(_mm256_cvtsi256_si32(mask_)), g = u32(_mm256_cvtsi256_si32(step_));
        u32 r = 1;
        for (std::size_t i = 0; i < e; ++i) r = r * g & mask;
        return r;
    }

    // Offsets of u and -u, for u = 5^k times a power of the step.
    Vec positive(Vec u) const { return _mm256_add_epi32(row_pos_, _mm256_srli_epi32(u, b_)); }
    Vec negative(Vec u) const { return _mm256_sub_epi32(row_neg_, _mm256_srli_epi32(u, b_)); }

private:
    int b_;
    Vec mask_, unit_, step_, row_pos_, row_neg_;
};

// The top layer of Transform::forward on Ways canonical vectors at stride nv / Ways (Ways = 2:
// x^(8 nv) - 1 = (x^(4 nv) - 1)(x^(4 nv) + 1); Ways = 4: group 0 of the radix-4 tree), canonical.
template <int Ways>
void top_layer(Vec (&f)[Ways], const Factor& z) {
    if constexpr (Ways == 2) {
        const Vec x = f[0], y = f[1];
        f[0] = add_canonical(x, y);
        f[1] = sub_canonical(x, y);
    } else {
        const Vec s02 = add_canonical(f[0], f[2]), d02 = sub_canonical(f[0], f[2]);
        const Vec s13 = add_canonical(f[1], f[3]), zd13 = reduce(multiply(sub_canonical(f[1], f[3]), z), kP);
        f[0] = add_canonical(s02, s13);
        f[1] = sub_canonical(s02, s13);
        f[2] = add_canonical(d02, zd13);
        f[3] = sub_canonical(d02, zd13);
    }
}

// Level s (M = m) of the grid -> plus[k], minus[k] = y(5^k) +- y(-5^k), k < 2^(m-2), after
// top_layer<Ways>. k = k_lo + stride k_hi: the top layer's vectors are k_hi + i kHeight / Ways.
template <int Ways>
void gather_units(const u32* grid, int m, int s, const Factor& z, u32* plus, u32* minus) {
    const std::size_t stride = std::size_t(1) << (m - kRowLog - 2), rounds = kHeight / Ways;
    const auto* base = reinterpret_cast<const int*>(grid);
    for (std::size_t k = 0; k < stride; k += 8) {
        UnitWalk walk(m, s, k);
        u32 jump[Ways];
        for (int i = 0; i < Ways; ++i) jump[i] = walk.step_power(i * rounds);
        for (std::size_t j = k; j < k + rounds * stride; j += stride, walk.next()) {
            Vec p[Ways], q[Ways];
            for (int i = 0; i < Ways; ++i) {
                const Vec u = i ? walk.times(jump[i]) : walk.unit();
                const Vec xp = _mm256_i32gather_epi32(base, walk.positive(u), 4);
                const Vec xn = _mm256_i32gather_epi32(base, walk.negative(u), 4);
                p[i] = add_canonical(xp, xn);
                q[i] = sub_canonical(xp, xn);
            }
            top_layer(p, z);
            top_layer(q, z);
            for (int i = 0; i < Ways; ++i) {
                store(plus + j + i * rounds * stride, p[i]);
                store(minus + j + i * rounds * stride, q[i]);
            }
        }
    }
}

// Values plus[k] +- minus[k] at +-5^k, k < 2^(m-2) (canonical) -> level s (M = m) of the grid.
void scatter_units(const u32* plus, const u32* minus, int m, int s, u32* grid) {
    const std::size_t stride = std::size_t(1) << (m - kRowLog - 2);
    for (std::size_t k = 0; k < stride; k += 8) {
        UnitWalk walk(m, s, k);
        for (std::size_t j = k; j < k + kHeight * stride; j += stride, walk.next()) {
            alignas(32) u32 at_pos[8], at_neg[8], pos[8], neg[8];
            _mm256_store_si256(reinterpret_cast<Vec*>(at_pos), walk.positive(walk.unit()));
            _mm256_store_si256(reinterpret_cast<Vec*>(at_neg), walk.negative(walk.unit()));
            const Vec p = load(plus + j), q = load(minus + j);
            _mm256_store_si256(reinterpret_cast<Vec*>(pos), add_canonical(p, q));
            _mm256_store_si256(reinterpret_cast<Vec*>(neg), sub_canonical(p, q));
            for (std::size_t l = 0; l < 8; ++l) grid[at_pos[l]] = pos[l], grid[at_neg[l]] = neg[l];
        }
    }
}

// Work memory, in words.
struct Buffers {
    u32* band;      // n >= kBlockedLog: kBand rows of x, 2^(n-4); else x, 2^n
    u32* grid;      // n >= kBlockedLog: 2^(n-kRowLog) rows
    u32* small[2];  // levels with M < kBlockedLog: 2^min(n, kBlockedLog - 1) each
    char* text;     // 10 words of band, at least fields::kTextBytes

    Buffers(Arena& arena, int n) {
        const bool blocked = n >= kBlockedLog;
        band = arena.words(std::size_t(1) << (blocked ? n - 4 : n));
        grid = blocked ? arena.words((std::size_t(1) << (n - kRowLog)) * kPitch) : nullptr;
        for (u32*& p : small) p = arena.words(std::size_t(1) << std::min(n, kBlockedLog - 1));
        text = arena.bytes(std::max(fields::kTextBytes, blocked ? 10 * (std::size_t(1) << (n - 4)) : 0));
    }
};

// Levels with m >= kBlockedLog.
bool top_done(int m) { return m >= kBlockedLog; }

void gather(const u32* grid, int m, int s, const Factor& z, std::array<u32*, 2> out) {
    if ((m - 5) % 2)  // nv = 2^(m-5)
        gather_units<2>(grid, m, s, z, out[0], out[1]);
    else
        gather_units<4>(grid, m, s, z, out[0], out[1]);
}

// Sets up level (M = m >= 2); returns where its sign parts go.
std::array<u32*, 2> open_level(Level& level, int m, Arena& arena) {
    const std::size_t len = std::size_t(1) << (m - 2);
    level.length = len;
    if (!transformed(level)) return {level.fold[0], level.fold[1]};
    for (u32*& p : level.part) p = arena.words(len);
    return {level.part[0], level.part[1]};
}

// Sum of a level whose folds are written.
void close_level(Level& level) {
    u64 t = 0;
    for (std::size_t i = 0; i < std::min(level.length, kFold); ++i) t += level.fold[0][i];
    level.sum = u32(t % kP);
}

// Reads 2^n values and splits them into their levels.
// Parts of levels with m >= kBlockedLog are after the top layer; folds and sums of transformed
// levels are left to the caller.
void split(io::Reader& in, int n, Level* levels, Arena& arena, const Buffers& buffers, const Factor& z) {
    u32* cur = buffers.band;  // cur[i] = x[i 2^s], i < 2^m
    int s = 0;
    if (n >= kBlockedLog) {
        const int rows_log = n - kRowLog;
        const std::size_t width = std::size_t(1) << rows_log;
        for (std::size_t first = 0; first < kHeight; first += kBand) {
            io::read_bulk(in, buffers.band, kBand * width);
            to_grid(buffers.band, width, first, buffers.grid);
        }
        for (; n - s >= kBlockedLog; ++s) gather(buffers.grid, n - s, s, z, open_level(levels[s], n - s, arena));
        cur = buffers.small[0];
        for (std::size_t i = 0; i < std::size_t(1) << (n - s); ++i) {
            const std::size_t x = i << s;
            cur[i] = buffers.grid[(x & (width - 1)) * kPitch + (x >> rows_log)];
        }
    } else {
        io::read_bulk(in, cur, std::size_t(1) << n);
    }
    for (; s <= n; ++s) {
        const int m = n - s;
        Level& level = levels[s];
        if (m < 2) {
            level.sum = cur[m];
            continue;  // level s + 1 is cur[0]
        }
        u32* other = cur == buffers.small[0] ? buffers.small[1] : buffers.small[0];
        const auto out = open_level(level, m, arena);
        const u32 mask = (u32(1) << m) - 1;
        u32 u = 1;
        for (std::size_t k = 0; k < level.length; ++k, u = u * 5 & mask) {
            out[0][k] = add_mod(cur[u], cur[mask + 1 - u]);
            out[1][k] = sub_mod(cur[u], cur[mask + 1 - u]);
        }
        for (std::size_t i = 0; i < std::size_t(1) << (m - 1); ++i) other[i] = cur[2 * i];
        if (!transformed(level)) close_level(level);
        cur = other;
    }
}

// Writes c from its levels, top down: after the level of the units mod 2^m, c[i 2^(n-m)] for
// i < 2^m are known. Levels with m >= kBlockedLog go to the grid, which goes to the output
// kBand rows at a time.
class Assembly {
public:
    Assembly(int n, const Buffers& buffers, io::Writer& out, u32 zero)
        : n_(n), buffers_(buffers), out_(out), cur_(buffers.small[0]), other_(buffers.small[1]) {
        cur_[0] = zero;
        if (n == 0) fields::write(out_, cur_, 1, buffers_.text);
    }

    // m = 1: the unit 1.
    void push_odd(u32 value) {
        cur_[1] = value;
        if (n_ == 1) fields::write(out_, cur_, 2, buffers_.text);
    }

    // m >= 2: the values at +-5^k are plus[k] +- minus[k], k < 2^(m-2); canonical.
    void push(int m, const u32* plus, const u32* minus) {
        if (m >= kBlockedLog) {
            const int s = n_ - m, rows_log = n_ - kRowLog;
            const std::size_t width = std::size_t(1) << rows_log;
            if (m == kBlockedLog)  // c[x], x = i 2^(s+1): the smaller levels
                for (std::size_t i = 0; i < std::size_t(1) << (m - 1); ++i) {
                    const std::size_t x = i << (s + 1);
                    buffers_.grid[(x & (width - 1)) * kPitch + (x >> rows_log)] = cur_[i];
                }
            scatter_units(plus, minus, m, s, buffers_.grid);
            if (m == n_)
                for (std::size_t first = 0; first < kHeight; first += kBand) {
                    from_grid(buffers_.grid, width, first, buffers_.band);
                    write_band(kBand * width, first + kBand == kHeight);
                }
            return;
        }
        const std::size_t len = std::size_t(1) << (m - 2);
        for (std::size_t i = 0; i < 2 * len; ++i) other_[2 * i] = cur_[i];
        const u32 mask = (u32(1) << m) - 1;
        u32 u = 1;
        for (std::size_t k = 0; k < len; ++k, u = u * 5 & mask) {
            other_[u] = add_mod(plus[k], minus[k]);
            other_[mask + 1 - u] = sub_mod(plus[k], minus[k]);
        }
        if (m == n_) fields::write(out_, other_, std::size_t(1) << m, buffers_.text);
        std::swap(cur_, other_);
    }

private:
    // band[0, count) as fixed-width fields (as fields::write); count a multiple of 16.
    void write_band(std::size_t count, bool last) {
        fields::detail::format(buffers_.text, buffers_.band, count, fields::detail::kConstants);
        if (last) buffers_.text[10 * count - 1] = '\n';
        out_.write(std::string_view(buffers_.text, 10 * count));
    }

    int n_;
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

// [w x, x] for leaf v of the levels s <= k of a (sign e).
using Windows = std::array<std::array<u32, 16>, kMaxLog + 1>;

void make_windows(const Level* a, int e, int k, std::size_t v, const Factor& w, Windows& windows) {
    for (int s = 0; s <= k; ++s) {
        const Vec x = load(a[s].part[e] + 8 * v);
        store(windows[s].data(), reduce(multiply(x, w), kP));
        store(windows[s].data() + 8, x);
    }
}

// Leaves [first, end) of the levels s <= k, sign e: level t of c gets the sum over s + t' = t of
// a_s b_t', in place of b_t. The windows of leaf v + 1 are stored before the products of leaf v
// load those of v: unaligned loads from recent stores stall.
void band_products(const Level* a, Level* b, int e, int k, std::size_t first, std::size_t end, const Transform& transform) {
    Windows windows[2];
    make_windows(a, e, k, first, transform.leaf_weight(first), windows[0]);
    for (std::size_t v = first; v < end; ++v) {
        if (v + 1 < end) make_windows(a, e, k, v + 1, transform.leaf_weight(v + 1), windows[(v + 1 - first) % 2]);
        const Windows& x = windows[(v - first) % 2];
        for (int t = k; t >= 0; --t) {  // b_t is last read for level t
            Vec even = _mm256_setzero_si256(), odd = even;
            for (int s = 0; s <= t; ++s) multiply_add(x[s].data(), b[t - s].part[e] + 8 * v, even, odd);
            store(b[t].part[e] + 8 * v, finish(even, odd));
        }
    }
}

// Output level m > kDirectLog from its leaves in b[n - m].
void transform_level(Assembly& c, int n, int m, const Level* b, const Transform& transform) {
    const std::size_t len = std::size_t(1) << (m - 2), nv = len / 8;
    const u32 scale = mul_mod(mul_mod(ntt::detail::power(u32(nv), kP - 2), ntt::detail::kR), kHalf);
    u32* const* out = b[n - m].part;
    for (int e = 0; e < 2; ++e) transform.inverse(reinterpret_cast<Vec*>(out[e]), nv, scale);
    c.push(m, out[0], out[1]);
}

void solve() {
    io::Reader in;
    const int n = in.read<int>();
    const std::size_t size = std::size_t(1) << n;
    // Under 8 words per value, the text (under 10 bytes per value) and 64 KiB of padding.
    Arena arena(4 * 8 * size + 10 * size + fields::kTextBytes + (64 << 10));
    const Buffers buffers(arena, n);

    const Transform transform(arena, std::max<std::size_t>(size / 32, 8));
    const int levels = std::max(n - kDirectLog, 0);  // transformed: s < levels
    std::array<Level, kMaxLog + 1> a, b;
    for (auto* x : {&a, &b}) {
        split(in, n, x->data(), arena, buffers, transform.top_root());
        for (int s = 0; s < levels; ++s) {
            Level& level = (*x)[s];
            for (int e = 0; e < 2; ++e) {
                auto* f = reinterpret_cast<Vec*>(level.part[e]);
                transform.forward(f, level.length / 8, top_done(n - s));
                transform.fold(f, level.fold[e]);
            }
            close_level(level);
        }
    }

    io::Writer out;
    u32 zero = 0;
    for (int s = 0; s <= n; ++s)
        for (int t = n - s; t <= n; ++t) zero = add_mod(zero, mul_mod(a[s].sum, b[t].sum));
    Assembly c(n, buffers, out, zero);
    if (n >= 1) {
        u32 odd = 0;
        for (int s = 0; s <= n - 1; ++s) odd = add_mod(odd, mul_mod(a[s].sum, b[n - 1 - s].sum));
        c.push_odd(odd);
    }
    for (int m = 2; m <= std::min(n, kDirectLog); ++m) direct_level(c, n, m, a.data(), b.data());

    if (n > kDirectLog) {
        for (int e = 0; e < 2; ++e)  // leaves [nv of level k + 1, nv of level k) are in levels 0..k
            for (int k = levels - 1; k >= 0; --k)
                band_products(a.data(), b.data(), e, k, k + 1 < levels ? a[k + 1].length / 8 : 0, a[k].length / 8, transform);
        for (int m = kDirectLog + 1; m <= n; ++m) transform_level(c, n, m, b.data(), transform);
    }
}

}  // namespace

RUN_EARLY(solve)
