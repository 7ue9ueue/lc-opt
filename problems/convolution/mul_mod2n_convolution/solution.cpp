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
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
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

    // Canonical f -> its leaves, canonical; with ext, leaf v goes to ext[2v + 1] and w times it to
    // ext[2v] instead. ext may be f - nv: leaves are written in order, behind the unread input.
    void forward(Vec* f, std::size_t nv, Vec* ext) const {
        if (std::countr_zero(nv) % 2) {
            const std::size_t h = nv / 2;
            ntt::detail::forward_radix2(f, h, false);
            visit(f, h, 0, ext);
            visit(f + h, h, 1, ext);
        } else {
            visit(f, nv, 0, ext);
        }
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
    void visit(Vec* f, std::size_t nv, std::size_t k, Vec* ext) const {
        if (nv == 4) return last_forward(f, k, ext);
        const std::size_t h = nv / 4;
        if (k == 0)
            ntt::kernels::forward_identity(f, h, roots_);
        else
            ntt::kernels::forward(f, h, roots_ + slot(k), roots_ + slot(2 * k));
        for (std::size_t t = 0; t < 4; ++t) visit(f + t * h, h, 4 * k + t, ext);
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
    void last_forward(Vec* f, std::size_t k, Vec* ext) const {
        const u32 *x = roots_ + slot(k), *y = roots_ + slot(2 * k);
        const Factor fx(x[0], x[8]), fy(y[0], y[8]), fz(y[1], y[9]);
        const Vec f0 = reduce(f[0], 2 * kP), f1 = reduce(f[1], 2 * kP);
        const Vec xf2 = multiply(f[2], fx), xf3 = multiply(f[3], fx);  // < 2P
        const Vec a = reduce(add(f0, xf2), 2 * kP), c = reduce(diff(f0, xf2), 2 * kP);
        const Vec yb = multiply(add(f1, xf3), fy), zd = multiply(diff(f1, xf3), fz);
        const Vec leaf[4] = {canonical(add(a, yb)), canonical(diff(a, yb)), canonical(add(c, zd)),
                             canonical(diff(c, zd))};
        if (!ext) {
            for (int t = 0; t < 4; ++t) f[t] = leaf[t];
            return;
        }
        const Factor w[4] = {fy, Factor(kP - y[0], ~y[8]), fz, Factor(kP - y[1], ~y[9])};  // q(P - w) = ~q(w)
        for (int t = 0; t < 4; ++t) {
            ext[8 * k + 2 * t] = reduce(multiply(leaf[t], w[t]), kP);
            ext[8 * k + 2 * t + 1] = leaf[t];
        }
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

// x y 2^-32 mod (z^8 - w), < 2P. ext = [w x, x], y: canonical leaves.
Vec leaf_product(const Vec* ext, const Vec* y) {
    const auto* window = reinterpret_cast<const u32*>(ext);  // z^j x = window [8 - j, 16 - j)
    const auto* yw = reinterpret_cast<const u32*>(y);
    Vec even = _mm256_setzero_si256(), odd = even;  // sums of 8 products < 8 P^2
    for (int j = 0; j < 8; ++j) {
        const Vec yj = broadcast(yw[j]);
        even = _mm256_add_epi64(even, _mm256_mul_epu32(_mm256_loadu_si256(reinterpret_cast<const Vec*>(window + 8 - j)), yj));
        odd = _mm256_add_epi64(odd, _mm256_mul_epu32(_mm256_loadu_si256(reinterpret_cast<const Vec*>(window + 9 - j)), yj));
    }
    const Vec r = _mm256_blend_epi32(_mm256_srli_epi64(montgomery(even), 32), montgomery(odd), 0xAA);  // < 2.9P
    return reduce(r, 2 * kP);
}

// One level of one factor: sign parts of length L = 2^(M-2), M = N - s.
struct Level {
    std::size_t length = 0;     // L, 0 for M < 2
    u32 sum = 0;                // sum of the level's values mod P
    u32 fold[2][kFold] = {};    // [+], [-] folded to min(L, kFold) words
    u32* part[2] = {};          // L > kFold: [+], [-], then their leaves (factor b)
    Vec* ext[2] = {};           // factor a, L > kFold: leaves as [w x, x]; part is its upper half
};

bool transformed(const Level& level) { return level.length > kFold; }

// Level s of x (2^n values) by sign and discrete log.
// Levels with M >= kBlockedLog are permuted through rows: the odd values of x[0, 2^M) as
// rows[t][h] = x[2t + 1 + 2^b h], t < 2^(b-1), h < 2^kRowLog, b = M - kRowLog. With
// k = k_lo + 2^(b-2) k_hi, 5^k = 5^k_lo G^k_hi, G = 5^(2^(b-2)) = 1 mod 2^b: the row of +-5^k
// depends on k_lo alone, so eight consecutive k_lo use 16 rows (16 KiB) for all k_hi.
constexpr int kBlockedLog = 13;
constexpr int kRowLog = 8;

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

// Words 1, 3, ..., 15 (odd) or 0, 2, ..., 14 of [lo, hi].
template <int Control>
Vec alternate_words(Vec lo, Vec hi) {
    const __m256 w = _mm256_shuffle_ps(_mm256_castsi256_ps(lo), _mm256_castsi256_ps(hi), Control);
    return _mm256_permute4x64_epi64(_mm256_castps_si256(w), 0xD8);
}

constexpr std::size_t kBand = 16;  // rows of x per pass: whole cache lines of rows

std::size_t row_width(int m) { return std::size_t(1) << (m - kRowLog); }

// Rows [first, first + kBand) of x (band) -> their odd values into rows, x[2i] to evens[i].
void to_rows(const u32* band, int m, std::size_t first, u32* rows, u32* evens) {
    const std::size_t width = row_width(m), height = std::size_t(1) << kRowLog;
    for (std::size_t c = 0; c < width; c += 16)
        for (std::size_t h = 0; h < kBand; h += 8) {
            Vec odd[8];
            for (std::size_t r = 0; r < 8; ++r) {
                const u32* p = band + (h + r) * width + c;
                const Vec lo = load(p), hi = load(p + 8);
                odd[r] = alternate_words<0xDD>(lo, hi);
                store(evens + (first + h + r) * width / 2 + c / 2, alternate_words<0x88>(lo, hi));
            }
            transpose(odd);
            for (std::size_t i = 0; i < 8; ++i) store(rows + (c / 2 + i) * height + first + h, odd[i]);
        }
}

// Inverse of to_rows.
void from_rows(const u32* rows, const u32* evens, int m, std::size_t first, u32* band) {
    const std::size_t width = row_width(m), height = std::size_t(1) << kRowLog;
    for (std::size_t c = 0; c < width; c += 16)
        for (std::size_t h = 0; h < kBand; h += 8) {
            Vec odd[8];
            for (std::size_t i = 0; i < 8; ++i) odd[i] = load(rows + (c / 2 + i) * height + first + h);
            transpose(odd);
            for (std::size_t r = 0; r < 8; ++r) {
                const Vec e = _mm256_permute4x64_epi64(load(evens + (first + h + r) * width / 2 + c / 2), 0xD8);
                const Vec o = _mm256_permute4x64_epi64(odd[r], 0xD8);
                u32* p = band + (h + r) * width + c;
                store(p, _mm256_unpacklo_epi32(e, o));
                store(p + 8, _mm256_unpackhi_epi32(e, o));
            }
        }
}

// Row and column offsets of +-5^k for eight consecutive k_lo, stepping k_hi.
class UnitWalk {
public:
    UnitWalk(int m, std::size_t k_lo) : b_(m - kRowLog), mask_(broadcast((u32(1) << m) - 1)) {
        const u32 mask = (u32(1) << m) - 1;
        u32 g = 5, p = 1;
        for (int i = 0; i < b_ - 2; ++i) g = g * g & mask;
        for (std::size_t i = 0; i < k_lo; ++i) p = p * 5 & mask;
        alignas(32) u32 lanes[8];
        for (u32& lane : lanes) lane = p, p = p * 5 & mask;
        unit_ = _mm256_load_si256(reinterpret_cast<const Vec*>(lanes));
        step_ = broadcast(g);
        const Vec t = _mm256_srli_epi32(_mm256_and_si256(unit_, broadcast((u32(1) << b_) - 1)), 1);
        row_pos_ = _mm256_slli_epi32(t, kRowLog);
        row_neg_ = _mm256_sub_epi32(broadcast(((u32(1) << (b_ - 1)) - 1) << kRowLog), row_pos_);
    }

    // Offsets in rows of 5^k and -5^k for the current k_hi; then the next k_hi.
    Vec positive() const { return _mm256_add_epi32(row_pos_, _mm256_srli_epi32(unit_, b_)); }
    Vec negative() const {
        return _mm256_sub_epi32(_mm256_add_epi32(row_neg_, broadcast((u32(1) << kRowLog) - 1)), _mm256_srli_epi32(unit_, b_));
    }
    void next() { unit_ = _mm256_and_si256(_mm256_mullo_epi32(unit_, step_), mask_); }

private:
    int b_;
    Vec mask_, unit_, step_, row_pos_, row_neg_;
};

// Rows of level m -> plus[k], minus[k] = x(5^k) +- x(-5^k), k < 2^(m-2).
void gather_units(const u32* rows, int m, u32* plus, u32* minus) {
    const std::size_t stride = std::size_t(1) << (m - kRowLog - 2), height = std::size_t(1) << kRowLog;
    const auto* base = reinterpret_cast<const int*>(rows);
    for (std::size_t k = 0; k < stride; k += 8) {
        UnitWalk walk(m, k);
        for (std::size_t j = k; j < k + height * stride; j += stride, walk.next()) {
            const Vec xp = _mm256_i32gather_epi32(base, walk.positive(), 4);
            const Vec xn = _mm256_i32gather_epi32(base, walk.negative(), 4);
            store(plus + j, add_canonical(xp, xn));
            store(minus + j, sub_canonical(xp, xn));
        }
    }
}

// Values at 5^k (pos) and -5^k (neg), k < 2^(m-2) -> rows of level m.
void scatter_units(const u32* pos, const u32* neg, int m, u32* rows) {
    const std::size_t stride = std::size_t(1) << (m - kRowLog - 2), height = std::size_t(1) << kRowLog;
    for (std::size_t k = 0; k < stride; k += 8) {
        UnitWalk walk(m, k);
        for (std::size_t j = k; j < k + height * stride; j += stride, walk.next()) {
            alignas(32) u32 at_pos[8], at_neg[8];
            _mm256_store_si256(reinterpret_cast<Vec*>(at_pos), walk.positive());
            _mm256_store_si256(reinterpret_cast<Vec*>(at_neg), walk.negative());
            for (std::size_t l = 0; l < 8; ++l) rows[at_pos[l]] = pos[j + l], rows[at_neg[l]] = neg[j + l];
        }
    }
}

// Canonical x[0, len) folded to kFold words: out[i] = sum of x[k], k = i mod kFold. len a multiple
// of kFold.
void fold(const u32* x, std::size_t len, u32* out) {
    Vec sum[kFold / 8] = {};
    for (std::size_t j = 0; j < len; j += kFold)
        for (std::size_t i = 0; i < kFold / 8; ++i) sum[i] = add_canonical(sum[i], load(x + j + 8 * i));
    for (std::size_t i = 0; i < kFold / 8; ++i) store(out + 8 * i, sum[i]);
}

// Work memory, in words. Below kBlockedLog every buffer holds 2^n words.
struct Buffers {
    u32* band;      // kBand rows of x: 2^(n-4)
    u32* level[2];  // c and the factors by level, 2^(n-1) each
    u32* rows;      // 2^(n-1)
    u32* acc;       // 2^(n-1) + 2 kPadding
    char* text;     // 10 words of band, at least fields::kTextBytes

    Buffers(Arena& arena, int n) {
        const std::size_t size = std::size_t(1) << n, half = n >= kBlockedLog ? size / 2 : size;
        band = arena.words(n >= kBlockedLog ? size / 16 : size);
        for (u32*& p : level) p = arena.words(half);
        rows = arena.words(half);
        acc = arena.words(half + 2 * kPadding);
        text = arena.bytes(std::max(fields::kTextBytes, n >= kBlockedLog ? 10 * size / 16 : 0));
    }
};

// Sets up level (M = m >= 2); returns where its sign parts go. extended: parts in the upper halves
// of ext (factor a).
std::array<u32*, 2> open_level(Level& level, int m, Arena& arena, bool extended) {
    const std::size_t len = std::size_t(1) << (m - 2);
    level.length = len;
    if (!transformed(level)) return {level.fold[0], level.fold[1]};
    for (int e = 0; e < 2; ++e) {
        u32* block = arena.words(extended ? 2 * len : len);
        if (extended) level.ext[e] = reinterpret_cast<Vec*>(block), block += len;
        level.part[e] = block;
    }
    return {level.part[0], level.part[1]};
}

// Folds and sum of a level whose parts are written.
void close_level(Level& level) {
    if (transformed(level))
        for (int e = 0; e < 2; ++e) fold(level.part[e], level.length, level.fold[e]);
    u64 t = 0;
    for (std::size_t i = 0; i < std::min(level.length, kFold); ++i) t += level.fold[0][i];
    level.sum = u32(t % kP);
}

// Reads 2^n values and splits them into their levels.
void split(io::Reader& in, int n, Level* levels, Arena& arena, const Buffers& buffers, bool extended) {
    u32* cur = buffers.band;  // cur[i] = x[i 2^s], i < 2^m
    int s = 0;
    if (n >= kBlockedLog) {  // level 0 straight from the input, kBand rows at a time
        const std::size_t width = row_width(n);
        for (std::size_t first = 0; first < std::size_t(1) << kRowLog; first += kBand) {
            in.read(buffers.band, kBand * width);
            to_rows(buffers.band, n, first, buffers.rows, buffers.level[0]);
        }
        const auto out = open_level(levels[0], n, arena, extended);
        gather_units(buffers.rows, n, out[0], out[1]);
        close_level(levels[0]);
        cur = buffers.level[0];
        s = 1;
    } else {
        in.read(cur, std::size_t(1) << n);
    }
    for (; s <= n; ++s) {
        const int m = n - s;
        Level& level = levels[s];
        if (m < 2) {
            level.sum = cur[m];
            continue;  // level s + 1 is cur[0]
        }
        u32* other = cur == buffers.level[0] ? buffers.level[1] : buffers.level[0];
        const auto out = open_level(level, m, arena, extended);
        if (m >= kBlockedLog) {
            for (std::size_t first = 0; first < std::size_t(1) << kRowLog; first += kBand)
                to_rows(cur + first * row_width(m), m, first, buffers.rows, other);
            gather_units(buffers.rows, m, out[0], out[1]);
        } else {
            const u32 mask = (u32(1) << m) - 1;
            u32 u = 1;
            for (std::size_t k = 0; k < level.length; ++k, u = u * 5 & mask) {
                out[0][k] = add_mod(cur[u], cur[mask + 1 - u]);
                out[1][k] = sub_mod(cur[u], cur[mask + 1 - u]);
            }
            for (std::size_t i = 0; i < std::size_t(1) << (m - 1); ++i) other[i] = cur[2 * i];
        }
        close_level(level);
        cur = other;
    }
}

// Writes c from its levels, top down: after the level of the units mod 2^m, c[i 2^(n-m)] for
// i < 2^m are known. The last level (m = n) goes to the output kBand rows at a time.
class Assembly {
public:
    Assembly(int n, const Buffers& buffers, io::Writer& out, u32 zero)
        : n_(n), buffers_(buffers), out_(out), cur_(buffers.level[0]), other_(buffers.level[1]) {
        cur_[0] = zero;
        if (n == 0) fields::write(out_, cur_, 1, buffers_.text);
    }

    // m = 1: the unit 1.
    void push_odd(u32 value) {
        cur_[1] = value;
        if (n_ == 1) fields::write(out_, cur_, 2, buffers_.text);
    }

    // m >= 2: the value at 5^k is pos[k], at -5^k neg[k], k < 2^(m-2).
    void push(int m, const u32* pos, const u32* neg) {
        if (m >= kBlockedLog) {
            scatter_units(pos, neg, m, buffers_.rows);
            const std::size_t width = row_width(m), height = std::size_t(1) << kRowLog;
            for (std::size_t first = 0; first < height; first += kBand) {
                if (m < n_) {
                    from_rows(buffers_.rows, cur_, m, first, other_ + first * width);
                } else {
                    from_rows(buffers_.rows, cur_, m, first, buffers_.band);
                    write_band(kBand * width, first + kBand == height);
                }
            }
        } else {
            const std::size_t len = std::size_t(1) << (m - 2);
            for (std::size_t i = 0; i < 2 * len; ++i) other_[2 * i] = cur_[i];
            const u32 mask = (u32(1) << m) - 1;
            u32 u = 1;
            for (std::size_t k = 0; k < len; ++k, u = u * 5 & mask) other_[u] = pos[k], other_[mask + 1 - u] = neg[k];
            if (m == n_) fields::write(out_, other_, std::size_t(1) << m, buffers_.text);
        }
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
    u32 pos[kFold], neg[kFold];
    for (std::size_t k = 0; k < len; ++k) {
        pos[k] = mul_mod(add_mod(result[0][k], result[1][k]), kHalf);
        neg[k] = mul_mod(sub_mod(result[0][k], result[1][k]), kHalf);
    }
    c.push(m, pos, neg);
}

// Output level m > kDirectLog by transforms; acc: 2 L + 2 kPadding words.
void transform_level(Assembly& c, int n, int m, const Level* a, const Level* b, const Transform& transform, u32* acc) {
    const std::size_t len = std::size_t(1) << (m - 2), nv = len / 8;
    const u32 scale = mul_mod(mul_mod(ntt::detail::power(u32(nv), kP - 2), ntt::detail::kR), kHalf);
    u32* out[2] = {acc, acc + len + kPadding};
    for (int e = 0; e < 2; ++e) {
        auto* f = reinterpret_cast<Vec*>(out[e]);
        for (std::size_t v = 0; v < nv; ++v) {
            Vec sum = _mm256_setzero_si256();
            for (int s = 0; s <= n - m; ++s) {
                const Vec p = leaf_product(a[s].ext[e] + 2 * v, reinterpret_cast<const Vec*>(b[n - m - s].part[e]) + v);
                sum = reduce(add(sum, p), 2 * kP);
            }
            f[v] = sum;
        }
        transform.inverse(f, nv, scale);
    }
    for (std::size_t k = 0; k < len; k += 8) {
        const Vec p = load(out[0] + k), q = load(out[1] + k);
        store(out[0] + k, add_canonical(p, q));
        store(out[1] + k, sub_canonical(p, q));
    }
    c.push(m, out[0], out[1]);
}

void solve() {
    io::Reader in;
    const int n = in.read<int>();
    const std::size_t size = std::size_t(1) << n;
    // Under 8 words per value, the text (under 10 bytes per value) and 64 KiB of padding.
    Arena arena(4 * 8 * size + 10 * size + fields::kTextBytes + (64 << 10));
    const Buffers buffers(arena, n);

    std::array<Level, kMaxLog + 1> a, b;
    split(in, n, a.data(), arena, buffers, true);
    split(in, n, b.data(), arena, buffers, false);

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
        const Transform transform(arena, size / 32);
        for (int s = 0; s <= n; ++s) {
            if (!transformed(a[s])) break;
            const std::size_t nv = a[s].length / 8;
            for (int e = 0; e < 2; ++e) {
                transform.forward(reinterpret_cast<Vec*>(a[s].part[e]), nv, a[s].ext[e]);
                transform.forward(reinterpret_cast<Vec*>(b[s].part[e]), nv, nullptr);
            }
        }
        for (int m = kDirectLog + 1; m <= n; ++m) transform_level(c, n, m, a.data(), b.data(), transform, buffers.acc);
    }
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
