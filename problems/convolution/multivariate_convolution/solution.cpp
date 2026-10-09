// Truncated multivariate product mod 998244353: c = f g mod (x_1^n_1, ..., x_k^n_k), N = prod n_l.
// Three methods (notes.md):
// - Ranked: every n_l <= 3 and at least three n_l = 2. Each variable is evaluated at n_l points
//   ({0, 1} or {0, 1, -1}); a rank t^|d| tracks the total degree, so terms that wrapped are dropped.
// - Graded: any shape. One cyclic NTT over the flat index, graded by chi(i) = sum_j floor(i / P_j)
//   mod m; each carry adds 1 to chi, so a product term is valid iff chi(i) + chi(j) = chi(i + j)
//   mod m. m >= k, m | P - 1.
// - Split: a few outer variables by schoolbook over their digits, the rest graded with fewer
//   grades and shorter transforms (transform.hpp). A cost model picks it or the graded method.
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
#include "../fixed_width.hpp"
#include "transform.hpp"

#ifndef BLOCK_BYTES
#define BLOCK_BYTES (320 << 10)  // the ranked method's working set per top point: half of Zen 3's L2
#endif

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Vec = __m256i;

constexpr u32 kP = 998244353;
constexpr u32 kPInverse = [] {  // P^-1 mod 2^32
    u32 x = kP;
    for (int i = 0; i < 5; ++i) x *= 2 - kP * x;
    return x;
}();
constexpr u32 k2To32 = u32((u64(1) << 32) % kP);

constexpr u32 multiply_mod(u32 a, u32 b) { return u32(u64(a) * b % kP); }

constexpr u32 power_mod(u32 a, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, a = multiply_mod(a, a))
        if (e & 1) r = multiply_mod(r, a);
    return r;
}

constexpr u32 inverse_mod(u32 a) { return power_mod(a, kP - 2); }
constexpr u32 montgomery(u32 a) { return multiply_mod(a, k2To32); }  // a 2^32

Vec all(u32 x) { return _mm256_set1_epi32(int(x)); }
Vec load(const void* p) { return _mm256_loadu_si256(static_cast<const Vec*>(p)); }
void store(void* p, Vec x) { _mm256_storeu_si256(static_cast<Vec*>(p), x); }

// Arithmetic on residues in [0, P).
Vec add_mod(Vec x, Vec y) {
    const Vec s = _mm256_add_epi32(x, y);
    return _mm256_min_epu32(s, _mm256_sub_epi32(s, all(kP)));
}

Vec sub_mod(Vec x, Vec y) {
    const Vec d = _mm256_sub_epi32(x, y);
    return _mm256_min_epu32(d, _mm256_add_epi32(d, all(kP)));
}

// Products in 64-bit lanes: even holds lanes 0, 2, 4, 6, odd lanes 1, 3, 5, 7. Returns the eight
// values times 2^-32 mod P, in [0, P). Wide: any 64-bit values; otherwise each is < P 2^32.
template <bool Wide>
Vec reduce(Vec even, Vec odd) {
    Vec high = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
    if constexpr (Wide) {  // high < 2^32 -> < P
        high = _mm256_min_epu32(high, _mm256_sub_epi32(high, all(2 * kP)));
        high = _mm256_min_epu32(high, _mm256_sub_epi32(high, all(2 * kP)));
        high = _mm256_min_epu32(high, _mm256_sub_epi32(high, all(kP)));
    }
    // q = low / P mod 2^32; (value - q P) / 2^32 = high - floor(q P / 2^32), in (-P, P).
    const Vec qe = _mm256_mul_epu32(even, all(kPInverse)), qo = _mm256_mul_epu32(odd, all(kPInverse));
    const Vec me = _mm256_mul_epu32(qe, all(kP)), mo = _mm256_mul_epu32(qo, all(kP));
    const Vec r = _mm256_sub_epi32(high, _mm256_blend_epi32(_mm256_srli_epi64(me, 32), mo, 0xAA));
    return _mm256_min_epu32(r, _mm256_add_epi32(r, all(kP)));
}

// x y 2^-32 mod P for x, y < P.
Vec montgomery_multiply(Vec x, Vec y) {
    const Vec even = _mm256_mul_epu32(x, y);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), _mm256_srli_epi64(y, 32));
    return reduce<false>(even, odd);
}

// Zero-filled memory, never freed (the process ends with _exit). Blocks of 256 KiB or more are
// aligned to 2 MiB and in huge pages where the kernel allows; smaller ones take small pages, so a
// small block does not fault in (and zero) a whole huge page.
template <class T>
T* allocate(std::size_t count) {
    constexpr std::size_t kHuge = std::size_t(1) << 21, kPage = std::size_t(1) << 12;
    const std::size_t bytes = count * sizeof(T) + 64;
    const bool huge = bytes >= kHuge / 8;
    const std::size_t mapped = huge ? (bytes + kHuge - 1) / kHuge * kHuge + kHuge : (bytes + kPage - 1) / kPage * kPage;
    void* p = ::mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) std::abort();
    if (!huge) return static_cast<T*>(p);
    void* aligned = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(p) + kHuge - 1) & ~(kHuge - 1));
#ifdef MADV_HUGEPAGE
    ::madvise(aligned, mapped - kHuge, MADV_HUGEPAGE);
#endif
    return static_cast<T*>(aligned);
}

std::size_t round8(std::size_t n) { return (n + 7) & ~std::size_t(7); }

// ---------------------------------------------------------------------------------------------
// Graded method. A carry out of digit j < k - 1 adds 1 to chi(i + j) - chi(i) - chi(j), and the
// linear product (length >= 2N - 1) has no carry out of the top digit, so at most k - 1 carries:
// grades mod m >= k separate them. t -> w^s for an m-th root w turns each grade class into one
// ordinary convolution: c_i = (1/m) sum_s w^(-s chi_i) [(f w^(s chi)) * (g w^(s chi))]_i.

u32 grade_modulus(std::size_t k) {
    for (u32 m : {1, 2, 4, 7, 8, 14, 16, 17, 28, 32})
        if (m >= k) return m;
    std::abort();
}

// chi(i) mod m for i < size (zero up to size rounded to 8), digits of sizes n.
std::uint8_t* grades(const std::vector<u32>& n, std::size_t size, u32 m) {
    std::uint8_t* grade = allocate<std::uint8_t>(round8(size));
    std::vector<u32> digit(n.size(), 0);
    std::uint8_t chi = 0;
    for (std::size_t i = 0; i < size; ++i) {
        grade[i] = chi;
        for (std::size_t j = 0; j + 1 < n.size() && ++digit[j] == n[j]; ++j) {
            digit[j] = 0;
            if (++chi == m) chi = 0;
        }
    }
    return grade;
}

// table[index] for 8 indices < 8 Tables, table in Tables vectors.
template <int Tables>
Vec lookup(const Vec* table, Vec index) {
    const auto pick = [index](Vec x, Vec y, int bit) {  // x or y by the given bit of index
        const __m256 select = _mm256_castsi256_ps(_mm256_slli_epi32(index, 31 - bit));
        return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), select));
    };
    const Vec x = _mm256_permutevar8x32_epi32(table[0], index);
    if constexpr (Tables == 1) {
        return x;
    } else {
        const Vec low = pick(x, _mm256_permutevar8x32_epi32(table[1], index), 3);
        if constexpr (Tables == 2) return low;
        const Vec high = pick(_mm256_permutevar8x32_epi32(table[2], index),
                              _mm256_permutevar8x32_epi32(table[3], index), 3);
        return pick(low, high, 4);
    }
}

template <int Tables>
void graded(const std::vector<u32>& n, std::size_t size, const u32* f, const u32* g, u32* c, u32 m) {
    const std::size_t padded = round8(size);
    const u32 root = power_mod(3, (kP - 1) / m), inverse_m = inverse_mod(m);

    const std::uint8_t* grade = grades(n, size, m);
    const auto grade_of = [grade](std::size_t i) {
        return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(grade + i)));
    };

    ntt::Convolution product(size, size);
    // Both factors fill at most half the transform, so lib/ntt reads only that half ("sparse").
    const std::size_t half = (std::size_t(1) << std::max(6, int(std::bit_width(2 * size - 2)))) / 2;
    u32 *a = product.a(), *b = product.b();
    for (u32 s = 0; s < m; ++s) {
        // w^(s r) and w^(-s r) / m in Montgomery form, r < m.
        alignas(32) u32 forward[32] = {}, inverse[32] = {};
        const u32 step = power_mod(root, s), inverse_step = inverse_mod(step);
        for (u32 r = 0, x = 1, y = inverse_m; r < m; ++r, x = multiply_mod(x, step), y = multiply_mod(y, inverse_step))
            forward[r] = montgomery(x), inverse[r] = montgomery(y);
        const Vec* weights = reinterpret_cast<const Vec*>(forward);
        const Vec* inverse_weights = reinterpret_cast<const Vec*>(inverse);
        if (s == 0) {
            std::memcpy(a, f, size * sizeof(u32));
            std::memcpy(b, g, size * sizeof(u32));
        } else {
            for (std::size_t i = 0; i < padded; i += 8) {
                const Vec w = lookup<Tables>(weights, grade_of(i));
                store(a + i, montgomery_multiply(load(f + i), w));
                store(b + i, montgomery_multiply(load(g + i), w));
            }
        }
        std::memset(a + size, 0, (half - size) * sizeof(u32));
        std::memset(b + size, 0, (half - size) * sizeof(u32));
        const u32* h = product.multiply();  // rebuilds its tables; reads only a() and b()
        for (std::size_t i = 0; i < padded; i += 8) {
            const Vec x = montgomery_multiply(load(h + i), lookup<Tables>(inverse_weights, grade_of(i)));
            store(c + i, s == 0 ? x : add_mod(load(c + i), x));
        }
    }
}

void graded(const std::vector<u32>& n, std::size_t size, const u32* f, const u32* g, u32* c) {
    const u32 m = grade_modulus(n.size());
    if (m <= 8) return graded<1>(n, size, f, g, c, m);
    if (m <= 16) return graded<2>(n, size, f, g, c, m);
    graded<4>(n, size, f, g, c, m);
}


// ---------------------------------------------------------------------------------------------
// Split method. The outer variables (a subset O, Q = prod n_o positions) are multiplied by
// schoolbook over their digits; for each outer position a, f_a is the slice over the inner
// variables (N_in = N / Q positions). The inner products use the graded method with m >= |inner|
// grades, each grade one set of transforms (transform.hpp) of length 2^lg >= 2 N_in - 1: 2Q
// forward, one product per pair a + b = e without carry (digitwise), Q inverse.

// Cost model (ns, fitted on lc-amd to max_random with 7 outer sets): per element and level of a
// transform, per element and pair of the pointwise products by block size (notes.md), per element
// of a grade's passes; lib/ntt's fused product per element and level. Transforms above 2^15 are
// slower than modeled (cache), so the split stays below.
constexpr double kTransformCost = 0.107, kPassCost = 1.0, kProductCost = 0.25;
constexpr double kPairCost[6] = {0, 0.19, 0.13, 0.094, 0.094, 0.094};
constexpr int kMaxSplitLog = 15;
constexpr u32 kMaxBlock = 5;  // largest lowest outer digit whose values Split::pointwise keeps in registers

struct Plan {
    std::vector<bool> outer;  // per variable
    double cost = 1e300;
};

int split_log(std::size_t inner) { return std::max(7, int(std::bit_width(2 * inner - 1))); }

double graded_cost(const std::vector<u32>& n, std::size_t size) {
    const int lg = std::max(6, int(std::bit_width(2 * size - 2)));
    return grade_modulus(n.size()) * kProductCost * double(std::size_t(1) << lg) * lg;
}

// The cheapest split with Q > 1; outer variables are taken from the top, which keeps runs of
// consecutive inner positions long.
Plan plan_split(const std::vector<u32>& n, std::size_t size) {
    std::vector<u32> values(n);
    std::ranges::sort(values);
    const auto [first, last] = std::ranges::unique(values);
    values.erase(first, last);
    std::vector<int> available, chosen(values.size(), 0), best_chosen;
    for (u32 v : values) available.push_back(int(std::ranges::count(n, v)));
    double best = 1e300;
    // block: the largest outer size <= kMaxBlock (Split's lowest outer digit), else 1.
    const auto search = [&](auto&& self, std::size_t index, std::size_t q, std::size_t pairs, int count,
                            std::size_t block) -> void {
        if (index == values.size()) {
            const int lg = split_log(size / q);
            if (q == 1 || q > 4096 || lg > kMaxSplitLog) return;
            const double len = double(std::size_t(1) << lg);
            const double cost = grade_modulus(n.size() - count) * (3.0 * q * len * lg * kTransformCost +
                                                                   double(pairs) * len * kPairCost[block] + size * kPassCost);
            if (cost < best) best = cost, best_chosen = chosen;
            return;
        }
        const std::size_t v = values[index];
        for (int c = 0; c <= available[index] && q <= 4096; ++c) {
            chosen[index] = c;
            self(self, index + 1, q, pairs, count + c, c > 0 && v <= kMaxBlock ? v : block);
            q *= v, pairs *= v * (v + 1) / 2;
        }
        chosen[index] = 0;
    };
    search(search, 0, 1, 1, 0, 1);
    Plan plan;
    plan.outer.assign(n.size(), false);
    if (best_chosen.empty()) return plan;
    for (std::size_t i = 0; i < values.size(); ++i)
        for (std::size_t l = n.size(), left = best_chosen[i]; l-- > 0 && left;)
            if (n[l] == values[i]) plan.outer[l] = true, --left;
    plan.cost = best;
    return plan;
}

class Split {
public:
    Split(const std::vector<u32>& n, std::size_t size, const std::vector<bool>& outer) : n_(n), outer_(outer) {
        std::vector<u32> inner_sizes;
        for (std::size_t l = 0; l < n.size(); ++l)
            if (outer[l])
                order_.push_back(u32(l));
            else
                inner_sizes.push_back(n[l]), inner_ *= n[l];
        // The lowest outer digit: the largest size <= kMaxBlock, whose values pointwise() keeps in
        // registers; none (block 1) if every size is larger.
        auto lowest = order_.end();
        for (auto it = order_.begin(); it != order_.end(); ++it)
            if (n[*it] <= kMaxBlock && (lowest == order_.end() || n[*it] > n[*lowest])) lowest = it;
        if (lowest != order_.end()) block_ = n[*lowest], std::rotate(order_.begin(), lowest, lowest + 1);
        q_ = size / inner_;
        stride_ = round8(inner_);
        m_ = grade_modulus(inner_sizes.size());
        grade_ = grades(inner_sizes, inner_, m_);
        std::vector<u32> rows;
        for (std::size_t j = block_ > 1; j < order_.size(); ++j) rows.push_back(n[order_[j]]);
        describe_pairs(rows);
    }

    void multiply(const u32* f, const u32* g, u32* c) {
        if (m_ <= 8) return multiply<1>(f, g, c);
        if (m_ <= 16) return multiply<2>(f, g, c);
        multiply<4>(f, g, c);
    }

private:
    // Copies between the input order and [outer position][inner position] (row stride stride_), in
    // runs of positions consecutive in both. Outer positions take their digits in order_.
    template <bool ToSplit>
    void permute(const u32* from, u32* to) const {
        const std::size_t k = n_.size();
        std::size_t low = 0, run = 1;
        while (!outer_[low]) run *= n_[low++];
        std::vector<std::size_t> outer_step(k, 0), inner_step(k, 0);
        for (std::size_t j = 0, so = 1; j < order_.size(); so *= n_[order_[j++]]) outer_step[order_[j]] = so;
        for (std::size_t l = 0, si = 1; l < k; ++l)
            if (!outer_[l]) inner_step[l] = si, si *= n_[l];
        std::vector<u32> digit(k, 0);
        std::size_t a = 0, i = 0;
        const std::size_t size = q_ * inner_;
        for (std::size_t s = 0; s < size; s += run) {
            const std::size_t split = a * stride_ + i;
            if (ToSplit)
                std::memcpy(to + split, from + s, run * sizeof(u32));
            else
                std::memcpy(to + s, from + split, run * sizeof(u32));
            for (std::size_t l = low; l < k; ++l) {
                a += outer_step[l], i += inner_step[l];
                if (++digit[l] < n_[l]) break;
                digit[l] = 0, a -= outer_step[l] * n_[l], i -= inner_step[l] * n_[l];
            }
        }
    }

    // Rows: outer positions without the block digit. For each row e, the rows a <= e digitwise
    // (then b = e - a).
    void describe_pairs(const std::vector<u32>& sizes) {
        const std::size_t rows = q_ / block_;
        std::vector<u32> digit(sizes.size(), 0);
        pair_start_.push_back(0);
        for (std::size_t e = 0; e < rows; ++e) {
            std::vector<u32> sub(sizes.size(), 0);
            while (true) {
                std::size_t a = 0;
                for (std::size_t j = sizes.size(), s = rows; j-- > 0;) s /= sizes[j], a += sub[j] * s;
                pairs_.push_back(u32(a));
                std::size_t j = 0;
                while (j < sizes.size() && sub[j] == digit[j]) sub[j++] = 0;
                if (j == sizes.size()) break;
                ++sub[j];
            }
            pair_start_.push_back(u32(pairs_.size()));
            for (std::size_t j = 0; j < sizes.size() && ++digit[j] == sizes[j]; ++j) digit[j] = 0;
        }
    }

    template <int Tables>
    void multiply(const u32* f, const u32* g, u32* c) {
        const int lg = split_log(inner_);
        const split::Transform transform(lg);
        const std::size_t piece = transform.vectors() + 2, count = stride_ / 8;
        // One block (fewer pages to fault in): f, g and c in rows, the transforms, scratch.
        const std::size_t rows = q_ * stride_;  // a multiple of 8 words
        u32* fs = allocate<u32>(3 * rows + 8 * (2 * q_ * piece + q_));
        u32 *gs = fs + rows, *cs = gs + rows;
        Vec* pieces = reinterpret_cast<Vec*>(cs + rows);  // transforms of f_a, then of g_a; h_e replaces f_e
        Vec* scratch = pieces + 2 * q_ * piece;
        permute<true>(f, fs);
        permute<true>(g, gs);

        const u32 root = power_mod(3, (kP - 1) / m_);
        // Undoes 2^lg (inverse transform), 2^-32 twice (pointwise and weight products) and m grades.
        const u32 scale = multiply_mod(inverse_mod(multiply_mod(u32((std::size_t(1) << lg) % kP), m_)),
                                       multiply_mod(k2To32, k2To32));
        const auto grade_of = [this](std::size_t j) {  // grades of vector j
            return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(grade_ + 8 * j)));
        };
        for (u32 s = 0; s < m_; ++s) {
            alignas(32) u32 forward[32] = {}, inverse[32] = {};  // w^(s r) 2^32, w^(-s r) scale
            const u32 step = power_mod(root, s), inverse_step = inverse_mod(step);
            for (u32 r = 0, x = 1, y = scale; r < m_; ++r, x = multiply_mod(x, step), y = multiply_mod(y, inverse_step))
                forward[r] = montgomery(x), inverse[r] = y;
            const Vec* weights = reinterpret_cast<const Vec*>(forward);
            const Vec* inverse_weights = reinterpret_cast<const Vec*>(inverse);
            const auto input = [&](std::size_t i, std::size_t j) {
                const Vec x = load((i < q_ ? fs + i * stride_ : gs + (i - q_) * stride_) + 8 * j);
                return s == 0 ? x : montgomery_multiply(x, lookup<Tables>(weights, grade_of(j)));
            };
            const auto leaf = [&](std::size_t offset) {
                switch (block_) {
                case 1: return pointwise<1>(pieces, piece, offset, scratch);
                case 2: return pointwise<2>(pieces, piece, offset, scratch);
                case 3: return pointwise<3>(pieces, piece, offset, scratch);
                case 4: return pointwise<4>(pieces, piece, offset, scratch);
                default: return pointwise<5>(pieces, piece, offset, scratch);
                }
            };
            const auto output = [&](std::size_t e, std::size_t j, Vec x) {
                u32* to = cs + e * stride_ + 8 * j;
                const Vec y = montgomery_multiply(x, lookup<Tables>(inverse_weights, grade_of(j)));
                store(to, s == 0 ? y : add_mod(load(to), y));
            };
            transform.run(pieces, piece, 2 * q_, q_, count, input, leaf, output);
        }
        permute<false>(cs, c);
    }

    // At vectors [offset, offset + 8) of the pieces (canonical points): h_e = sum over pairs
    // a + b = e of f_a g_b (times 2^-32), into f_e.
    template <int Block>
    void pointwise(Vec* pieces, std::size_t piece, std::size_t offset, Vec* even) const {
        const Vec* g = pieces + q_ * piece;
        const std::size_t rows = q_ / Block;
        for (std::size_t u = offset; u < offset + 8; ++u) {
            for (std::size_t e = 0; e < rows; ++e) row_sums<Block, false>(pieces + u, g + u, piece, e, even + e * Block);
            // Descending: row e reads rows a <= e only, so rows above e may be overwritten.
            for (std::size_t e = rows; e-- > 0;) {
                Vec odd[Block];
                row_sums<Block, true>(pieces + u, g + u, piece, e, odd);
                for (int t = 0; t < Block; ++t) pieces[(e * Block + t) * piece + u] = reduce<true>(even[e * Block + t], odd[t]);
            }
        }
    }

    // 64-bit sums of the even (or odd) lanes of row e's Block positions: sum over row pairs (a, b)
    // of the truncated product of the Block values of f row a and g row b.
    template <int Block, bool Odd>
    void row_sums(const Vec* f, const Vec* g, std::size_t piece, std::size_t e, Vec* out) const {
        constexpr int kFold = 15 / Block;  // row pairs between folds: at most 15 products per sum
        const Vec low = _mm256_set1_epi64x(0xFFFFFFFF), fold = _mm256_set1_epi64x(k2To32);
        Vec sum[Block];
#pragma GCC unroll 8
        for (int t = 0; t < Block; ++t) sum[t] = _mm256_setzero_si256();
        for (u32 i = pair_start_[e], since = 0; i < pair_start_[e + 1]; ++i) {
            const Vec* x = f + pairs_[i] * Block * piece;
            const Vec* y = g + (e - pairs_[i]) * Block * piece;
            Vec fx[Block], gy[Block];
#pragma GCC unroll 8
            for (int j = 0; j < Block; ++j) {
                fx[j] = x[j * piece], gy[j] = y[j * piece];
                if constexpr (Odd) fx[j] = _mm256_srli_epi64(fx[j], 32), gy[j] = _mm256_srli_epi64(gy[j], 32);
            }
#pragma GCC unroll 8
            for (int t = 0; t < Block; ++t)
#pragma GCC unroll 8
                for (int j = 0; j <= t; ++j) sum[t] = _mm256_add_epi64(sum[t], _mm256_mul_epu32(fx[j], gy[t - j]));
            if (++since == kFold) {  // products < P^2 < 2^60; a folded sum is < 2^61
                since = 0;
#pragma GCC unroll 8
                for (int t = 0; t < Block; ++t)
                    sum[t] = _mm256_add_epi64(_mm256_and_si256(sum[t], low), _mm256_mul_epu32(_mm256_srli_epi64(sum[t], 32), fold));
            }
        }
#pragma GCC unroll 8
        for (int t = 0; t < Block; ++t) out[t] = sum[t];
    }

    std::vector<u32> n_, order_;  // order_: outer variables, lowest digit first
    std::vector<bool> outer_;
    std::size_t q_ = 1, inner_ = 1, stride_ = 8, block_ = 1;
    u32 m_ = 1;
    std::uint8_t* grade_ = nullptr;
    std::vector<u32> pairs_, pair_start_;
};

// ---------------------------------------------------------------------------------------------
// Ranked method. Variable l is evaluated at n_l points: {0, 1} for n_l = 2, {0, 1, -1} for n_l = 3;
// a coefficient of total degree r carries t^r, products are truncated at t^R, R = sum (n_l - 1).
// Evaluation reduces x^d with d >= n_l to lower powers, so a term that wrapped sits at a rank above
// its position's degree and is never read. Ranks of a point p are at most cap(p), the sum of n_l - 1
// over its nonzero coordinates; position s reads only points whose nonzero coordinates are nonzero
// in s, so rank r is needed only where r >= nz(p), the count of nonzero coordinates.
//
// Digit order: three variables of size 2 first (the 8 lanes of a vector, evaluated in registers),
// then the others in input order. The lowest b variables (Nb positions) form a block whose R + 1
// rank planes for f and g fit L2. For each point t of the top variables, the block is evaluated
// at t directly from the rows of f and g it depends on, multiplied, and interpolated back into
// the rows of c that depend on t.

constexpr int kMaxRank = 24;  // R <= 21 for N <= 2^18 with three n_l = 2 (3^9 * 8)

// Lane v of x[popcount(v)]: lanes by popcount are {0}, {1, 2, 4}, {3, 5, 6}, {7}.
inline Vec by_popcount(const Vec* x) {
    const Vec low = _mm256_blend_epi32(x[0], x[1], 0x16);
    return _mm256_blend_epi32(_mm256_blend_epi32(low, x[2], 0x68), x[3], 0x80);
}

// Lane v of the result: sum (or alternating sum) of lanes w subset of v.
template <bool Inverse>
Vec lanes_transform(Vec x) {
    const auto step = [](Vec x, Vec t) { return Inverse ? sub_mod(x, t) : add_mod(x, t); };
    x = step(x, _mm256_slli_epi64(x, 32));                // lane v += lane v - 1 for odd v
    x = step(x, _mm256_slli_si256(x, 8));                 // lanes 2, 3 (and 6, 7) += lanes 0, 1 (4, 5)
    return step(x, _mm256_permute2x128_si256(x, x, 0x08));  // lanes 4-7 += lanes 0-3
}

// Evaluation (or interpolation) along one variable of size N on v[0], v[stride], v[2 stride].
// Inverse for N = 3 returns twice the coefficients.
template <bool Inverse, int N>
void evaluate_line(Vec* v, int stride) {
    Vec &a0 = v[0], &a1 = v[stride];
    if constexpr (N == 2) {
        a1 = Inverse ? sub_mod(a1, a0) : add_mod(a1, a0);
    } else if constexpr (!Inverse) {  // a0, a0 + a1 + a2, a0 - a1 + a2
        Vec& a2 = v[2 * stride];
        const Vec s = add_mod(a0, a2);
        a2 = sub_mod(s, a1), a1 = add_mod(s, a1);
    } else {  // 2 a0 = 2 F0, 2 a1 = F1 - F2, 2 a2 = F1 + F2 - 2 F0
        Vec& a2 = v[2 * stride];
        const Vec f0 = add_mod(a0, a0), f1 = a1, f2 = a2;
        a0 = f0, a1 = sub_mod(f1, f2), a2 = sub_mod(add_mod(f1, f2), f0);
    }
}

// Evaluation (or interpolation) along two variables of sizes A and B (B = 1: one variable) on the
// group p[r + i sa + j sb], i < A, j < B, for ranks r in [first, last].
template <bool Inverse, int A, int B>
void variable_step(Vec* p, std::size_t sa, std::size_t sb, int first, int last) {
    for (int r = first; r <= last; ++r) {
        Vec v[A * B];
#pragma GCC unroll 9
        for (int j = 0; j < B; ++j)
#pragma GCC unroll 3
            for (int i = 0; i < A; ++i) v[i + A * j] = p[r + i * sa + j * sb];
#pragma GCC unroll 3
        for (int j = 0; j < B; ++j) evaluate_line<Inverse, A>(v + A * j, 1);
        if constexpr (B > 1)
#pragma GCC unroll 3
            for (int i = 0; i < A; ++i) evaluate_line<Inverse, B>(v + i, A);
#pragma GCC unroll 9
        for (int j = 0; j < B; ++j)
#pragma GCC unroll 3
            for (int i = 0; i < A; ++i) p[r + i * sa + j * sb] = v[i + A * j];
    }
}

// dst +-= src over `count` vectors.
void add_row(Vec* dst, const Vec* src, std::size_t count, bool negative) {
    if (negative)
        for (std::size_t i = 0; i < count; ++i) dst[i] = sub_mod(dst[i], load(src + i));
    else
        for (std::size_t i = 0; i < count; ++i) dst[i] = add_mod(dst[i], load(src + i));
}

// One row of a top evaluation or interpolation: row index, sign, total degree of the row's digits.
struct Term {
    u32 row;
    bool negative;
    std::uint8_t rank;
};

class Ranked {
public:
    static bool fits(const std::vector<u32>& n) {
        int rank = 0;
        for (u32 x : n) rank += int(x) - 1;
        return std::ranges::count(n, 2u) >= 3 && std::ranges::all_of(n, [](u32 x) { return x <= 3; }) &&
               rank < kMaxRank;
    }

    Ranked(const std::vector<u32>& n, std::size_t size) : size_(size) {
        const std::size_t k = n.size();
        std::vector<u32> order;
        for (std::size_t l = 0; l < k && order.size() < 3; ++l)
            if (n[l] == 2) order.push_back(u32(l));
        for (std::size_t l = 0; l < k; ++l)
            if (std::ranges::find(order, u32(l)) == order.end()) order.push_back(u32(l));
        for (u32 l : order) size_of_.push_back(n[l]);
        for (u32 x : n) rank_ += int(x) - 1, threes_ += x == 3;

        bottom_ = 3, bottom_size_ = 8;
        while (bottom_ < k && 8 * std::size_t(rank_ + 1) * bottom_size_ * size_of_[bottom_] <= BLOCK_BYTES)
            bottom_size_ *= size_of_[bottom_++];
        top_size_ = size_ / bottom_size_;
        int top_rank = 0;
        for (std::size_t l = bottom_; l < k; ++l) top_rank += int(size_of_[l]) - 1;
        planes_ = top_rank + 1;

        // Position in our order -> input index, unless the orders agree.
        if (!std::ranges::is_sorted(order)) {
            std::vector<u32> stride(k);
            for (std::size_t l = 0, s = 1; l < k; s *= n[l], ++l) stride[l] = u32(s);
            origin_.resize(size_);
            std::vector<u32> digit(k, 0);
            u32 origin = 0;
            for (std::size_t i = 0; i < size_; ++i) {
                origin_[i] = origin;
                for (std::size_t j = 0; j < k; ++j) {
                    const u32 v = order[j];
                    if (++digit[j] < n[v]) {
                        origin += stride[v];
                        break;
                    }
                    digit[j] = 0, origin -= (n[v] - 1) * stride[v];
                }
            }
        }
        describe_vectors();
        describe_ranges();
        describe_top();
    }

    void multiply(const u32* f, const u32* g, u32* c) {
        const u32* fo = to_our_order(f);
        const u32* go = to_our_order(g);
        u32* co = origin_.empty() ? c : allocate<u32>(size_);
        const std::size_t vectors = bottom_size_ / 8, plane = std::size_t(planes_) * vectors;
        Vec* top_f = allocate<Vec>(plane);
        Vec* top_g = allocate<Vec>(plane);
        Vec* a = allocate<Vec>(std::size_t(rank_ + 1) * vectors);
        Vec* b = allocate<Vec>(std::size_t(rank_ + 1) * vectors);
        for (std::size_t t = 0; t < top_size_; ++t) {
            evaluate_top(fo, t, top_f);
            evaluate_top(go, t, top_g);
            if (const int z = top_zeros3_[t]) {  // the interpolation below takes 2 F0 for each
                const Vec factor = all(montgomery(power_mod(2, z)));
                for (std::size_t i = 0; i < plane; ++i) top_f[i] = montgomery_multiply(top_f[i], factor);
            }
            spread(top_f, a);
            spread(top_g, b);
            bottom_transform<false>(a, top_cap_[t]);
            bottom_transform<false>(b, top_cap_[t]);
            pointwise(a, b, top_cap_[t], top_nz_[t]);
            bottom_transform<true>(a, top_cap_[t]);
            gather(a, top_f);
            interpolate_top(top_f, t, co);
        }
        // Scale: the products carry 2^-32, interpolation along each variable of size 3 a factor 2.
        const Vec scale = all(montgomery(multiply_mod(power_mod(2, 32), power_mod(inverse_mod(2), threes_))));
        for (std::size_t i = 0; i < size_; i += 8) store(co + i, montgomery_multiply(load(co + i), scale));
        if (!origin_.empty())
            for (std::size_t i = 0; i < size_; ++i) c[origin_[i]] = co[i];
    }

private:
    struct Range {
        std::uint8_t first, span;
    };

    const u32* to_our_order(const u32* x) const {
        if (origin_.empty()) return x;
        u32* y = allocate<u32>(size_);
        for (std::size_t i = 0; i < size_; ++i) y[i] = x[origin_[i]];
        return y;
    }

    // Digit sum (coefficient side), cap and nz (point side) of each vector of the block.
    void describe_vectors() {
        const std::size_t count = bottom_size_ / 8;
        vector_sum_.resize(count), vector_cap_.resize(count), vector_nz_.resize(count);
        std::vector<u32> digit(bottom_, 0);
        int sum = 0, cap = 0, nz = 0;
        for (std::size_t u = 0; u < count; ++u) {
            vector_sum_[u] = std::uint8_t(sum), vector_cap_[u] = std::uint8_t(cap), vector_nz_[u] = std::uint8_t(nz);
            pointwise_order_.push_back(u32(u));
            for (std::size_t l = 3; l < bottom_; ++l) {
                const int n = int(size_of_[l]);
                if (++digit[l] < u32(n)) {
                    ++sum;
                    if (digit[l] == 1) cap += n - 1, ++nz;
                    break;
                }
                digit[l] = 0, sum -= n - 1, cap -= n - 1, --nz;
            }
        }
        std::ranges::stable_sort(pointwise_order_, {}, [this](u32 u) { return vector_cap_[u] * 64 + vector_nz_[u]; });
    }

    // Steps of the block transforms: variables l, l + 1 together (the inverse only if both have
    // size 2: a pair's rank range is the union of its positions' ranges, wider for larger sizes).
    // Rank ranges of the step along variables l..e at group base u (digits l..e zero).
    // Forward: positions are points in variables below l, digits from l on; the group's nonzero
    // ranks are [first, first + span + top cap + 3] with first the digit sum above e and span the
    // sum of n - 1 over l..e and over nonzero point coordinates below l (the lanes add up to 3).
    // Inverse: positions are digits below l, points from l on; the group's ranks read later are
    // [first, first + span + planes + 2] with first the digit sum below l and span the sum of
    // n - 1 from l on (output ranks of position d: sum(d) plus top rank plus lanes, < planes + 3).
    void describe_ranges() {
        for (const bool inverse : {false, true}) {
            Pass& pass = inverse ? inverse_ : forward_;
            for (std::size_t l = 3; l < bottom_;) {
                const bool pair = l + 1 < bottom_ && (!inverse || (size_of_[l] == 2 && size_of_[l + 1] == 2));
                pass.steps.push_back({u32(l), pair ? 2u : 1u});
                l += pair ? 2 : 1;
            }
            const std::size_t count = bottom_size_ / 8;
            pass.ranges.resize(pass.steps.size() * count);
            std::vector<u32> digit(bottom_, 0);
            for (std::size_t u = 0; u < count; ++u) {
                for (std::size_t k = 0; k < pass.steps.size(); ++k) {
                    const std::size_t l = pass.steps[k].first, e = l + pass.steps[k].width - 1;
                    int above = 0, below = 0, cap = 0, rest = 0, own = 0;
                    for (std::size_t m = 3; m < bottom_; ++m) {
                        const int d = int(digit[m]), n = int(size_of_[m]);
                        if (m > e) above += d;
                        if (m < l) below += d, cap += d ? n - 1 : 0;
                        if (m >= l) rest += n - 1;
                        if (m >= l && m <= e) own += n - 1;
                    }
                    pass.ranges[k * count + u] = inverse ? Range{std::uint8_t(below), std::uint8_t(rest)}
                                                         : Range{std::uint8_t(above), std::uint8_t(own + cap)};
                }
                for (std::size_t l = 3; l < bottom_ && ++digit[l] == size_of_[l]; ++l) digit[l] = 0;
            }
        }
    }

    // For each top point t: the rows of f its evaluation reads, the rows of c its interpolation
    // writes (interpolation as 2 a0 = 2 F0, 2 a1 = F1 - F2, 2 a2 = F1 + F2 - 2 F0 for size 3, with
    // 2 F0 folded into the evaluation of f), its cap and nz.
    void describe_top() {
        struct Entry {
            u32 digit;
            bool negative;
        };
        static constexpr Entry kEvaluate[2][3][3] = {  // [size 3][point][entry]
            {{{0, false}}, {{0, false}, {1, false}}, {}},
            {{{0, false}}, {{0, false}, {1, false}, {2, false}}, {{0, false}, {1, true}, {2, false}}}};
        static constexpr Entry kInterpolate[2][3][3] = {
            {{{0, false}, {1, true}}, {{1, false}}, {}},
            {{{0, false}, {2, true}}, {{1, false}, {2, false}}, {{1, true}, {2, false}}}};
        static constexpr int kEvaluateCount[2][3] = {{1, 2, 0}, {1, 3, 3}};
        static constexpr int kInterpolateCount[2][3] = {{2, 1, 0}, {2, 2, 2}};

        const std::size_t k = size_of_.size();
        evaluate_start_.push_back(0), interpolate_start_.push_back(0);
        std::vector<u32> point(k, 0);
        for (std::size_t t = 0; t < top_size_; ++t) {
            std::vector<Term> evaluate{{0, false, 0}}, interpolate{{0, false, 0}};
            int cap = 0, nz = 0, zeros3 = 0;
            u32 stride = 1;
            for (std::size_t l = bottom_; l < k; stride *= size_of_[l++]) {
                const u32 p = point[l];
                const bool three = size_of_[l] == 3;
                if (p) cap += int(size_of_[l]) - 1, ++nz;
                zeros3 += three && p == 0;
                const auto extend = [&](std::vector<Term>& terms, const Entry* table, int count) {
                    std::vector<Term> next;
                    for (const Term& x : terms)
                        for (int e = 0; e < count; ++e) {
                            const u32 d = table[e].digit;
                            next.push_back({x.row + d * stride, x.negative != table[e].negative, std::uint8_t(x.rank + d)});
                        }
                    terms = std::move(next);
                };
                extend(evaluate, kEvaluate[three][p], kEvaluateCount[three][p]);
                extend(interpolate, kInterpolate[three][p], kInterpolateCount[three][p]);
            }
            terms_.insert(terms_.end(), evaluate.begin(), evaluate.end());
            evaluate_start_.push_back(u32(terms_.size()));
            inverse_terms_.insert(inverse_terms_.end(), interpolate.begin(), interpolate.end());
            interpolate_start_.push_back(u32(inverse_terms_.size()));
            top_cap_.push_back(std::uint8_t(cap)), top_nz_.push_back(std::uint8_t(nz));
            top_zeros3_.push_back(std::uint8_t(zeros3));
            for (std::size_t l = bottom_; l < k && ++point[l] == size_of_[l]; ++l) point[l] = 0;
        }
    }

    // Planes [top rank][Nb / 8] of x (our order) evaluated at top point t.
    void evaluate_top(const u32* x, std::size_t t, Vec* planes) const {
        const std::size_t vectors = bottom_size_ / 8;
        std::memset(static_cast<void*>(planes), 0, std::size_t(planes_) * vectors * sizeof(Vec));
        for (u32 i = evaluate_start_[t]; i < evaluate_start_[t + 1]; ++i) {
            const Term& term = terms_[i];
            add_row(planes + term.rank * vectors, reinterpret_cast<const Vec*>(x + term.row * bottom_size_), vectors,
                    term.negative);
        }
    }

    // Adds the planes of top point t to the rows of c (our order) that depend on it.
    void interpolate_top(const Vec* planes, std::size_t t, u32* c) const {
        const std::size_t vectors = bottom_size_ / 8;
        for (u32 i = interpolate_start_[t]; i < interpolate_start_[t + 1]; ++i) {
            const Term& term = inverse_terms_[i];
            Vec* row = reinterpret_cast<Vec*>(c + term.row * bottom_size_);
            const Vec* src = planes + term.rank * vectors;
            if (term.negative)
                for (std::size_t j = 0; j < vectors; ++j) store(row + j, sub_mod(load(row + j), src[j]));
            else
                for (std::size_t j = 0; j < vectors; ++j) store(row + j, add_mod(load(row + j), src[j]));
        }
    }

    // Along the block's variables above the lanes, by steps, on the ranks that can be nonzero
    // (forward) or are read later (inverse): see describe_ranges().
    template <bool Inverse>
    void bottom_transform(Vec* x, int top_cap) const {
        const int extra = Inverse ? planes_ + 2 : top_cap + 3;
        const std::size_t count = bottom_size_ / 8;
        const Pass& pass = Inverse ? inverse_ : forward_;
        for (std::size_t k = 0, step = 1; k < pass.steps.size(); ++k) {
            const Range* range = pass.ranges.data() + k * count;
            const std::size_t l = pass.steps[k].first;
            const u32 a = size_of_[l], b = pass.steps[k].width == 2 ? size_of_[l + 1] : 1;
            switch (a * 4 + b) {
            case 9: block_step<Inverse, 2, 1>(x, step, range, extra); break;
            case 13: block_step<Inverse, 3, 1>(x, step, range, extra); break;
            case 10: block_step<Inverse, 2, 2>(x, step, range, extra); break;
            case 11: block_step<Inverse, 2, 3>(x, step, range, extra); break;
            case 14: block_step<Inverse, 3, 2>(x, step, range, extra); break;
            default: block_step<Inverse, 3, 3>(x, step, range, extra); break;
            }
            step *= a * b;
        }
    }

    // One step along block variables of sizes A and B (B = 1: one variable); step: the position
    // stride of the first.
    template <bool Inverse, int A, int B>
    void block_step(Vec* x, std::size_t step, const Range* range, int extra) const {
        const std::size_t ranks = std::size_t(rank_ + 1), count = bottom_size_ / 8;
        for (std::size_t base = 0; base < count; base += step * A * B)
            for (std::size_t u = base; u < base + step; ++u)
                variable_step<Inverse, A, B>(x + u * ranks, step * ranks, step * A * ranks, range[u].first,
                                             std::min(rank_, range[u].first + range[u].span + extra));
    }

    // Top-rank planes -> ranks [Nb / 8][R + 1] with the lane variables evaluated: rank r of
    // position 8u + v is top rank r - sum(u) - popcount(v).
    void spread(const Vec* planes, Vec* out) const {
        const std::size_t vectors = bottom_size_ / 8;
        const Vec zero = _mm256_setzero_si256();
        Vec p[kMaxRank + 8];  // p[3 + rho]: plane rho; zero around
        for (int i = 0; i < 3; ++i) p[i] = p[3 + planes_ + i] = zero;
        for (std::size_t u = 0; u < vectors; ++u) {
            for (int rho = 0; rho < planes_; ++rho) p[3 + rho] = planes[rho * vectors + u];
            const int d = vector_sum_[u], last = std::min(rank_, d + planes_ + 2);
            Vec* ranks = out + u * (rank_ + 1);
            for (int r = 0; r < d; ++r) ranks[r] = zero;
            for (int r = d; r <= last; ++r) {  // lane v: plane r - d - popcount(v)
                const Vec* q = p + 3 + r - d;
                const Vec x[4] = {q[0], q[-1], q[-2], q[-3]};
                ranks[r] = lanes_transform<false>(by_popcount(x));
            }
            for (int r = last + 1; r <= rank_; ++r) ranks[r] = zero;
        }
    }

    // Inverse of spread for the ranks that positions read: top-rank planes from rank planes.
    void gather(const Vec* in, Vec* planes) const {
        const std::size_t vectors = bottom_size_ / 8;
        Vec m[kMaxRank + 4];
        for (std::size_t u = 0; u < vectors; ++u) {
            const int d = vector_sum_[u], last = std::min(rank_, d + planes_ + 2);
            for (int r = d; r <= last; ++r) m[r - d] = lanes_transform<true>(in[u * (rank_ + 1) + r]);
            for (int k = last - d + 1; k < planes_ + 3; ++k) m[k] = _mm256_setzero_si256();  // ranks above R
            for (int rho = 0; rho < planes_; ++rho) planes[rho * vectors + u] = by_popcount(m + rho);
        }
    }

    // a[r] <- sum_{i + j = r} a[i] b[j] (times 2^-32) for the ranks some position reads.
    void pointwise(Vec* a, const Vec* b, int top_cap, int top_nz) const {
        const Vec zero = _mm256_setzero_si256();
        Vec ae[kMaxRank], ao[kMaxRank], be[kMaxRank + 2 * kPad], bo[kMaxRank + 2 * kPad];
        for (int i = 0; i < kPad; ++i) be[i] = bo[i] = zero;
        for (const u32 u : pointwise_order_) {
            const int cap = std::min(rank_, top_cap + vector_cap_[u] + 3);
            const int first = top_nz + vector_nz_[u], last = std::min(rank_, 2 * cap);
            Vec* x = a + u * (rank_ + 1);
            const Vec* y = b + u * (rank_ + 1);
            for (int i = 0; i <= cap; ++i) {
                ae[i] = x[i], ao[i] = _mm256_srli_epi64(ae[i], 32);
                be[kPad + i] = y[i], bo[kPad + i] = _mm256_srli_epi64(y[i], 32);
            }
            for (int i = cap + 1; i <= cap + kPad; ++i) be[kPad + i] = bo[kPad + i] = zero;
            for (int r = first; r <= last; r += kBlock) {
                Vec even[kBlock], odd[kBlock];
                block_sums(ae, be + kPad, r, cap, even);
                block_sums(ao, bo + kPad, r, cap, odd);
                for (int k = 0; k < kBlock && r + k <= last; ++k) x[r + k] = reduce<true>(even[k], odd[k]);
            }
        }
    }

    // sum[k] = sum_i x[i] y[r + k - i] (64-bit lanes 0, 2, 4, 6) for k < kBlock, i <= cap; y is
    // zero outside [0, cap] for kPad entries on each side.
    static constexpr int kBlock = 4, kPad = kBlock - 1;
    static void block_sums(const Vec* x, const Vec* y, int r, int cap, Vec* sum) {
        const Vec low = _mm256_set1_epi64x(0xFFFFFFFF), fold = _mm256_set1_epi64x(k2To32);
        Vec s[kBlock];
#pragma GCC unroll 4
        for (int k = 0; k < kBlock; ++k) s[k] = _mm256_setzero_si256();
        for (int i = std::max(0, r - cap), end = std::min(r + kPad, cap), terms = 0; i <= end; ++i) {
            if (++terms == 16) {  // at most 15 products < P^2 per sum; a folded sum is < 2^61
                terms = 1;
#pragma GCC unroll 4
                for (int k = 0; k < kBlock; ++k)
                    s[k] = _mm256_add_epi64(_mm256_and_si256(s[k], low), _mm256_mul_epu32(_mm256_srli_epi64(s[k], 32), fold));
            }
            const Vec* yj = y + r - i;
#pragma GCC unroll 4
            for (int k = 0; k < kBlock; ++k) s[k] = _mm256_add_epi64(s[k], _mm256_mul_epu32(x[i], yj[k]));
        }
#pragma GCC unroll 4
        for (int k = 0; k < kBlock; ++k) sum[k] = s[k];
    }

    std::size_t size_, bottom_size_ = 0, top_size_ = 0, bottom_ = 0;
    int rank_ = 0, planes_ = 0, threes_ = 0;
    std::vector<u32> size_of_, origin_;
    std::vector<std::uint8_t> vector_sum_, vector_cap_, vector_nz_;
    std::vector<u32> pointwise_order_;  // vectors by (cap, nz): equal loop bounds in a row
    struct Step {
        u32 first, width;  // variables first, ..., first + width - 1
    };
    struct Pass {
        std::vector<Step> steps;
        std::vector<Range> ranges;  // [step * Nb / 8 + u]
    };
    Pass forward_, inverse_;  // block transforms
    std::vector<std::uint8_t> top_cap_, top_nz_, top_zeros3_;
    std::vector<Term> terms_, inverse_terms_;
    std::vector<u32> evaluate_start_, interpolate_start_;
};

// ---------------------------------------------------------------------------------------------

// f g mod (x_1^n_1, ..., x_k^n_k); f may be overwritten.
const u32* multiply(const std::vector<u32>& n, std::size_t size, u32* f, const u32* g) {
    u32* c = nullptr;
#if !defined(FORCE_GRADED) && !defined(FORCE_SPLIT)
    if (Ranked::fits(n)) {
        Ranked(n, size).multiply(f, g, c = allocate<u32>(round8(size)));
        return c;
    }
#endif
#ifndef FORCE_GRADED
    const Plan plan = plan_split(n, size);
#ifdef FORCE_SPLIT
    if (plan.cost < 1e300) {
#else
    if (plan.cost < graded_cost(n, size)) {
#endif
        Split(n, size, plan.outer).multiply(f, g, f);  // reads f before writing the product
        return f;
    }
#endif
    graded(n, size, f, g, c = allocate<u32>(round8(size)));
    return c;
}

void solve() {
    io::Reader in;
    const auto k = in.read<u32>();
    std::vector<u32> n(k);
    std::size_t size = 1;
    for (auto& x : n) size *= x = in.read<u32>();
    const std::size_t padded = round8(size);
    u32* f = allocate<u32>(padded);
    u32* g = allocate<u32>(padded);
    in.read(f, size);
    in.read(g, size);
    const u32* c = multiply(n, size, f, g);
    io::Writer out;
    fixed_width::write(out, c, size);
}

#ifdef __ELF__
// Runs from .preinit_array, before libstdc++ initializes iostreams and locales; _exit skips teardown.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
