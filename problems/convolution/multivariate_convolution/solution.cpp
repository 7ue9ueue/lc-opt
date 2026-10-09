// Truncated multivariate product mod 998244353: c = f g mod (x_1^n_1, ..., x_k^n_k), N = prod n_l.
// Two methods (notes.md):
// - Ranked: every n_l <= 3 and at least three n_l = 2. Each variable is evaluated at n_l points
//   ({0, 1} or {0, 1, -1}); a rank t^|d| tracks the total degree, so terms that wrapped are dropped.
// - Graded: any shape. One cyclic NTT over the flat index, graded by chi(i) = sum_j floor(i / P_j)
//   mod m; a carry adds 1 to chi, so a product term is valid iff no grade was lost. m >= k, m | P - 1.
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

// Zero-filled memory in huge pages where the kernel allows; never freed (the process ends with _exit).
template <class T>
T* allocate(std::size_t count) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    const std::size_t bytes = (count * sizeof(T) + 64 + kHuge - 1) / kHuge * kHuge;
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) std::abort();
#ifdef MADV_HUGEPAGE
    if (bytes >= kHuge) ::madvise(p, bytes, MADV_HUGEPAGE);
#endif
    return static_cast<T*>(p);
}

std::size_t round8(std::size_t n) { return (n + 7) & ~std::size_t(7); }

// ---------------------------------------------------------------------------------------------
// Graded method. A carry out of digit j < k - 1 adds 1 to chi(i + j) - chi(i) - chi(j), and the
// linear product (length >= 2N - 1) has no carry out of the top digit, so at most k - 1 carries:
// grades mod m >= k separate them. t -> w^s for an m-th root w turns each grade class into one
// ordinary convolution: c_i = (1/m) sum_s w^(-s chi_i) [(f w^(s chi)) * (g w^(s chi))]_i.

u32 grade_modulus(std::size_t k) {
    for (u32 m : {1, 2, 4, 7, 8, 14, 16, 17, 28, 32, 56, 64})
        if (m >= k) return m;
    std::abort();
}

void graded(const std::vector<u32>& n, std::size_t size, const u32* f, const u32* g, u32* c) {
    const std::size_t k = n.size(), padded = round8(size);
    const u32 m = grade_modulus(k);
    const u32 root = power_mod(3, (kP - 1) / m), inverse_root = inverse_mod(root);

    // base_i = w^chi_i, inverse_base_i = w^-chi_i, in Montgomery form.
    std::array<u32, 64> powers{}, inverse_powers{};
    for (u32 r = 0, x = 1, y = 1; r < m; ++r, x = multiply_mod(x, root), y = multiply_mod(y, inverse_root))
        powers[r] = montgomery(x), inverse_powers[r] = montgomery(y);
    u32* base = allocate<u32>(padded);
    u32* inverse_base = allocate<u32>(padded);
    {
        std::vector<u32> digit(k + 1, 0);
        u32 chi = 0;  // mod m
        for (std::size_t i = 0; i < size; ++i) {
            base[i] = powers[chi], inverse_base[i] = inverse_powers[chi];
            for (std::size_t j = 0; j + 1 < k && ++digit[j] == n[j]; ++j) {
                digit[j] = 0;
                if (++chi == m) chi = 0;
            }
        }
    }

    ntt::Convolution product(size, size);
    // Both factors fill at most half the transform, so lib/ntt reads only that half ("sparse").
    const std::size_t half = (std::size_t(1) << std::max(6, int(std::bit_width(2 * size - 2)))) / 2;
    u32 *a = product.a(), *b = product.b();
    const Vec scale = all(montgomery(inverse_mod(m)));
    u32* weight = allocate<u32>(padded);          // w^(s chi), Montgomery form
    u32* inverse_weight = allocate<u32>(padded);  // w^(-s chi) / m, Montgomery form
    for (u32 s = 0; s < m; ++s) {
        if (s == 0) {
            std::memcpy(a, f, size * sizeof(u32));
            std::memcpy(b, g, size * sizeof(u32));
        } else {
            for (std::size_t i = 0; i < padded; i += 8) {
                Vec w = load(base + i), iw = load(inverse_base + i);
                if (s == 1) {
                    iw = montgomery_multiply(iw, scale);
                } else {
                    w = montgomery_multiply(w, load(weight + i));
                    iw = montgomery_multiply(iw, load(inverse_weight + i));
                }
                store(weight + i, w), store(inverse_weight + i, iw);
                store(a + i, montgomery_multiply(load(f + i), w));
                store(b + i, montgomery_multiply(load(g + i), w));
            }
        }
        std::memset(a + size, 0, (half - size) * sizeof(u32));
        std::memset(b + size, 0, (half - size) * sizeof(u32));
        const u32* h = product.multiply();  // rebuilds its tables; reads only a() and b()
        for (std::size_t i = 0; i < padded; i += 8) {
            const Vec x = load(h + i);
            if (s == 0)
                store(c + i, montgomery_multiply(x, scale));
            else
                store(c + i, add_mod(load(c + i), montgomery_multiply(x, load(inverse_weight + i))));
        }
    }
}

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

alignas(32) constexpr u32 kPopcountMask[4][8] = {
    {~0u, 0, 0, 0, 0, 0, 0, 0},
    {0, ~0u, ~0u, 0, ~0u, 0, 0, 0},
    {0, 0, 0, ~0u, 0, ~0u, ~0u, 0},
    {0, 0, 0, 0, 0, 0, 0, ~0u},
};

// Lane v of the result: sum (or alternating sum) of lanes w subset of v.
template <bool Inverse>
Vec lanes_transform(Vec x) {
    const auto step = [](Vec x, Vec t) { return Inverse ? sub_mod(x, t) : add_mod(x, t); };
    const Vec zero = _mm256_setzero_si256();
    x = step(x, _mm256_blend_epi32(zero, _mm256_shuffle_epi32(x, 0xA0), 0xAA));
    x = step(x, _mm256_blend_epi32(zero, _mm256_shuffle_epi32(x, 0x44), 0xCC));
    return step(x, _mm256_permute2x128_si256(x, x, 0x08));
}

// Evaluation (or interpolation) along one variable of size n at vector stride `stride`.
// Inverse for n = 3 returns twice the coefficients.
template <bool Inverse>
void variable_transform(Vec* x, std::size_t count, u32 n, std::size_t stride) {
    for (std::size_t base = 0; base < count; base += n * stride) {
        Vec* p = x + base;
        for (std::size_t j = 0; j < stride; ++j) {
            if (n == 2) {
                p[j + stride] = Inverse ? sub_mod(p[j + stride], p[j]) : add_mod(p[j + stride], p[j]);
            } else if (!Inverse) {  // a0, a0 + a1 + a2, a0 - a1 + a2
                const Vec s = add_mod(p[j], p[j + 2 * stride]), a1 = p[j + stride];
                p[j + stride] = add_mod(s, a1), p[j + 2 * stride] = sub_mod(s, a1);
            } else {  // 2 a0 = 2 F0, 2 a1 = F1 - F2, 2 a2 = F1 + F2 - 2 F0
                const Vec f0 = add_mod(p[j], p[j]), f1 = p[j + stride], f2 = p[j + 2 * stride];
                p[j] = f0, p[j + stride] = sub_mod(f1, f2), p[j + 2 * stride] = sub_mod(add_mod(f1, f2), f0);
            }
        }
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
            for (int r = 0; r <= rank_; ++r) {
                bottom_transform<false>(a + r * vectors);
                bottom_transform<false>(b + r * vectors);
            }
            pointwise(a, b, top_cap_[t], top_nz_[t]);
            for (int r = 0; r <= rank_; ++r) bottom_transform<true>(a + r * vectors);
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

    // Planes [top rank][Nb] of x (our order) evaluated at top point t.
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

    template <bool Inverse>
    void bottom_transform(Vec* x) const {
        const std::size_t count = bottom_size_ / 8;
        for (std::size_t l = 3, stride = 1; l < bottom_; stride *= size_of_[l++])
            variable_transform<Inverse>(x, count, size_of_[l], stride);
    }

    // Top-rank planes -> rank planes [R + 1][Nb / 8] with the lane variables evaluated: rank r of
    // position 8u + v is top rank r - sum(u) - popcount(v).
    void spread(const Vec* planes, Vec* out) const {
        const std::size_t vectors = bottom_size_ / 8;
        const Vec mask[4] = {load(kPopcountMask[0]), load(kPopcountMask[1]), load(kPopcountMask[2]),
                             load(kPopcountMask[3])};
        const Vec zero = _mm256_setzero_si256();
        for (std::size_t u = 0; u < vectors; ++u) {
            Vec x[kMaxRank + 4];
            for (int r = 0; r < planes_ + 3; ++r) x[r] = zero;
            for (int rho = 0; rho < planes_; ++rho) {
                const Vec v = planes[rho * vectors + u];
                for (int j = 0; j < 4; ++j) x[rho + j] = _mm256_or_si256(x[rho + j], _mm256_and_si256(v, mask[j]));
            }
            const int d = vector_sum_[u], last = std::min(rank_, d + planes_ + 2);
            for (int r = 0; r < d; ++r) out[r * vectors + u] = zero;
            for (int r = d; r <= last; ++r) out[r * vectors + u] = lanes_transform<false>(x[r - d]);
            for (int r = last + 1; r <= rank_; ++r) out[r * vectors + u] = zero;
        }
    }

    // Inverse of spread for the ranks that positions read: top-rank planes from rank planes.
    void gather(const Vec* in, Vec* planes) const {
        const std::size_t vectors = bottom_size_ / 8;
        const Vec mask[4] = {load(kPopcountMask[0]), load(kPopcountMask[1]), load(kPopcountMask[2]),
                             load(kPopcountMask[3])};
        for (std::size_t u = 0; u < vectors; ++u) {
            const int d = vector_sum_[u], last = std::min(rank_, d + planes_ + 2);
            Vec m[kMaxRank + 4];
            for (int r = d; r <= last; ++r) m[r - d] = lanes_transform<true>(in[r * vectors + u]);
            for (int rho = 0; rho < planes_; ++rho) {
                Vec x = _mm256_setzero_si256();
                for (int j = 0; j < 4 && d + rho + j <= rank_; ++j)
                    x = _mm256_or_si256(x, _mm256_and_si256(m[rho + j], mask[j]));
                planes[rho * vectors + u] = x;
            }
        }
    }

    // a[r] <- sum_{i + j = r} a[i] b[j] (times 2^-32) for the ranks some position reads.
    void pointwise(Vec* a, const Vec* b, int top_cap, int top_nz) const {
        const std::size_t vectors = bottom_size_ / 8;
        const Vec low = _mm256_set1_epi64x(0xFFFFFFFF), fold = _mm256_set1_epi64x(k2To32);
        for (std::size_t u = 0; u < vectors; ++u) {
            const int cap = std::min(rank_, top_cap + vector_cap_[u] + 3);
            const int nz = top_nz + vector_nz_[u];
            Vec ae[kMaxRank], ao[kMaxRank], be[kMaxRank], bo[kMaxRank];
            for (int i = 0; i <= cap; ++i) {
                ae[i] = a[i * vectors + u], ao[i] = _mm256_srli_epi64(ae[i], 32);
                be[i] = b[i * vectors + u], bo[i] = _mm256_srli_epi64(be[i], 32);
            }
            for (int r = std::min(rank_, 2 * cap); r >= nz; --r) {
                Vec even = _mm256_setzero_si256(), odd = even;
                const int first = std::max(0, r - cap), last = std::min(r, cap);
                for (int i = first, terms = 0; i <= last; ++i) {
                    if (++terms == 17) {  // 16 products < 16 P^2; fold to < 2^61 before more
                        even = _mm256_add_epi64(_mm256_and_si256(even, low),
                                                _mm256_mul_epu32(_mm256_srli_epi64(even, 32), fold));
                        odd = _mm256_add_epi64(_mm256_and_si256(odd, low),
                                               _mm256_mul_epu32(_mm256_srli_epi64(odd, 32), fold));
                    }
                    even = _mm256_add_epi64(even, _mm256_mul_epu32(ae[i], be[r - i]));
                    odd = _mm256_add_epi64(odd, _mm256_mul_epu32(ao[i], bo[r - i]));
                }
                a[r * vectors + u] = reduce<true>(even, odd);
            }
        }
    }

    std::size_t size_, bottom_size_ = 0, top_size_ = 0, bottom_ = 0;
    int rank_ = 0, planes_ = 0, threes_ = 0;
    std::vector<u32> size_of_, origin_;
    std::vector<std::uint8_t> vector_sum_, vector_cap_, vector_nz_;
    std::vector<std::uint8_t> top_cap_, top_nz_, top_zeros3_;
    std::vector<Term> terms_, inverse_terms_;
    std::vector<u32> evaluate_start_, interpolate_start_;
};

// ---------------------------------------------------------------------------------------------

void solve() {
    io::Reader in;
    const auto k = in.read<u32>();
    std::vector<u32> n(k);
    std::size_t size = 1;
    for (auto& x : n) size *= x = in.read<u32>();
    const std::size_t padded = round8(size);
    u32* f = allocate<u32>(padded);
    u32* g = allocate<u32>(padded);
    u32* c = allocate<u32>(padded);
    in.read(f, size);
    in.read(g, size);
#ifndef FORCE_GRADED
    if (Ranked::fits(n))
        Ranked(n, size).multiply(f, g, c);
    else
#endif
        graded(n, size, f, g, c);
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
