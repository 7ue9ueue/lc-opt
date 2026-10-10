// Power series with few nonzero terms, modulo P = 998244353: linear recurrences with sparse taps,
// the core of inv, exp, log, pow and sqrt of a sparse series. x86-64 with AVX2.
// Log: lib/poly/notes.md.
//
//   const poly::sparse::Term taps[] = {{1, c1}, {40, c40}};  // {distance >= 1, coefficient}
//   const poly::sparse::Term rhs[] = {{0, r0}};              // {index, value}, increasing index
//   poly::sparse::Recurrence recurrence(taps, rhs);
//   recurrence.next(out, count);  // g[i] = r[i] + c1 g[i - 1] + c40 g[i - 40], next count values
//
// Values are in [0, P); g[i] = 0 for i < 0. next(out, count) writes the next count coefficients
// (rounded up to a multiple of kBlock) to out and reads the history(), the coefficients before
// them, from out[-history(), 0) (zeros before g[0]). With a whole array g whose first kPadding
// words are zero, out = g + kPadding + (coefficients solved so far) always works.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace poly::sparse {

inline constexpr std::uint32_t kModulus = 998244353;

struct Term {
    std::uint32_t index;  // a tap's distance, or an exponent
    std::uint32_t value;  // in [0, P)
};

inline std::uint32_t multiply(std::uint32_t a, std::uint32_t b) {
    return std::uint32_t(std::uint64_t(a) * b % kModulus);
}

inline std::uint32_t power(std::uint32_t x, std::uint64_t e) {
    std::uint32_t result = 1;
    for (; e; e >>= 1, x = multiply(x, x))
        if (e & 1) result = multiply(result, x);
    return result;
}

// x^-1 for x in [1, P).
inline std::uint32_t inverse(std::uint32_t x) { return power(x, kModulus - 2); }

namespace detail {

using Vec = __m256i;

inline constexpr std::uint32_t kP = kModulus;
inline constexpr std::uint32_t kR = std::uint32_t((std::uint64_t(1) << 32) % kP);  // 2^32 mod P
inline constexpr std::uint32_t kNI = 998244351;                                    // -1 / P mod 2^32
static_assert(std::uint32_t(kP * kNI) == ~std::uint32_t(0));

// Products below P^2 that reduce<false> may sum: (12 P^2 + 2^32 P) / 2^32 < 4P.
inline constexpr std::size_t kUnfolded = 12;

inline std::uint32_t montgomery_form(std::uint32_t x) { return std::uint32_t((std::uint64_t(x) << 32) % kP); }
inline Vec broadcast(std::uint32_t x) { return _mm256_set1_epi32(int(x)); }
inline Vec load(const std::uint32_t* p) { return _mm256_loadu_si256(reinterpret_cast<const Vec*>(p)); }
inline void store(std::uint32_t* p, Vec x) { _mm256_storeu_si256(reinterpret_cast<Vec*>(p), x); }
inline Vec multiply_add(Vec sum, Vec a, Vec b) { return _mm256_add_epi64(sum, _mm256_mul_epu32(a, b)); }

// Qword sums x of the even and odd values of 8 -> x / 2^32 mod P in [0, P), in order. kFold:
// any x < 2^64; x = 2^32 h + l is first folded to h (2^32 mod P) + l < 2^62. Else x < 12 P^2:
// one Montgomery step leaves x / 2^32 + P < 4P, and two subtractions follow.
template <bool kFold>
Vec reduce(Vec even, Vec odd) {
    const auto redc = [](Vec x) {  // in the high dword
        if constexpr (kFold)
            x = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), broadcast(kR)),
                                 _mm256_blend_epi32(x, _mm256_setzero_si256(), 0xAA));
        return _mm256_add_epi64(x, _mm256_mul_epu32(_mm256_mul_epu32(x, broadcast(kNI)), broadcast(kP)));
    };
    Vec r = _mm256_blend_epi32(_mm256_srli_epi64(redc(even), 32), redc(odd), 0xAA);
    if constexpr (!kFold) r = _mm256_min_epu32(r, _mm256_sub_epi32(r, broadcast(2 * kP)));
    return _mm256_min_epu32(r, _mm256_sub_epi32(r, broadcast(kP)));
}

}  // namespace detail

// g[i] = r[i] + sum over taps (d, c) of c g[i - d], solved in blocks of 16 coefficients.
//
// Taps with d < 16 ("short") enter through the state, the last w = max d values:
// g[n + t] = sum_{j < w} A[t][j] g[n - 1 - j] + sum_{s <= t} u[t - s] r'[n + s], where u is the
// impulse response of the short taps and r' is r plus the long taps' terms, which read only
// values before the block. Where no long tap reaches and r is zero, only the A part remains:
// 64 coefficients per step from one state, w products each, in four independent blocks.
//
// Products of a coefficient times 2^32 (cancelling the Montgomery factor) and a value are summed
// in 64-bit lanes and reduced once: at most 15 in the A part, 17 elsewhere, each below P^2.
class Recurrence {
public:
    static constexpr std::size_t kBlock = 16;
    static constexpr std::size_t kPadding = 16;
    static constexpr std::size_t kMaxTaps = 16;

    // At most kMaxTaps taps with distinct distances d >= 1; rhs by increasing index.
    Recurrence(std::span<const Term> taps, std::span<const Term> rhs) {
        using detail::montgomery_form;
        std::array<std::uint32_t, kBlock> near{};  // near[d]: coefficient of a short tap
        for (const auto [d, c] : taps) {
            if (d < kBlock) {
                near[d] = c;
                width_ = std::max<std::size_t>(width_, d);
            } else {
                far_.push_back({d, montgomery_form(c)});
            }
        }
        std::ranges::sort(far_, {}, &Term::index);
        for (const auto [i, r] : rhs) rhs_.push_back({i, montgomery_form(r)});

        // y[kBlock + t] for t in [-kBlock, kStep): one solution of the short taps.
        using Run = std::array<std::uint32_t, kBlock + kStep>;
        const auto run = [&](Run& y) {
            for (std::size_t t = kBlock; t < y.size(); ++t)
                for (std::size_t d = 1; d <= width_; ++d) y[t] = (y[t] + multiply(near[d], y[t - d])) % kModulus;
        };
        for (std::size_t j = 0; j < width_; ++j) {
            Run y{};
            y[kBlock - 1 - j] = 1;
            run(y);
            state_[j] = columns<kStep>(std::span(y).subspan<kBlock>());
        }
        Run u{};
        u[kBlock] = 1;
        run(u);
        for (std::size_t s = 0; s < kBlock; ++s) {
            std::array<std::uint32_t, kBlock> shifted{};
            std::copy_n(u.begin() + kBlock, kBlock - s, shifted.begin() + s);
            impulse_[s] = columns<kBlock>(shifted);
        }
    }

    // Coefficients next() reads before out: kPadding, or the largest tap distance if larger.
    std::size_t history() const { return far_.empty() ? kPadding : std::max<std::size_t>(kPadding, far_.back().index); }

    void next(std::uint32_t* out, std::size_t count) {
        const std::size_t end = done_ + (count + kBlock - 1) / kBlock * kBlock;
        while (done_ < end) {
            if (active_ == 0) {  // up to the first block a long tap reaches, or that holds r
                std::size_t stop = end;
                if (!far_.empty()) stop = std::min<std::size_t>(stop, far_[0].index / kBlock * kBlock);
                if (next_rhs_ < rhs_.size()) stop = std::min<std::size_t>(stop, rhs_[next_rhs_].index / kBlock * kBlock);
                if (done_ < stop) {
                    if (width_) (this->*homogeneous_kernel(width_))(out, stop - done_);
                    else std::fill(out, out + (stop - done_), 0);
                    out += stop - done_;
                    done_ = stop;
                    continue;
                }
            }
            forced(out);
            out += kBlock;
            done_ += kBlock;
        }
    }

private:
    using Vec = detail::Vec;
    static constexpr std::size_t kStep = 64;  // coefficients per step of homogeneous()

    // Values t = 0 .. 8 kHalves - 1 as qword lanes: q[2h] holds t = 8h + 0, 2, 4, 6 and q[2h + 1]
    // t = 8h + 1, 3, 5, 7. Products with a broadcast value need no shuffles.
    template <std::size_t kHalves>
    struct Columns {
        Vec q[2 * kHalves];
    };
    using Sums = Columns<kBlock / 8>;

    // v times 2^32, as Columns.
    template <std::size_t kSize>
    static Columns<kSize / 8> columns(std::span<const std::uint32_t, kSize> v) {
        using detail::montgomery_form;
        Columns<kSize / 8> c;
        for (std::size_t h = 0; h < kSize / 4; ++h) {
            const std::size_t t = 8 * (h / 2) + h % 2;
            c.q[h] = _mm256_setr_epi64x(montgomery_form(v[t]), montgomery_form(v[t + 2]), montgomery_form(v[t + 4]),
                                        montgomery_form(v[t + 6]));
        }
        return c;
    }

    // sums += c x[0, 16) for one coefficient c (in every dword).
    static void multiply_add(Sums& sums, const std::uint32_t* x, Vec c) {
        using detail::multiply_add;
        const Vec lo = detail::load(x), hi = detail::load(x + 8);
        sums.q[0] = multiply_add(sums.q[0], lo, c);
        sums.q[1] = multiply_add(sums.q[1], _mm256_srli_epi64(lo, 32), c);
        sums.q[2] = multiply_add(sums.q[2], hi, c);
        sums.q[3] = multiply_add(sums.q[3], _mm256_srli_epi64(hi, 32), c);
    }

    template <bool kFold>
    static void reduce(const Sums& sums, std::uint32_t* out) {
        detail::store(out, detail::reduce<kFold>(sums.q[0], sums.q[1]));
        detail::store(out + 8, detail::reduce<kFold>(sums.q[2], sums.q[3]));
    }

    // Rows [16 first, 16 first + 16) of the A part, from state[-width, 0). The empty asm keeps the
    // sums in registers in this order: GCC would otherwise form all products first and spill.
    Sums state_part(const std::uint32_t* state, std::size_t width, std::size_t first = 0) const {
        Sums sums{};
#pragma GCC unroll 16
        for (std::size_t j = 0; j < width; ++j) {
            const Vec x = detail::broadcast(state[-1 - std::ptrdiff_t(j)]);
#pragma GCC unroll 4
            for (std::size_t h = 0; h < 4; ++h) sums.q[h] = detail::multiply_add(sums.q[h], state_[j].q[4 * first + h], x);
            asm("" : "+x"(sums.q[0]), "+x"(sums.q[1]), "+x"(sums.q[2]), "+x"(sums.q[3]));
        }
        return sums;
    }

    // count (a multiple of kBlock) coefficients where no long tap reaches and r is zero. The last
    // block of a step holds the next state and comes first, so the next step can start while the
    // other blocks run.
    template <std::size_t kWidth>
    void homogeneous(std::uint32_t* out, std::size_t count) const {
        constexpr bool kFold = kWidth > detail::kUnfolded;
        constexpr std::size_t kLast = kStep / kBlock - 1;
        std::uint32_t* const end = out + count;
        for (; out + kStep <= end; out += kStep) {
            Vec x[kWidth];  // the state, broadcast once for the step's four blocks
#pragma GCC unroll 16
            for (std::size_t j = 0; j < kWidth; ++j) x[j] = detail::broadcast(out[-1 - std::ptrdiff_t(j)]);
            reduce<kFold>(state_part(x, kLast), out + kBlock * kLast);
#pragma GCC unroll 4
            for (std::size_t b = 0; b < kLast; ++b) reduce<kFold>(state_part(x, b), out + kBlock * b);
        }
        for (; out < end; out += kBlock) reduce<kFold>(state_part(out, kWidth), out);
    }

    // state_part() from the state broadcast in x.
    template <std::size_t kWidth>
    Sums state_part(const Vec (&x)[kWidth], std::size_t first) const {
        Sums sums{};
#pragma GCC unroll 16
        for (std::size_t j = 0; j < kWidth; ++j) {
#pragma GCC unroll 4
            for (std::size_t h = 0; h < 4; ++h) sums.q[h] = detail::multiply_add(sums.q[h], state_[j].q[4 * first + h], x[j]);
            asm("" : "+x"(sums.q[0]), "+x"(sums.q[1]), "+x"(sums.q[2]), "+x"(sums.q[3]));
        }
        return sums;
    }

    using Kernel = void (Recurrence::*)(std::uint32_t*, std::size_t) const;

    static Kernel homogeneous_kernel(std::size_t width) {
        static constexpr auto kKernels = []<std::size_t... W>(std::index_sequence<W...>) {
            return std::array<Kernel, sizeof...(W)>{&Recurrence::homogeneous<W + 1>...};
        }(std::make_index_sequence<kBlock - 1>());
        return kKernels[width - 1];
    }

    // The block g[done_, done_ + 16) at out, with long taps or r.
    void forced(std::uint32_t* out) {
        const std::size_t n = done_;
        while (active_ < far_.size() && far_[active_].index < n + kBlock) ++active_;
        Sums right{};  // r'[n, n + 16) times 2^32
        for (std::size_t k = 0; k < active_; ++k) multiply_add(right, out - far_[k].index, detail::broadcast(far_[k].value));
        for (; next_rhs_ < rhs_.size() && rhs_[next_rhs_].index < n + kBlock; ++next_rhs_) {
            const std::size_t t = rhs_[next_rhs_].index - n;
            alignas(32) std::uint64_t lanes[4] = {};
            lanes[t % 8 / 2] = rhs_[next_rhs_].value;
            Vec& q = right.q[t / 8 * 2 + t % 2];
            q = _mm256_add_epi64(q, _mm256_load_si256(reinterpret_cast<const Vec*>(lanes)));
        }
        if (width_ == 0) return reduce<true>(right, out);
        alignas(32) std::array<std::uint32_t, kBlock> h, r;
        reduce<true>(state_part(out, width_), h.data());
        reduce<true>(right, r.data());
        Sums sums{};
        multiply_add(sums, h.data(), detail::broadcast(detail::kR));  // h times 2^32
        for (std::size_t s = 0; s < kBlock; ++s) {
            const Vec x = detail::broadcast(r[s]);
#pragma GCC unroll 4
            for (std::size_t q = 0; q < 4; ++q) sums.q[q] = detail::multiply_add(sums.q[q], impulse_[s].q[q], x);
        }
        reduce<true>(sums, out);
    }

    std::size_t width_ = 0;                 // the largest short distance, 0 without short taps
    Columns<kStep / 8> state_[kBlock - 1];  // state_[j][t] = A[t][j]
    Sums impulse_[kBlock];                  // impulse_[s][t] = u[t - s], 0 for t < s
    std::vector<Term> far_;                 // long taps by increasing distance, values times 2^32
    std::vector<Term> rhs_;                 // values times 2^32
    std::size_t active_ = 0;                // far_[0, active_) reach the current block
    std::size_t next_rhs_ = 0;              // rhs_[next_rhs_] is the first not yet added
    std::size_t done_ = 0;                  // g[0, done_) is solved
};

}  // namespace poly::sparse
