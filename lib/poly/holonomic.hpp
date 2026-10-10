// Recurrences with coefficients linear in the index and few taps, modulo P = 998244353:
// n g[n] = sum over taps (d, a, b) of (a + b n) g[n - d] for n >= 1, g[0] given. The core of exp,
// pow and sqrt of a sparse series; log is Recurrence (lib/poly/sparse.hpp) on n g[n], then a
// product with inverses(). x86-64 with AVX2. Log: lib/poly/notes.md ("Sparse").
//
//   const poly::sparse::Tap taps[] = {{1, a1, b1}, {40, a40, 0}};  // {distance >= 1, a, b}
//   poly::sparse::Holonomic recurrence(taps, g0, n);  // n: coefficients to produce in all
//   recurrence.next(out, count);  // n g[n] = (a1 + b1 n) g[n - 1] + a40 g[n - 40], g[0] = g0
//
//   poly::sparse::inverses(first, count, y);  // y[i] = 2^32 / (first + i) mod P
//
// next() works as Recurrence::next: values in [0, P); it writes the next count coefficients
// (rounded up to a multiple of kBlock) to out and reads the history() coefficients before them
// from out[-history(), 0) (zeros before g[0]).
//
// -DHOLONOMIC_CHAINED=0 or 1 turns the chained kernel (below) off, or on for every w <= 8 (tests).
#pragma once

#include <immintrin.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "lib/poly/sparse.hpp"

namespace poly::sparse {

// The term (constant + slope n) g[n - distance] of a Holonomic recurrence.
struct Tap {
    std::uint32_t distance;  // >= 1
    std::uint32_t constant;  // in [0, P)
    std::uint32_t slope;     // in [0, P)
};

namespace detail {

// One Montgomery step on each qword x < 2^64 - 2^32 P: x / 2^32 mod P in the high dword,
// below x / 2^32 + P.
inline Vec redc(Vec x) {
    return _mm256_add_epi64(x, _mm256_mul_epu32(_mm256_mul_epu32(x, broadcast(kNI)), broadcast(kP)));
}

// x mod P for dwords x < 2P.
inline Vec subtract(Vec x) { return _mm256_min_epu32(x, _mm256_sub_epi32(x, broadcast(kP))); }

// x / 2 mod P for dwords x < P.
inline Vec halve(Vec x) {
    const Vec odd = _mm256_and_si256(x, broadcast(1));
    return _mm256_srli_epi32(_mm256_add_epi32(x, _mm256_and_si256(_mm256_sub_epi32(_mm256_setzero_si256(), odd), broadcast(kP))), 1);
}

// x = 2^32 h + l -> h (2^32 mod P) + l < 2^62, the same modulo P, for any qword x.
[[gnu::always_inline]] inline Vec fold(Vec x) {
    return _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), broadcast(kR)),
                            _mm256_blend_epi32(x, _mm256_setzero_si256(), 0xAA));
}

// x / 2^32 mod P in [0, 4P) in the low dword of each qword. kFold: any x < 2^64, folded first;
// else x < 12 P^2.
template <bool kFold>
[[gnu::always_inline]] inline Vec partial(Vec x) {
    if constexpr (kFold) x = fold(x);
    return _mm256_srli_epi64(redc(x), 32);
}

// (w y + s) / 2^32 mod P in [0, P) in the low dwords, for w < 4P and y < P in the low dwords and
// s a qword sum of kTerms products below P^2. Unfolded (kTerms <= 10), w y + s is below 14 P^2
// and the Montgomery step leaves (14 P / 2^32 + 1) P < 4.26 P < 2^32; with s folded, below 3P.
template <std::size_t kTerms = 0>
[[gnu::always_inline]] inline Vec scale(Vec w, Vec y, Vec s = _mm256_setzero_si256()) {
    constexpr bool kFold = kTerms > 10;
    Vec x = _mm256_mul_epu32(w, y);
    if constexpr (kTerms > 0) x = _mm256_add_epi64(x, kFold ? fold(s) : s);
    x = _mm256_srli_epi64(redc(x), 32);
    if constexpr (!kFold && kTerms > 8) x = _mm256_min_epu32(x, _mm256_sub_epi32(x, broadcast(4 * kP)));
    if constexpr (kTerms > 0) x = _mm256_min_epu32(x, _mm256_sub_epi32(x, broadcast(2 * kP)));
    return subtract(x);
}

// a b / 2^32 mod P in [0, P), dword by dword, for a, b < P.
inline Vec montgomery(Vec a, Vec b) {
    const Vec even = redc(_mm256_mul_epu32(a, b));
    const Vec odd = redc(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32)));
    return subtract(_mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA));
}

// y[i] = 2^32 / (first + step i) mod P for i < count rounded up to a multiple of 32 (y holds
// that many), all these integers in [1, P). Montgomery's batch inversion in 4 interleaved chains
// of 8 lanes: y first holds the chains' prefix products, each step with a factor 2^-32 that the
// backward pass cancels; the 32 lane totals by a product tree and one scalar inversion. The work
// comes in half steps (two chains of one step), to be interleaved with other work: start(), then
// step() until done(), or finish().
class BatchInverter {
public:
    void start(std::uint32_t first, std::uint32_t step, std::size_t count, std::uint32_t* y) {
        end_ = (count + kLanes - 1) / kLanes * kLanes;
        if (end_ == 0) {
            phase_ = Phase::kDone;
            return;
        }
        advance_ = broadcast(std::uint32_t(kLanes) * step);
        const Vec lane = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), broadcast(step));
        for (std::size_t c = 0; c < kChains; ++c) {
            x_[c] = _mm256_add_epi32(broadcast(first + std::uint32_t(8 * c) * step), lane);
            value_[c] = x_[c];
        }
        y_ = y;
        i_ = kLanes;
        chain_ = 0;
        phase_ = Phase::kForward;
    }

    bool done() const { return phase_ == Phase::kDone; }

    // Forward: prefix(i) = prefix(i - 32) x(i) / 2^32. Backward, with q(i) = 2^32 / prefix(i):
    // y(i) = prefix(i - 32) q(i) / 2^32, q(i - 32) = q(i) x(i) / 2^32.
    [[gnu::always_inline]] void step() {
        if (phase_ == Phase::kForward) {
            if (i_ == end_) {
                invert_totals();
                phase_ = Phase::kBackward;
                i_ = end_ - kLanes;
                return;
            }
            for (std::size_t c = chain_; c < chain_ + 2; ++c) {
                store(y_ + i_ - kLanes + 8 * c, value_[c]);
                x_[c] = _mm256_add_epi32(x_[c], advance_);
                value_[c] = montgomery(value_[c], x_[c]);
            }
            chain_ ^= 2;
            if (chain_ == 0) i_ += kLanes;
        } else if (phase_ == Phase::kBackward) {
            if (i_ == 0) {
                for (std::size_t c = 0; c < kChains; ++c) store(y_ + 8 * c, value_[c]);
                phase_ = Phase::kDone;
                return;
            }
            for (std::size_t c = chain_; c < chain_ + 2; ++c) {
                store(y_ + i_ + 8 * c, montgomery(load(y_ + i_ - kLanes + 8 * c), value_[c]));
                value_[c] = montgomery(value_[c], x_[c]);
                x_[c] = _mm256_sub_epi32(x_[c], advance_);
            }
            chain_ ^= 2;
            if (chain_ == 0) i_ -= kLanes;
        }
    }

    void finish() {
        while (!done()) step();
    }

private:
    static constexpr std::size_t kChains = 4, kLanes = 8 * kChains;
    enum class Phase { kForward, kBackward, kDone };

    // value_ = 2^32 / value_ lane by lane: Montgomery products keep every node of the tree a
    // Montgomery form, so 2^32 / root at the root gives 2^32 / p at the leaves.
    void invert_totals() {
        const Vec a = montgomery(value_[0], value_[1]), b = montgomery(value_[2], value_[3]), c = montgomery(a, b);
        const Vec c_swap = _mm256_permute2x128_si256(c, c, 1), d = montgomery(c, c_swap);  // lanes l, l ^ 4
        const Vec d_swap = _mm256_shuffle_epi32(d, 0x4E), e = montgomery(d, d_swap);       // l ^ 2
        const Vec e_swap = _mm256_shuffle_epi32(e, 0xB1), root = montgomery(e, e_swap);    // l ^ 1
        const Vec inverse_root = broadcast(multiply(inverse(std::uint32_t(_mm256_cvtsi256_si32(root))), kR));
        const Vec inverse_c = montgomery(montgomery(montgomery(inverse_root, e_swap), d_swap), c_swap);
        const Vec inverse_a = montgomery(inverse_c, b), inverse_b = montgomery(inverse_c, a);
        const Vec p0 = value_[0], p2 = value_[2];
        value_[0] = montgomery(inverse_a, value_[1]);
        value_[1] = montgomery(inverse_a, p0);
        value_[2] = montgomery(inverse_b, value_[3]);
        value_[3] = montgomery(inverse_b, p2);
    }

    Vec x_[kChains], value_[kChains];  // value: prefix products forward, their reciprocals backward
    Vec advance_;
    std::uint32_t* y_ = nullptr;
    std::size_t i_ = 0, end_ = 0;
    std::size_t chain_ = 0;  // the first of the two chains of the next half step
    Phase phase_ = Phase::kDone;
};

inline void batch_inverses(std::uint32_t first, std::uint32_t step, std::size_t count, std::uint32_t* y) {
    BatchInverter inverter;
    inverter.start(first, step, count, y);
    inverter.finish();
}

// Words in transparent huge pages where available: a 2 MiB table in 4 KiB pages takes 512 page
// faults, about 1 ms in a fresh process.
class HugeWords {
public:
    explicit HugeWords(std::size_t words) : bytes_((words * sizeof(std::uint32_t) + kHuge - 1) / kHuge * kHuge + kHuge) {
        region_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (region_ == MAP_FAILED) std::abort();
        const std::uintptr_t aligned = (reinterpret_cast<std::uintptr_t>(region_) + kHuge - 1) & ~(kHuge - 1);
#ifdef MADV_HUGEPAGE
        ::madvise(reinterpret_cast<void*>(aligned), bytes_ - kHuge, MADV_HUGEPAGE);
#endif
        data_ = reinterpret_cast<std::uint32_t*>(aligned);
    }
    HugeWords(const HugeWords&) = delete;
    HugeWords& operator=(const HugeWords&) = delete;
    ~HugeWords() { ::munmap(region_, bytes_); }

    std::uint32_t* data() const { return data_; }

private:
    static constexpr std::size_t kHuge = std::size_t(1) << 21;
    std::size_t bytes_;
    void* region_;
    std::uint32_t* data_;
};

}  // namespace detail

// y[i] = 2^32 / (first + i) mod P for i < count rounded up to a multiple of 32 (y holds that
// many). first >= 1, and first + count + 31 <= P.
inline void inverses(std::uint32_t first, std::size_t count, std::uint32_t* y) { detail::batch_inverses(first, 1, count, y); }

// n g[n] = sum over taps (d, a, b) of (a + b n) g[n - d], solved in blocks of 16 coefficients.
//
// In the block g[n, n + 16) = G, the recurrence reads (1 - B)(n + θ) G - (A + B') G = R, with
// θ = x d/dx, A, B, B' the sums of a x^d, b x^d, b d x^d over taps with d < 16 ("short"), and R
// the terms that reach before the block: the last w = max d values (the state), and taps with
// d >= 16 ("long"). F, the solution for n = 0 with F[0] = 1, turns it into
// G = F (y ⊙ W) mod x^16 with W = Q R, y[t] = 1 / (n + t) and Q = 1 / ((1 - B) F) mod x^16.
// From the state, R = R0 + n R1 (R1 from the slopes b), and n y[t] = 1 - t y[t] gives
// y ⊙ W = y ⊙ (Q R0 - t Q R1) + Q R1: G = F H, H = y ⊙ (V S) + V' S with V, V' precomputed
// (w columns each; V' only if a short tap has b != 0).
//
// Chained kernel, for w <= 8 where it is faster (from w = 5 without slopes, at w = 8 with them):
// blocks of 8, whose state is the previous block's top w values, F H_prev at those rows. So
// W = M H_prev and V' S = M' H_prev with M = V T, M' = V' T (8 x 8, T those rows of F): the chain
// from block to block is two reductions and one matrix product, without the triangle F H.
//
// Reciprocals: 1 / (n + t) for odd n + t by batch inversion in windows of kWindow coefficients,
// for even n + t as 1 / ((n + t) / 2) from a table of the first half, times 1/2. That 1/2 is
// folded into the even lanes of W. The kernel loop runs the batch inversion of the next window, a
// half step per block, in the cycles the chain from block to block leaves free.
//
// Products of a coefficient times 2^32 (Montgomery form) and a value are summed in 64-bit lanes
// and reduced once per stage: W, the product with y, the product with F.
class Holonomic {
public:
    static constexpr std::size_t kBlock = 16;
    static constexpr std::size_t kPadding = 16;
    static constexpr std::size_t kMaxTaps = 16;
    static constexpr std::size_t kWindow = 16384;  // coefficients per batch of odd reciprocals

    // At most kMaxTaps taps with distinct distances d >= 1; initial = g[0]; size: the number of
    // coefficients next() will produce in all, or more (size + 64 <= P).
    Holonomic(std::span<const Tap> taps, std::uint32_t initial, std::size_t size)
        : size_(round_up(size)), table_end_(round_up(size_ / 2 + 8)), reciprocals_(table_end_) {
        using detail::montgomery_form;
        std::array<Tap, kBlock> near{};  // near[d]: the short tap at distance d, or zeros
        for (const Tap& tap : taps) {
            if (tap.distance < kBlock) {
                near[tap.distance] = tap;
                width_ = std::max<std::size_t>(width_, tap.distance);
                slope_ |= tap.slope != 0;
            } else {
                far_.push_back({tap.distance, montgomery_form(tap.constant), montgomery_form(tap.slope)});
                far_slope_ |= tap.slope != 0;
            }
        }
        std::ranges::sort(far_, {}, &Tap::distance);

        Series f{};  // F
        f[0] = 1;
        for (std::uint32_t t = 1; t < kBlock; ++t) {
            std::uint64_t sum = 0;
            for (std::uint32_t d = 1; d <= std::min<std::uint32_t>(t, std::uint32_t(width_)); ++d)
                sum += multiply((near[d].constant + multiply(near[d].slope, t)) % kModulus, f[t - d]);
            f[t] = multiply(std::uint32_t(sum % kModulus), inverse(t));
        }
        Series c = f;  // (1 - B) F
        for (std::size_t d = 1; d <= width_; ++d)
            for (std::size_t t = d; t < kBlock; ++t) c[t] = (c[t] + kModulus - multiply(near[d].slope, f[t - d])) % kModulus;
        Series q{};  // Q
        q[0] = 1;
        for (std::size_t t = 1; t < kBlock; ++t) {
            std::uint64_t sum = 0;
            for (std::size_t j = 1; j <= t; ++j) sum += multiply(c[j], q[t - j]);
            q[t] = (kModulus - std::uint32_t(sum % kModulus)) % kModulus;
        }

        for (std::size_t s = 0; s < kBlock; ++s) {
            f_[s] = columns(shift(f, s), false);
            if (width_ && !far_.empty()) q_[s] = columns(shift(q, s), true);
        }
        Matrix m{}, m_slope{}, f_half{};  // M, M', F mod x^8 by columns (chained kernel)
        for (std::size_t j = 0; j < width_; ++j) {  // the state value g[n - 1 - j] enters R at t < w - j
            Series r{}, r_slope{};
            for (std::size_t t = 0; t + 1 + j <= width_; ++t) {
                const Tap& tap = near[t + 1 + j];
                r[t] = (tap.constant + multiply(tap.slope, std::uint32_t(t))) % kModulus;
                r_slope[t] = tap.slope;
            }
            const Series w = product(q, r), w_slope = product(q, r_slope);  // Q R0, Q R1 for this value
            Series v;  // Q R0 - t Q R1
            for (std::size_t t = 0; t < kBlock; ++t) v[t] = (w[t] + kModulus - multiply(std::uint32_t(t), w_slope[t])) % kModulus;
            v_[j] = columns(v, true);
            if (slope_) v_slope_[j] = columns(w_slope, false);
            if (width_ <= kHalf)  // g[n - 1 - j] = sum over s <= 7 - j of F[7 - j - s] H_prev[s]
                for (std::size_t s = 0; s + j < kHalf; ++s)
                    for (std::size_t t = 0; t < kHalf; ++t) {
                        m[s][t] = (m[s][t] + multiply(v[t], f[kHalf - 1 - j - s])) % kModulus;
                        m_slope[s][t] = (m_slope[s][t] + multiply(w_slope[t], f[kHalf - 1 - j - s])) % kModulus;
                    }
        }
        for (std::size_t s = 0; s < kHalf; ++s)
            for (std::size_t t = s; t < kHalf; ++t) f_half[s][t] = f[t - s];
        m_ = crossed(m, true);
        m_slope_ = crossed(m_slope, false);
        f_half_ = crossed(f_half, false);
        for (std::size_t t = 0; t < kBlock; ++t) first_[t] = multiply(initial, f[t]);
        for (std::uint32_t m = 1; m < kBlock; ++m) reciprocals_.data()[m] = montgomery_form(inverse(m));
        if (size_ > kBlock)
            for (auto& odd : odd_) odd.resize(kWindow / 2 + 32);
    }

    // Coefficients next() reads before out: kPadding, or the largest tap distance if larger.
    std::size_t history() const { return far_.empty() ? kPadding : std::max<std::size_t>(kPadding, far_.back().distance); }

    void next(std::uint32_t* out, std::size_t count) {
        const std::size_t end = done_ + round_up(count);
        if (done_ == 0 && end > 0) {
            std::ranges::copy(first_, out);
            out += kBlock;
            done_ = kBlock;
        }
        if (done_ == end || (width_ == 0 && far_.empty())) {
            std::fill(out, out + (end - done_), 0);
            done_ = end;
            return;
        }
        while (done_ < end) {
            if (done_ >= window_end_) begin_window();
            const std::uint32_t* const even = reciprocals_.data() + done_ / 2;  // 2 / (n + t) for even t
            const std::uint32_t* const odd = odd_[window_ % 2].data() + (done_ + kWindow - window_end_) / 2;  // odd t
            const std::size_t stop = std::min(end, window_end_);
            if (active_ == 0) {  // up to the first block a long tap reaches
                const std::size_t free = far_.empty() ? stop : std::min<std::size_t>(stop, far_[0].distance / kBlock * kBlock);
                if (done_ < free) {
                    const std::size_t m = free - done_;
                    if (width_) (this->*homogeneous_kernel(width_, slope_))(out, m, even, odd);
                    else std::fill(out, out + m, 0);
                    out += m;
                    done_ = free;
                    continue;
                }
            }
            forced(out, even, odd);
            out += kBlock;
            done_ += kBlock;
        }
    }

private:
    using Vec = detail::Vec;
    using Series = std::array<std::uint32_t, kBlock>;  // a series mod x^16

    static constexpr std::size_t kHalf = kBlock / 2;                     // the chained kernel's block
    using Matrix = std::array<std::array<std::uint32_t, kHalf>, kHalf>;  // 8 x 8, by columns

    // An 8 x 8 matrix for products with H (8 values) broadcast within 128-bit halves: x[k] holds H[k]
    // in its lower half and H[k ^ 4] in its upper half (no lane crossing for k < 4), and c[k][i]
    // holds rows i, i + 2 of column k and rows i + 4, i + 6 of column k ^ 4. Sums come out as rows
    // i, i + 2 | i + 4, i + 6 in qwords: the layout of Lanes.q[i].
    struct Crossed {
        Vec c[kHalf][2];
    };

    // 16 values as qword lanes: q[2h] holds t = 8h + 0, 2, 4, 6 and q[2h + 1] t = 8h + 1, 3, 5, 7.
    struct Lanes {
        Vec q[4];
    };

    // The 16 values of a block in registers: lo = g[n, n + 8), hi = g[n + 8, n + 16). The block
    // chain goes through registers: broadcasts from a 32-byte store wait for it to complete.
    struct Block {
        Vec lo, hi;
    };

    static constexpr std::size_t round_up(std::size_t x) { return (x + kBlock - 1) / kBlock * kBlock; }

    static Series shift(const Series& a, std::size_t s) {
        Series b{};
        std::copy_n(a.begin(), kBlock - s, b.begin() + s);
        return b;
    }

    static Series product(const Series& a, const Series& b) {  // a b mod x^16
        Series c{};
        for (std::size_t i = 0; i < kBlock; ++i)
            for (std::size_t j = 0; i + j < kBlock; ++j) c[i + j] = (c[i + j] + multiply(a[i], b[j])) % kModulus;
        return c;
    }

    // v times 2^32, the even lanes also times 1/2 if halve_even (columns of W).
    static Lanes columns(const Series& v, bool halve_even) {
        using detail::montgomery_form;
        const std::uint32_t half = halve_even ? (kModulus + 1) / 2 : 1;
        Lanes c;
        for (std::size_t h = 0; h < 4; ++h) {
            const std::size_t t = 8 * (h / 2) + h % 2;
            const auto form = [&](std::size_t i) { return montgomery_form(multiply(v[i], i % 2 ? 1 : half)); };
            c.q[h] = _mm256_setr_epi64x(form(t), form(t + 2), form(t + 4), form(t + 6));
        }
        return c;
    }

    // The next window [w kWindow, (w + 1) kWindow): its odd reciprocals in odd_[w % 2] (computed
    // during window w - 1, finished here), the table over it below table_end_, block by block
    // (each block reads the table at half its indices). Starts the odd reciprocals of window w + 1,
    // which the homogeneous kernels advance by a half step per block.
    void begin_window() {
        if (window_end_ == 0) start_inverter(0);
        inverter_.finish();
        window_ = window_end_ / kWindow;
        const std::uint32_t* const odd = odd_[window_ % 2].data();
        for (std::size_t n = window_end_, k = 0; n < std::min(window_end_ + kWindow, table_end_); n += kBlock, k += kBlock / 2)
            reciprocals(reciprocals_.data() + n / 2, odd + k, reciprocals_.data() + n);
        window_end_ += kWindow;
        if (window_end_ < size_) start_inverter(window_ + 1);
    }

    void start_inverter(std::size_t window) {
        const std::size_t first = window * kWindow;
        inverter_.start(std::uint32_t(first + 1), 2, std::min(kWindow, size_ - first) / 2, odd_[window % 2].data());
    }

    // y[t] = 1 / (n + t) for t < 16 (Montgomery forms): even t from even[t / 2] = 2 / (n + t),
    // odd t from odd[t / 2].
    static void reciprocals(const std::uint32_t* even, const std::uint32_t* odd, std::uint32_t* y) {
        using namespace detail;
        const Vec e = halve(load(even)), o = load(odd);
        const Vec lo = _mm256_unpacklo_epi32(e, o), hi = _mm256_unpackhi_epi32(e, o);
        store(y, _mm256_permute2x128_si256(lo, hi, 0x20));
        store(y + 8, _mm256_permute2x128_si256(lo, hi, 0x31));
    }

    // Four words at p as the low dwords of four qwords.
    static Vec widen(const std::uint32_t* p) { return _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }

    // Lane-half k of x (128 bits) in both halves. Broadcasts are a half and an in-lane shuffle:
    // vpermd and vpermq take 8 and 6.5 cycles on Zen 3, vperm2i128 3, vpshufd 1.
    template <int kHalf>
    [[gnu::always_inline]] static Vec half(Vec x) {
        return _mm256_permute2x128_si256(x, x, kHalf * 0x11);
    }

    // sums = V S and, if kSlope, slope = V' S (after V S, which the chain from block to block
    // needs first), for the state S = g[n - 1 - j], j = kJ = 0 .. w - 1, where last = g[n - 16, n).
    template <bool kSlope, std::size_t... kJ>
    [[gnu::always_inline]] void state_part(const Block& last, Lanes& sums, Lanes& slope, std::index_sequence<kJ...>) const {
        const Vec halves[4] = {half<1>(last.hi), half<0>(last.hi), half<1>(last.lo), half<0>(last.lo)};
        (add_state<kJ>(sums, v_[kJ], halves), ...);
        if constexpr (kSlope) (add_state<kJ>(slope, v_slope_[kJ], halves), ...);
    }

    // sums += column g[n - 1 - j]: dword 3 - j % 4 of halves[j / 4].
    template <std::size_t kJ>
    [[gnu::always_inline]] static void add_state(Lanes& sums, const Lanes& column, const Vec (&halves)[4]) {
        const Vec x = _mm256_shuffle_epi32(halves[kJ / 4], (3 - kJ % 4) * 0x55);
#pragma GCC unroll 4
        for (std::size_t h = 0; h < 4; ++h) sums.q[h] = detail::multiply_add(sums.q[h], column.q[h], x);
        asm("" : "+x"(sums.q[0]), "+x"(sums.q[1]), "+x"(sums.q[2]), "+x"(sums.q[3]));
    }

    // The same with a run-time width, for blocks with long taps.
    static Lanes state_part(const Lanes* column, const std::uint32_t* out, std::size_t width) {
        Lanes sums{};
        for (std::size_t j = 0; j < width; ++j) {
            const Vec x = detail::broadcast(out[-1 - std::ptrdiff_t(j)]);
            for (std::size_t h = 0; h < 4; ++h) sums.q[h] = detail::multiply_add(sums.q[h], column[j].q[h], x);
        }
        return sums;
    }

    // 16 values in [0, P) as the low dwords of Lanes.
    static void spread(const std::uint32_t* v, Vec (&w)[4]) {
        const Vec lo = detail::load(v), hi = detail::load(v + 8);
        w[0] = lo, w[1] = _mm256_srli_epi64(lo, 32), w[2] = hi, w[3] = _mm256_srli_epi64(hi, 32);
    }

    // The block F H mod x^16, H = y ⊙ W + V' S: W in the low dwords of w, below 4P, with even lanes
    // halved; slope = V' S, kSlope products per lane (0: none); even = 2 y at even t, odd = y at
    // odd t. The upper half, which the next block reads, comes first, with columns s even and odd
    // in two chains of additions; the lower half follows, off the chain from block to block.
    template <std::size_t kSlope>
    [[gnu::always_inline]] Block solve(const Vec (&w)[4], const Lanes& slope, const std::uint32_t* even,
                                       const std::uint32_t* odd) const {
        using namespace detail;
        const Vec h[4] = {scale<kSlope>(w[0], widen(even), slope.q[0]), scale<kSlope>(w[1], widen(odd), slope.q[1]),
                          scale<kSlope>(w[2], widen(even + 4), slope.q[2]), scale<kSlope>(w[3], widen(odd + 4), slope.q[3])};
        Vec upper_even[2]{}, upper_odd[2]{}, lower[2]{};
        rows<2>(upper_even, upper_odd, h, std::make_index_sequence<kBlock>());
        const Vec hi = reduce<true>(_mm256_add_epi64(upper_even[0], upper_odd[0]),  // 16 products
                                    _mm256_add_epi64(upper_even[1], upper_odd[1]));
        rows<0>(lower, lower, h, std::make_index_sequence<kBlock / 2>());
        return {reduce<false>(lower[0], lower[1]), hi};  // at most 8 products
    }

    template <std::size_t kQ, std::size_t... kS>
    [[gnu::always_inline]] void rows(Vec (&even)[2], Vec (&odd)[2], const Vec (&h)[4], std::index_sequence<kS...>) const {
        (column<kQ, kS>(kS % 2 ? odd : even, h), ...);
    }

    // sums[i] += H[s] times lanes kQ + i of F shifted by s; lanes t < s are zero and skipped. H[s]
    // is qword s % 8 / 2 of h[s / 8 * 2 + s % 2].
    template <std::size_t kQ, std::size_t kS>
    [[gnu::always_inline]] void column(Vec (&sums)[2], const Vec (&h)[4]) const {
        constexpr std::size_t kQword = kS % 8 / 2;
        const Vec x = _mm256_shuffle_epi32(half<kQword / 2>(h[kS / 8 * 2 + kS % 2]), kQword % 2 ? 0xEE : 0x44);
#pragma GCC unroll 2
        for (std::size_t i = 0; i < 2; ++i)
            if (kS <= 8 * (kQ / 2) + i + 6) sums[i] = detail::multiply_add(sums[i], f_[kS].q[kQ + i], x);
        asm("" : "+x"(sums[0]), "+x"(sums[1]));
    }

    // count (a multiple of kBlock) coefficients where no long tap reaches.
    template <std::size_t kWidth, bool kSlope>
    void homogeneous(std::uint32_t* out, std::size_t count, const std::uint32_t* even, const std::uint32_t* odd) {
        constexpr bool kFold = kWidth > detail::kUnfolded;
        constexpr auto kColumns = std::make_index_sequence<kWidth>();
        Block last = load_block(out - kBlock);
        for (std::uint32_t* const end = out + count; out < end; out += kBlock, even += kBlock / 2, odd += kBlock / 2) {
            Lanes sums{}, slope{};
            state_part<kSlope>(last, sums, slope, kColumns);
            using detail::partial;
            const Vec w[4] = {partial<kFold>(sums.q[0]), partial<kFold>(sums.q[1]), partial<kFold>(sums.q[2]),
                              partial<kFold>(sums.q[3])};
            // The inverter's half step is independent work for the cycles the chain leaves free:
            // before the triangle without slopes, after it with them (1-3% faster each way).
            if constexpr (!kSlope) inverter_.step();
            last = solve<kSlope ? kWidth : 0>(w, slope, even, odd);
            detail::store(out, last.lo);
            detail::store(out + 8, last.hi);
            if constexpr (kSlope) inverter_.step();
        }
    }

    static Block load_block(const std::uint32_t* p) { return {detail::load(p), detail::load(p + 8)}; }

    // count (a multiple of kBlock) coefficients where no long tap reaches, for w <= 8, in blocks of
    // 8 (class comment). w and s carry W and V' S of the next block, sums of at most 8 products.
    template <bool kSlope>
    void chained(std::uint32_t* out, std::size_t count, const std::uint32_t* even, const std::uint32_t* odd) {
        Vec w[2]{}, s[2]{};  // the first block's from the state in memory: V S and V' S
        for (std::size_t j = 0; j < width_; ++j) {
            const Vec x = detail::broadcast(out[-1 - std::ptrdiff_t(j)]);
            for (std::size_t i = 0; i < 2; ++i) {
                w[i] = detail::multiply_add(w[i], v_[j].q[i], x);
                if constexpr (kSlope) s[i] = detail::multiply_add(s[i], v_slope_[j].q[i], x);
            }
        }
        Vec g[2]{};  // F H of the block at pending, reduced after the next block's chain operations
        std::uint32_t* pending = nullptr;
        for (std::uint32_t* const end = out + count; out < end; out += kBlock, even += kBlock / 2, odd += kBlock / 2) {
            chained_block<kSlope>(out, w, s, g, pending, even, odd);
            chained_block<kSlope>(out + kHalf, w, s, g, pending, even + kHalf / 2, odd + kHalf / 2);
            inverter_.step();
        }
        detail::store(pending, detail::reduce<false>(g[0], g[1]));
    }

    // The block of 8 at out: H from w, s and y, then w, s for the next block, then G = F H into g
    // (the previous block's G stored first). even and odd as in solve().
    template <bool kSlope>
    [[gnu::always_inline]] void chained_block(std::uint32_t* out, Vec (&w)[2], Vec (&s)[2], Vec (&g)[2], std::uint32_t*& pending,
                                              const std::uint32_t* even, const std::uint32_t* odd) const {
        using namespace detail;
        constexpr std::size_t kTerms = kSlope ? kHalf : 0;
        const Vec h[2] = {scale<kTerms>(partial<false>(w[0]), widen(even), s[0]),
                          scale<kTerms>(partial<false>(w[1]), widen(odd), s[1])};
        const Vec swapped[2] = {_mm256_permute2x128_si256(h[0], h[0], 1), _mm256_permute2x128_si256(h[1], h[1], 1)};
        const Vec x[kHalf] = {_mm256_shuffle_epi32(h[0], 0x44),       _mm256_shuffle_epi32(h[1], 0x44),
                              _mm256_shuffle_epi32(h[0], 0xEE),       _mm256_shuffle_epi32(h[1], 0xEE),
                              _mm256_shuffle_epi32(swapped[0], 0x44), _mm256_shuffle_epi32(swapped[1], 0x44),
                              _mm256_shuffle_epi32(swapped[0], 0xEE), _mm256_shuffle_epi32(swapped[1], 0xEE)};
        products(w, m_, x);
        if constexpr (kSlope) products(s, m_slope_, x);
        if (pending) store(pending, reduce<false>(g[0], g[1]));  // after the chain's products: Zen 3
        g[0] = g[1] = _mm256_setzero_si256();                     // issues the oldest ready first
        products(g, f_half_, x, std::make_index_sequence<kHalf>());
        pending = out;
    }

    // sums = a H, x[k] holding H[k] in its lower half and H[k ^ 4] in its upper half (Crossed).
    [[gnu::always_inline]] static void products(Vec (&sums)[2], const Crossed& a, const Vec (&x)[kHalf]) {
        sums[0] = sums[1] = _mm256_setzero_si256();
#pragma GCC unroll 8
        for (std::size_t k = 0; k < kHalf; ++k) {
            sums[0] = detail::multiply_add(sums[0], a.c[k][0], x[k]);
            sums[1] = detail::multiply_add(sums[1], a.c[k][1], x[k]);
            asm("" : "+x"(sums[0]), "+x"(sums[1]));
        }
    }

    // sums += a H for a lower triangular a (F), skipping the products whose four lanes are zero.
    template <std::size_t... kK>
    [[gnu::always_inline]] static void products(Vec (&sums)[2], const Crossed& a, const Vec (&x)[kHalf],
                                                std::index_sequence<kK...>) {
        const auto column = [&]<std::size_t k>() {
#pragma GCC unroll 2
            for (std::size_t i = 0; i < 2; ++i)  // rows i, i + 2 at column k, i + 4, i + 6 at column k ^ 4
                if (k <= i + 2 || (k ^ 4) <= i + 6) sums[i] = detail::multiply_add(sums[i], a.c[k][i], x[k]);
            asm("" : "+x"(sums[0]), "+x"(sums[1]));
        };
        (column.template operator()<kK>(), ...);
    }

    // a by columns (a[t][r]: row r, column t) as Crossed, times 2^32, even rows also times 1/2 if
    // halve_even.
    static Crossed crossed(const Matrix& a, bool halve_even) {
        using detail::montgomery_form;
        const std::uint32_t half = halve_even ? (kModulus + 1) / 2 : 1;
        const auto form = [&](std::size_t r, std::size_t t) { return montgomery_form(multiply(a[t][r], r % 2 ? 1 : half)); };
        Crossed c;
        for (std::size_t k = 0; k < kHalf; ++k)
            for (std::size_t i = 0; i < 2; ++i)
                c.c[k][i] = _mm256_setr_epi64x(form(i, k), form(i + 2, k), form(i + 4, k ^ 4), form(i + 6, k ^ 4));
        return c;
    }

    using Kernel = void (Holonomic::*)(std::uint32_t*, std::size_t, const std::uint32_t*, const std::uint32_t*);

    // Measured on Zen 3 and Intel (lib/poly/notes.md): the block kernel's cost grows with w (and
    // doubles its state part with slopes), the chained kernel's does not.
    static bool chained_wins(std::size_t width, [[maybe_unused]] bool slope) {
#ifdef HOLONOMIC_CHAINED
        return HOLONOMIC_CHAINED && width <= kHalf;
#else
        return width <= kHalf && width >= (slope ? 8 : 5);
#endif
    }

    static Kernel homogeneous_kernel(std::size_t width, bool slope) {
        if (chained_wins(width, slope)) return slope ? &Holonomic::chained<true> : &Holonomic::chained<false>;
        static constexpr auto kKernels = []<std::size_t... W>(std::index_sequence<W...>) {
            return std::array<std::array<Kernel, 2>, sizeof...(W)>{
                {{&Holonomic::homogeneous<W + 1, false>, &Holonomic::homogeneous<W + 1, true>}...}};
        }(std::make_index_sequence<kBlock - 1>());
        return kKernels[width - 1][slope];
    }

    // The block g[done_, done_ + 16) at out, with long taps; even and odd as in solve().
    void forced(std::uint32_t* out, const std::uint32_t* even, const std::uint32_t* odd) {
        using namespace detail;
        const std::uint32_t n = std::uint32_t(done_);
        while (active_ < far_.size() && far_[active_].distance < n + kBlock) ++active_;
        Lanes constant{}, slope{};  // sums of a g[n + t - d] and b g[n + t - d], times 2^32
        for (std::size_t k = 0; k < active_; ++k) {
            const std::uint32_t* const source = out - far_[k].distance;
            const Vec lo = load(source), hi = load(source + 8), any = _mm256_or_si256(lo, hi);
            if (_mm256_testz_si256(any, any)) continue;
            add_column(constant, lo, hi, broadcast(far_[k].constant));
            if (far_slope_) add_column(slope, lo, hi, broadcast(far_[k].slope));
        }
        alignas(32) std::uint32_t r[kBlock];  // R from the long taps, in [0, P)
        store(r, reduce<true>(constant.q[0], constant.q[1]));
        store(r + 8, reduce<true>(constant.q[2], constant.q[3]));
        if (far_slope_) {  // r += (n + t) b-part
            const Vec m = broadcast(montgomery_form(n));
            for (std::size_t half = 0; half < 2; ++half) {
                const Vec index = add_mod(m, load(kIndices.data() + 8 * half));  // (n + t) 2^32
                const Vec b = reduce<true>(slope.q[2 * half], slope.q[2 * half + 1]);
                store(r + 8 * half, add_mod(load(r + 8 * half), montgomery(b, index)));
            }
        }
        if (width_ == 0) {  // F = Q = 1: G = y ⊙ R
            alignas(32) std::uint32_t y[kBlock];
            reciprocals(even, odd, y);
            store(out, montgomery(load(r), load(y)));
            store(out + 8, montgomery(load(r + 8), load(y + 8)));
            return;
        }
        // W = V S + Q R_long, each part reduced to [0, P); H = y ⊙ W + V' S.
        const Lanes state = state_part(v_, out, width_);
        const Lanes state_slope = slope_ ? state_part(v_slope_, out, width_) : Lanes{};
        Lanes far{};
        for (std::size_t s = 0; s < kBlock; ++s) {
            const Vec x = broadcast(r[s]);
            for (std::size_t q = 0; q < 4; ++q) far.q[q] = multiply_add(far.q[q], q_[s].q[q], x);
        }
        alignas(32) std::uint32_t sum[kBlock];
        for (std::size_t half = 0; half < 2; ++half)
            store(sum + 8 * half, add_mod(reduce<true>(state.q[2 * half], state.q[2 * half + 1]),
                                          reduce<true>(far.q[2 * half], far.q[2 * half + 1])));
        Vec w[4];
        spread(sum, w);
        const Block g = slope_ ? solve<kBlock - 1>(w, state_slope, even, odd) : solve<0>(w, state_slope, even, odd);
        store(out, g.lo);
        store(out + 8, g.hi);
    }

    static void add_column(Lanes& sums, Vec lo, Vec hi, Vec c) {
        using detail::multiply_add;
        sums.q[0] = multiply_add(sums.q[0], lo, c);
        sums.q[1] = multiply_add(sums.q[1], _mm256_srli_epi64(lo, 32), c);
        sums.q[2] = multiply_add(sums.q[2], hi, c);
        sums.q[3] = multiply_add(sums.q[3], _mm256_srli_epi64(hi, 32), c);
    }

    static Vec add_mod(Vec a, Vec b) { return detail::subtract(_mm256_add_epi32(a, b)); }

    static constexpr std::array<std::uint32_t, kBlock> kIndices = [] {  // t 2^32 mod P
        std::array<std::uint32_t, kBlock> a{};
        for (std::uint32_t t = 0; t < kBlock; ++t) a[t] = std::uint32_t((std::uint64_t(t) << 32) % kModulus);
        return a;
    }();

    std::size_t width_ = 0;                          // the largest short distance, 0 without short taps
    bool slope_ = false;                             // a short tap has b != 0
    bool far_slope_ = false;                         // a long tap has b != 0
    Lanes f_[kBlock];                                // f_[s][t] = F[t - s], 0 for t < s
    Lanes q_[kBlock];                                // q_[s][t] = Q[t - s], with long and short taps only
    Lanes v_[kBlock - 1];                            // v_[j] = V column j
    Lanes v_slope_[kBlock - 1];                      // V'
    Crossed m_;                                      // M (w <= 8), even rows halved as in V
    Crossed m_slope_;                                // M'
    Crossed f_half_;                                 // F mod x^8
    Series first_;                                   // g[0, 16) = g[0] F
    std::vector<Tap> far_;                           // long taps by increasing distance, a and b times 2^32
    std::size_t size_;                               // coefficients next() produces in all, or more
    std::size_t table_end_;                          // reciprocals_ holds [1, table_end_) once next() reaches it
    detail::HugeWords reciprocals_;                  // 2^32 / m mod P at m
    std::array<std::vector<std::uint32_t>, 2> odd_;  // 2^32 / m for the odd m of window w in odd_[w % 2]
    detail::BatchInverter inverter_;                 // fills odd_ for the window after the current one
    std::size_t window_ = 0;                         // the current window
    std::size_t window_end_ = 0;                     // its end; 0 before the first
    std::size_t active_ = 0;                         // far_[0, active_) reach the current block
    std::size_t done_ = 0;                           // g[0, done_) is solved
};

}  // namespace poly::sparse
