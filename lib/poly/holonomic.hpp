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

// x / 2^32 mod P in [0, 4P) in the low dword of each qword. kFold: any x < 2^64, folded first as in
// reduce(); else x < 12 P^2.
template <bool kFold>
[[gnu::always_inline]] inline Vec partial(Vec x) {
    if constexpr (kFold)
        x = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), broadcast(kR)),
                             _mm256_blend_epi32(x, _mm256_setzero_si256(), 0xAA));
    return _mm256_srli_epi64(redc(x), 32);
}

// w y / 2^32 mod P in [0, P) in the low dwords, for w < 4P and y < P in the low dwords.
[[gnu::always_inline]] inline Vec scale(Vec w, Vec y) { return subtract(_mm256_srli_epi64(redc(_mm256_mul_epu32(w, y)), 32)); }

// a b / 2^32 mod P in [0, P), dword by dword, for a, b < P.
inline Vec montgomery(Vec a, Vec b) {
    const Vec even = redc(_mm256_mul_epu32(a, b));
    const Vec odd = redc(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32)));
    return subtract(_mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA));
}

// y[i] = 2^32 / (first + step i) mod P for i < count rounded up to a multiple of 32 (y holds
// that many), all these integers in [1, P). Montgomery's batch inversion in 4 interleaved chains
// of 8 lanes: y first holds the chains' prefix products, each step with a factor 2^-32 that the
// backward pass cancels; one scalar inversion for the 32 lane totals.
inline void batch_inverses(std::uint32_t first, std::uint32_t step, std::size_t count, std::uint32_t* y) {
    constexpr std::size_t kChains = 4, kLanes = 8 * kChains;
    const std::size_t end = (count + kLanes - 1) / kLanes * kLanes;
    if (end == 0) return;
    const Vec advance = broadcast(std::uint32_t(kLanes) * step);
    const Vec lane = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), broadcast(step));
    Vec x[kChains], prefix[kChains];
    for (std::size_t c = 0; c < kChains; ++c) {
        x[c] = _mm256_add_epi32(broadcast(first + std::uint32_t(8 * c) * step), lane);
        prefix[c] = x[c];
    }
    // prefix(i) = prefix(i - 32) x(i) / 2^32, 8 lanes at a time.
    for (std::size_t i = kLanes; i < end; i += kLanes) {
        for (std::size_t c = 0; c < kChains; ++c) {
            store(y + i - kLanes + 8 * c, prefix[c]);
            x[c] = _mm256_add_epi32(x[c], advance);
            prefix[c] = montgomery(prefix[c], x[c]);
        }
    }

    alignas(32) std::uint32_t total[kLanes], reciprocal[kLanes];  // reciprocal = 2^32 / total
    for (std::size_t c = 0; c < kChains; ++c) store(total + 8 * c, prefix[c]);
    std::uint32_t running = 1;
    for (std::size_t l = 0; l < kLanes; ++l) {
        reciprocal[l] = running;
        running = multiply(running, total[l]);
    }
    running = multiply(inverse(running), kR);
    for (std::size_t l = kLanes; l-- > 0;) {
        reciprocal[l] = multiply(reciprocal[l], running);
        running = multiply(running, total[l]);
    }

    // With q(i) = 2^32 / prefix(i): y(i) = prefix(i - 32) q(i) / 2^32, q(i - 32) = q(i) x(i) / 2^32.
    Vec q[kChains];
    for (std::size_t c = 0; c < kChains; ++c) q[c] = load(reciprocal + 8 * c);
    for (std::size_t i = end - kLanes; i > 0; i -= kLanes) {
        for (std::size_t c = 0; c < kChains; ++c) {
            store(y + i + 8 * c, montgomery(load(y + i - kLanes + 8 * c), q[c]));
            q[c] = montgomery(q[c], x[c]);
            x[c] = _mm256_sub_epi32(x[c], advance);
        }
    }
    for (std::size_t c = 0; c < kChains; ++c) store(y + 8 * c, q[c]);
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
// Q R from the state is V S: w precomputed columns, plus n V' S if a short tap has b != 0.
//
// Reciprocals: 1 / (n + t) for odd n + t by batch inversion, for even n + t as 1 / ((n + t) / 2)
// from a table of the first half, times 1/2. That 1/2 is folded into the even lanes of W.
//
// Products of a coefficient times 2^32 (Montgomery form) and a value are summed in 64-bit lanes
// and reduced once per stage: W, the product with y, the product with F.
class Holonomic {
public:
    static constexpr std::size_t kBlock = 16;
    static constexpr std::size_t kPadding = 16;
    static constexpr std::size_t kMaxTaps = 16;

    // At most kMaxTaps taps with distinct distances d >= 1; initial = g[0]; size: the number of
    // coefficients next() will produce in all, or more (size + 64 <= P).
    Holonomic(std::span<const Tap> taps, std::uint32_t initial, std::size_t size)
        : table_end_(round_up((size + kBlock - 1) / kBlock * kBlock / 2 + 8)), reciprocals_(table_end_) {
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
        for (std::size_t j = 0; j < width_; ++j) {  // the state value g[n - 1 - j] enters R at t < w - j
            Series r{}, r_slope{};
            for (std::size_t t = 0; t + 1 + j <= width_; ++t) {
                const Tap& tap = near[t + 1 + j];
                r[t] = (tap.constant + multiply(tap.slope, std::uint32_t(t))) % kModulus;
                r_slope[t] = tap.slope;
            }
            v_[j] = columns(product(q, r), true);
            if (slope_) v_slope_[j] = columns(product(q, r_slope), true);
        }
        for (std::size_t t = 0; t < kBlock; ++t) first_[t] = multiply(initial, f[t]);
        for (std::uint32_t m = 1; m < kBlock; ++m) reciprocals_.data()[m] = montgomery_form(inverse(m));
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
        prepare_reciprocals(end);
        const std::uint32_t* odd = odd_.data();  // 1 / (n + t) for odd t
        while (done_ < end) {
            const std::uint32_t* const even = reciprocals_.data() + done_ / 2;  // 2 / (n + t) for even t
            if (active_ == 0) {  // up to the first block a long tap reaches
                std::size_t stop = end;
                if (!far_.empty()) stop = std::min<std::size_t>(stop, far_[0].distance / kBlock * kBlock);
                if (done_ < stop) {
                    const std::size_t m = stop - done_;
                    if (width_) (this->*homogeneous_kernel(width_, slope_))(out, m, even, odd, std::uint32_t(done_));
                    else std::fill(out, out + m, 0);
                    out += m;
                    odd += m / 2;
                    done_ = stop;
                    continue;
                }
            }
            forced(out, even, odd);
            out += kBlock;
            odd += kBlock / 2;
            done_ += kBlock;
        }
    }

private:
    using Vec = detail::Vec;
    using Series = std::array<std::uint32_t, kBlock>;  // a series mod x^16

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

    // odd_ = 1 / m for the odd m in [done_, end); reciprocals_ filled over [done_, end) below
    // table_end_, block by block (each block reads the table at half its indices).
    void prepare_reciprocals(std::size_t end) {
        const std::size_t count = (end - done_) / 2;
        odd_.resize(count + 32);
        detail::batch_inverses(std::uint32_t(done_ + 1), 2, count, odd_.data());
        for (std::size_t n = done_, k = 0; n < std::min(end, table_end_); n += kBlock, k += kBlock / 2)
            reciprocals(reciprocals_.data() + n / 2, odd_.data() + k, reciprocals_.data() + n);
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

    // sum over j < width of column[j] g[n - 1 - j] for kJ = 0 .. width - 1, where last = g[n - 16, n).
    template <std::size_t... kJ>
    [[gnu::always_inline]] static Lanes state_part(const Lanes* column, const Block& last, std::index_sequence<kJ...>) {
        const Vec halves[4] = {half<1>(last.hi), half<0>(last.hi), half<1>(last.lo), half<0>(last.lo)};
        Lanes sums{};
        (add_state<kJ>(sums, column[kJ], halves), ...);
        return sums;
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

    // The block F (y ⊙ W) mod x^16, W in the low dwords of w, below 4P, with even lanes halved;
    // even = 2 y at even t, odd = y at odd t. The upper half, which the next block reads, comes
    // first, with columns s even and odd in two chains of additions; the lower half follows,
    // off the chain from block to block.
    [[gnu::always_inline]] Block solve(const Vec (&w)[4], const std::uint32_t* even, const std::uint32_t* odd) const {
        using namespace detail;
        const Vec h[4] = {scale(w[0], widen(even)), scale(w[1], widen(odd)), scale(w[2], widen(even + 4)),
                          scale(w[3], widen(odd + 4))};
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

    // count (a multiple of kBlock) coefficients where no long tap reaches; n = the first index.
    template <std::size_t kWidth, bool kSlope>
    void homogeneous(std::uint32_t* out, std::size_t count, const std::uint32_t* even, const std::uint32_t* odd,
                     std::uint32_t n) const {
        constexpr bool kFold = kWidth + kSlope > detail::kUnfolded;
        constexpr auto kColumns = std::make_index_sequence<kWidth>();
        Block last = load_block(out - kBlock);
        for (std::uint32_t* const end = out + count; out < end; out += kBlock, even += kBlock / 2, odd += kBlock / 2, n += kBlock) {
            Lanes sums = state_part(v_, last, kColumns);
            if constexpr (kSlope) add_slope<(kWidth > detail::kUnfolded)>(sums, state_part(v_slope_, last, kColumns), n);
            using detail::partial;
            const Vec w[4] = {partial<kFold>(sums.q[0]), partial<kFold>(sums.q[1]), partial<kFold>(sums.q[2]),
                              partial<kFold>(sums.q[3])};
            last = solve(w, even, odd);
            detail::store(out, last.lo);
            detail::store(out + 8, last.hi);
        }
    }

    static Block load_block(const std::uint32_t* p) { return {detail::load(p), detail::load(p + 8)}; }

    // sums += n x, x = V' S: x reduced, then one product per lane with n 2^32. sums has at most
    // 15 products; one more keeps it below 2^64.
    template <bool kFold>
    [[gnu::always_inline]] static void add_slope(Lanes& sums, const Lanes& x, std::uint32_t n) {
        using namespace detail;
        const Vec lo = reduce<kFold>(x.q[0], x.q[1]), hi = reduce<kFold>(x.q[2], x.q[3]);
        const Vec m = broadcast(montgomery_form(n));
        sums.q[0] = multiply_add(sums.q[0], lo, m);
        sums.q[1] = multiply_add(sums.q[1], _mm256_srli_epi64(lo, 32), m);
        sums.q[2] = multiply_add(sums.q[2], hi, m);
        sums.q[3] = multiply_add(sums.q[3], _mm256_srli_epi64(hi, 32), m);
    }

    using Kernel = void (Holonomic::*)(std::uint32_t*, std::size_t, const std::uint32_t*, const std::uint32_t*,
                                       std::uint32_t) const;

    static Kernel homogeneous_kernel(std::size_t width, bool slope) {
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
        // W = V S (+ n V' S) + Q R_long, each part reduced to [0, P).
        Lanes state = state_part(v_, out, width_);
        if (slope_) add_slope<true>(state, state_part(v_slope_, out, width_), n);
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
        const Block g = solve(w, even, odd);
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
    Series first_;                                   // g[0, 16) = g[0] F
    std::vector<Tap> far_;                           // long taps by increasing distance, a and b times 2^32
    std::size_t table_end_;                          // reciprocals_ holds [1, table_end_) once next() reaches it
    detail::HugeWords reciprocals_;                  // 2^32 / m mod P at m
    std::vector<std::uint32_t> odd_;                 // 2^32 / (n + t) for the odd n + t of a next() call
    std::size_t active_ = 0;                         // far_[0, active_) reach the current block
    std::size_t done_ = 0;                           // g[0, done_) is solved
};

}  // namespace poly::sparse
