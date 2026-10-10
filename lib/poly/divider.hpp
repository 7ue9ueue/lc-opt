// Division by the index of a series produced range by range, modulo P = 998244353:
// g[n] = G[n] / n for n >= 1, g[0] = 0. log of a sparse series is Recurrence (lib/poly/sparse.hpp)
// on G = n g, then this. x86-64 with AVX2. Log: lib/poly/notes.md ("Sparse").
//
//   poly::sparse::Divider divider(n);        // indices below n
//   divider.divide(G, g, first, count);      // g[i] = G[i] / (first + i) for i < count
//
// Calls cover [0, n) in order; every count but the last is a multiple of kStep. divide() reads
// G and writes g over count rounded up to kStep values, and reads one more word of G; g may be G.
// Values are in [0, P).
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/poly/holonomic.hpp"

namespace poly::sparse {

// 1 / n in Montgomery form (2^32 / n): odd n by batch inversion in 4 interleaved chains of 8 lanes,
// each chain the 8 odd n of one block of 16; even n = 2m from a table of 2^32 / (2m) at m. The
// table is written below its end as blocks finish and read at half their indices, so a call is
// cut into pieces [a, b) with b <= 2a, and at the table's end; each piece costs one scalar
// inversion.
//
// A piece takes three passes: prefix products, the odd reciprocals (backward), then the blocks:
// G times the reciprocals (Montgomery products), and the table's entries. Pieces that write the
// table invert 2x instead of x (its entries for odd m); their products take twice that.
class Divider {
public:
    static constexpr std::size_t kStep = 64;  // indices per step: 4 chains of 8 odd n

    explicit Divider(std::size_t size) : table_end_(round_up(round_up(size) / 2)), table_(table_end_) {
        for (std::uint32_t m = 1; m < kStep / 2; ++m) table_.data()[m] = detail::montgomery_form(inverse(2 * m));
    }

    void divide(const std::uint32_t* G, std::uint32_t* g, std::size_t first, std::size_t count) {
        const std::size_t end = first + round_up(count);
        while (first < end) {
            const std::size_t stop = first == 0 ? kStep : std::min({end, 2 * first, first < table_end_ ? table_end_ : end});
            if (first < table_end_) piece<true>(G, g, first, stop);
            else piece<false>(G, g, first, stop);
            G += stop - first;
            g += stop - first;
            first = stop;
        }
    }

private:
    using Vec = detail::Vec;
    static constexpr std::size_t kBlock = 16;
    static constexpr std::size_t kChains = 4;

    static constexpr std::size_t round_up(std::size_t x) { return (x + kStep - 1) / kStep * kStep; }

    // Lane l of chain c: odd n = base + 16 c + 2 l + 1. odd: the odd lanes' n in the low dwords.
    static constexpr auto kOffsets = [] {
        struct {
            alignas(32) std::uint32_t all[kChains][8], odd[kChains][8];
        } t{};
        for (std::uint32_t c = 0; c < kChains; ++c)
            for (std::uint32_t l = 0; l < 8; ++l) {
                t.all[c][l] = std::uint32_t(kBlock) * c + 2 * l + 1;
                t.odd[c][l] = std::uint32_t(kBlock) * c + 2 * (l | 1) + 1;
            }
        return t;
    }();

    // a b / 2^32 mod P in [0, 2P) for a, b < 2P: (4P^2 + 2^32 P) / 2^32 < 2P. The even lanes come
    // from the low dwords of a and b, the odd lanes from the low dwords of a_odd and b_odd.
    static Vec product(Vec a, Vec a_odd, Vec b, Vec b_odd) {
        using namespace detail;
        const Vec even = redc(_mm256_mul_epu32(a, b)), odd = redc(_mm256_mul_epu32(a_odd, b_odd));
        return _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
    }

    static Vec odd_lanes(Vec x) { return _mm256_srli_epi64(x, 32); }

    static Vec product(Vec a, Vec b) { return product(a, odd_lanes(a), b, odd_lanes(b)); }

    // Four words at p as the low dwords of four qwords.
    static Vec widen(const std::uint32_t* p) { return _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }

    // g[first, end) from G[first, end), both offset to first; end - first a multiple of kStep.
    // Chain c, step j: x = first + kStep j + offsets. Forward: prefix products
    // p_j = p_(j-1) x_j / 2^32; backward from q = 2^32 s / p (s = 1/2 if kWrite, else 1):
    // 2^32 s / x_j = p_(j-1) q_j / 2^32 and q_(j-1) = q_j x_j / 2^32. kWrite: the piece lies
    // below the table's end.
    template <bool kWrite>
    void piece(const std::uint32_t* G, std::uint32_t* g, std::size_t first, std::size_t end) {
        using namespace detail;
        const std::size_t steps = (end - first) / kStep;
        prefix_.resize(std::max<std::size_t>(prefix_.size(), steps * 8 * kChains + 8));
        std::uint32_t* const prefix = prefix_.data();
        const auto slot = [&](std::size_t j, std::size_t c) { return prefix + 8 * (kChains * j + c); };
        const auto base = [&](std::size_t j) { return broadcast(std::uint32_t(first + kStep * j)); };
        const auto x = [&](std::size_t j, std::size_t c) { return _mm256_add_epi32(base(j), load(kOffsets.all[c])); };
        const auto x_odd = [&](std::size_t j, std::size_t c) { return _mm256_add_epi32(base(j), load(kOffsets.odd[c])); };

        Vec p[kChains];
#pragma GCC unroll 4
        for (std::size_t c = 0; c < kChains; ++c) p[c] = x(0, c);
        for (std::size_t j = 1; j < steps; ++j)
#pragma GCC unroll 4
            for (std::size_t c = 0; c < kChains; ++c) {
                store(slot(j, c), p[c]);
                p[c] = product(p[c], odd_lanes(p[c]), x(j, c), x_odd(j, c));
            }

        Vec q[kChains];
        reciprocals(p, q, kWrite ? (kModulus + 1) / 2 : 1);
        for (std::size_t j = steps; j-- > 1;)
#pragma GCC unroll 4
            for (std::size_t c = 0; c < kChains; ++c) {
                const Vec q_odd = odd_lanes(q[c]);
                store(slot(j, c), product(load(slot(j, c)), load(slot(j, c) + 1), q[c], q_odd));
                q[c] = product(q[c], q_odd, x(j, c), x_odd(j, c));
            }
#pragma GCC unroll 4
        for (std::size_t c = 0; c < kChains; ++c) store(slot(0, c), q[c]);

        std::uint32_t* const table = table_.data();  // in a register while stores through g run
        for (std::size_t j = 0; j < steps; ++j)
#pragma GCC unroll 4
            for (std::size_t c = 0; c < kChains; ++c) {
                const std::size_t offset = kStep * j + kBlock * c;
                block<kWrite>(table, G + offset, g + offset, first + offset, slot(j, c));
            }
    }

    // q = 2^32 s / p lane by lane, p < 2P: a product tree over the 32 lanes and one scalar
    // inversion at its root. Montgomery products keep every node a Montgomery form: the root is
    // 2^32 T for the product T of the p / 2^32, and 2^32 s / (2^32 T) at the root gives
    // 2^32 (s / 2^32) / (p / 2^32) = 2^32 s / p at the leaves.
    static void reciprocals(const Vec (&p)[kChains], Vec (&q)[kChains], std::uint32_t s) {
        using namespace detail;
        static_assert(kChains == 4);
        const Vec a = product(p[0], p[1]), b = product(p[2], p[3]), c = product(a, b);
        const Vec c_swap = _mm256_permute2x128_si256(c, c, 1), d = product(c, c_swap);  // lanes l, l ^ 4
        const Vec d_swap = _mm256_shuffle_epi32(d, 0x4E), e = product(d, d_swap);       // l ^ 2
        const Vec e_swap = _mm256_shuffle_epi32(e, 0xB1), root = product(e, e_swap);    // l ^ 1
        const std::uint32_t total = std::uint32_t(_mm256_cvtsi256_si32(root)) % kModulus;
        const Vec inverse_root = broadcast(multiply(inverse(total), multiply(kR, s)));
        const Vec inverse_c = product(product(product(inverse_root, e_swap), d_swap), c_swap);
        const Vec inverse_a = product(inverse_c, b), inverse_b = product(inverse_c, a);
        q[0] = product(inverse_a, p[1]);
        q[1] = product(inverse_a, p[0]);
        q[2] = product(inverse_b, p[3]);
        q[3] = product(inverse_b, p[2]);
    }

    // g[0, 16) = G[0, 16) / (n + t), from h[0, 8) = 2^32 s / (n + t) at odd t (below 2P) and the
    // table at n / 2. kWrite: s = 1/2, and the table's entries [n, n + 16) are written; else
    // s = 1, and the products take their factors widened from memory.
    template <bool kWrite>
    static void block(std::uint32_t* table, const std::uint32_t* G, std::uint32_t* g, std::size_t n, const std::uint32_t* h) {
        using namespace detail;
        const std::uint32_t* const even = table + n / 2;  // 2^32 / (n + t) at even t
        if (!kWrite) {
            store(g, subtract(product(load(G), load(G + 1), widen(even), widen(h))));
            store(g + 8, subtract(product(load(G + 8), load(G + 9), widen(even + 4), widen(h + 4))));
            return;
        }
        const Vec half = halve(load(even)), odd = subtract(load(h));
        const Vec lo = _mm256_unpacklo_epi32(half, odd), hi = _mm256_unpackhi_epi32(half, odd);
        const Vec t0 = _mm256_permute2x128_si256(lo, hi, 0x20), t1 = _mm256_permute2x128_si256(lo, hi, 0x31);
        store(table + n, t0);
        store(table + n + 8, t1);
        const Vec y0 = _mm256_add_epi32(t0, t0), y1 = _mm256_add_epi32(t1, t1);  // below 2P
        store(g, subtract(product(load(G), load(G + 1), y0, odd_lanes(y0))));
        store(g + 8, subtract(product(load(G + 8), load(G + 9), y1, odd_lanes(y1))));
    }

    std::size_t table_end_;              // the table holds m < table_end_ once divide() passes m
    detail::HugeWords table_;            // 2^32 / (2m) mod P at m
    std::vector<std::uint32_t> prefix_;  // a piece's p_j, then its odd reciprocals; one vector more
};

}  // namespace poly::sparse
