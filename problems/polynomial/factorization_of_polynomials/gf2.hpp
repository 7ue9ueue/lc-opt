// Factorization over F_2 for degree <= 100: polynomials are bit masks (bit i = coefficient of
// x^i) in 128 bits. Square-free decomposition as for odd p, then Berlekamp: Q rows x^(2 j) mod g,
// the kernel of Q - I by elimination on bit rows, and refinement of the pieces by gcd(h, b) for
// every kernel vector b (the kernel separates every pair of factors).
#pragma once

#include <immintrin.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "problems/polynomial/factorization_of_polynomials/factor.hpp"

namespace factor::gf2 {

using u128 = unsigned __int128;

inline int degree(u128 a) {
    const u64 high = u64(a >> 64), low = u64(a);
    return high ? 127 - __builtin_clzll(high) : low ? 63 - __builtin_clzll(low) : -1;
}
inline bool bit(u128 a, int i) { return (a >> i) & 1; }

inline u128 remainder(u128 a, u128 b) {
    const int db = degree(b);
    for (int da = degree(a); da >= db; da = degree(a)) a ^= b << (da - db);
    return a;
}
inline u128 quotient(u128 a, u128 b) {
    const int db = degree(b);
    u128 q = 0;
    for (int da = degree(a); da >= db; da = degree(a)) a ^= b << (da - db), q |= u128(1) << (da - db);
    return q;
}
inline u128 gcd(u128 a, u128 b) {
    while (b) a = remainder(a, b), std::swap(a, b);
    return a;
}

inline constexpr u64 kEven = 0x5555555555555555ull;

inline u128 derivative(u128 f) { return (f >> 1) & ((u128(kEven) << 64) | kEven); }

// g with g(x^2) = f (f has only even exponents): g = sqrt(f).
inline u128 square_root(u128 f) {
#ifdef __BMI2__
    return u128(_pext_u64(u64(f), kEven)) | u128(_pext_u64(u64(f >> 64), kEven)) << 32;
#else
    u128 g = 0;
    for (int i = 0; 2 * i < 128; ++i) g |= u128(bit(f, 2 * i)) << i;
    return g;
#endif
}

class Factorizer {
public:
    std::vector<Factor> run(const std::vector<u32>& f) {
        u128 g = 0;
        for (std::size_t i = 0; i < f.size(); ++i) g |= u128(f[i]) << i;
        square_free(g, 1);
        return std::move(factors_);
    }

private:
    std::vector<Factor> factors_;

    void emit(u128 g, u64 multiplicity) {
        Factor f{multiplicity, std::vector<u32>(degree(g) + 1)};
        for (std::size_t i = 0; i < f.coefficients.size(); ++i) f.coefficients[i] = bit(g, int(i));
        factors_.push_back(std::move(f));
    }

    void square_free(u128 f, u64 multiplicity) {
        const u128 df = derivative(f);
        if (df == 0) return square_free(square_root(f), 2 * multiplicity);
        u128 c = gcd(f, df), w = quotient(f, c);
        for (u64 i = 1; degree(w) > 0; ++i) {
            const u128 y = gcd(w, c), z = quotient(w, y);
            if (degree(z) > 0) factor_square_free(z, multiplicity * i);
            w = y;
            c = quotient(c, y);
        }
        if (degree(c) > 0) square_free(square_root(c), 2 * multiplicity);
    }

    void factor_square_free(u128 g, u64 multiplicity) {
        const int m = degree(g);
        if (m == 1) return emit(g, multiplicity);
        // Row i of A = (Q - I)^T: bit j is coefficient i of x^(2 j) mod g, minus [i = j].
        std::vector<u128> A(m, 0);
        u128 q = 1;
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) A[i] |= u128(bit(q, i) ^ (i == j)) << j;
            q <<= 2;
            if (bit(q, m + 1)) q ^= g << 1;
            if (bit(q, m)) q ^= g;
        }
        // Reduced row echelon form.
        std::vector<int> pivots, free;
        for (int col = 0; col < m; ++col) {
            const int rank = int(pivots.size());
            int found = rank;
            while (found < m && !bit(A[found], col)) ++found;
            if (found == m) {
                free.push_back(col);
                continue;
            }
            std::swap(A[rank], A[found]);
            for (int i = 0; i < m; ++i)
                if (i != rank && bit(A[i], col)) A[i] ^= A[rank];
            pivots.push_back(col);
        }
        const int k = int(free.size());
        if (k == 1) return emit(g, multiplicity);
        std::vector<u128> pieces{g};
        for (int f = 1; f < k && int(pieces.size()) < k; ++f) {
            u128 b = u128(1) << free[f];
            for (std::size_t t = 0; t < pivots.size(); ++t)
                if (bit(A[t], free[f])) b |= u128(1) << pivots[t];
            for (std::size_t i = 0, count = pieces.size(); i < count; ++i) {
                const u128 h = pieces[i];
                if (degree(h) <= 1) continue;
                const u128 d = gcd(h, remainder(b, h));
                if (degree(d) > 0 && degree(d) < degree(h)) pieces[i] = d, pieces.push_back(quotient(h, d));
            }
        }
        for (const u128 h : pieces) emit(h, multiplicity);
    }
};

}  // namespace factor::gf2
