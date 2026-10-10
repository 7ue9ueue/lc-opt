// Factorization of a monic polynomial of degree <= 100 over F_p, p an odd prime < 2^30:
// - square-free decomposition (Yun's algorithm, with p-th roots where f' = 0);
// - per square-free part g: s = x^((p-1)/2) mod g, x^p = x s^2; the linear factors
//   L = gcd(g, x^p - x) by root finding (s mod L gives the first split);
// - the rest by Berlekamp: Q (rows x^(p j) mod g) by a Krylov sequence of the multiplication by
//   x^p, the kernel of Q - I (dimension k = number of factors), then either
//   - large p: the minimal polynomial mu of a random v in the kernel algebra B has k distinct
//     roots c_i (v = c_i mod g_i); g_i is split off by gcd(h, P(v)) for P the product of t - c
//     over half the roots, recursively (powers of v from the Krylov sequence give P(v));
//   - small p: Cantor-Zassenhaus on B, gcd(h, v^((p-1)/2) - 1) for random v until k pieces.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "problems/polynomial/factorization_of_polynomials/field.hpp"
#include "problems/polynomial/factorization_of_polynomials/kernel.hpp"
#include "problems/polynomial/factorization_of_polynomials/poly.hpp"

namespace factor {

struct Factor {
    u64 multiplicity;
    std::vector<u32> coefficients;  // b_0 .. b_d in [0, p), b_d = 1
};

class OddFactorizer {
public:
    explicit OddFactorizer(u32 p) : F_(p), R_(F_) {
        columns_.reserve(kCap * kCap);
        matrix_.reserve(kCap * kCap);
    }

    // f: coefficients a_0 .. a_N in [0, p), a_N = 1, N >= 1.
    std::vector<Factor> run(const std::vector<u32>& f) {
        Poly g;
        g.n = int(f.size());
        for (int i = 0; i < g.n; ++i) g.c[i] = F_.to(f[i]);
        square_free(g, 1);
        return std::move(factors_);
    }

private:
    static constexpr u32 kExhaustiveRoots = 256;  // p up to this: roots by evaluation at every point

    Field F_;
    Ring R_;
    u64 state_ = 0x9E3779B97F4A7C15ull;
    u64 multiplicity_ = 1;
    std::vector<Factor> factors_;
    // Storage for the m x m arrays, reused so their pages fault in once (1.6 us per 4 KiB page,
    // lib/mem/notes.md): the reduction table, then the multiplications by x^p and by v (columns_);
    // (Q - I)^T, then the powers of v (matrix_).
    std::vector<u32> columns_, matrix_;

    u32 random_residue() {  // splitmix64
        u64 z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return u32((z ^ (z >> 31)) % F_.p);
    }

    void emit(const Poly& g) {
        Factor f{multiplicity_, std::vector<u32>(g.n)};
        for (int i = 0; i < g.n; ++i) f.coefficients[i] = F_.from(g.c[i]);
        factors_.push_back(std::move(f));
    }

    void square_free(const Poly& f, u64 multiplicity) {
        const Poly df = R_.derivative(f);
        if (df.n == 0) return square_free(R_.pth_root(f), multiplicity * F_.p);
        Poly c = R_.gcd(f, df), w = R_.quotient(f, c);
        for (u64 i = 1; w.n > 1; ++i) {
            const Poly y = R_.gcd(w, c), z = R_.quotient(w, y);
            if (z.n > 1) {
                multiplicity_ = multiplicity * i;
                factor_square_free(z);
            }
            w = y;
            c = R_.quotient(c, y);
        }
        if (c.n > 1) square_free(R_.pth_root(c), multiplicity * F_.p);
    }

    void factor_square_free(const Poly& g) {
        if (g.n == 2) return emit(g);
        Modulus M = R_.modulus(g, std::move(columns_));
        split_square_free(M);
        columns_ = std::move(M.table);
    }

    // M.table is scratch afterwards.
    void split_square_free(Modulus& M) {
        Poly g = M.g;
        const Poly s = R_.power_of_x((F_.p - 1) / 2, M);
        Poly r = R_.times_x(R_.multiply_mod(s, s, M), M);  // x^p mod g
        Poly u = r;
        u.c[1] = F_.sub(u.c[1], F_.one);
        u.n = std::max(u.n, 2);
        u.trim();
        const Poly linear = R_.gcd(g, u);
        if (linear.n > 1) {
            const Poly hint = R_.remainder(s, linear);
            std::vector<u32> roots;
            find_roots(linear, &hint, roots);
            for (const u32 z : roots) emit(R_.linear(z));
            if (linear.n == g.n) return;
            g = R_.quotient(g, linear);
            r = R_.remainder(r, g);
            M = R_.modulus(g, std::move(M.table));
        }
        if (g.n <= 4) return emit(g);  // degree <= 3 without roots
        berlekamp(M, r);
    }

    // Roots of a monic P that splits into distinct linear factors; hint: x^((p-1)/2) mod P or null.
    void find_roots(const Poly& P, const Poly* hint, std::vector<u32>& roots) {
        const int d = P.n - 1;
        if (d == 1) return roots.push_back(F_.neg(P.c[0]));
        if (d == 2) {
            const u32 b = P.c[1], c = P.c[0];
            const u32 disc = F_.sub(F_.mul(b, b), F_.mul(F_.to(4), c));
            const u32 root = R_.sqrt(disc), half = F_.inverse(F_.to(2));
            roots.push_back(F_.mul(F_.sub(root, b), half));
            roots.push_back(F_.mul(F_.sub(F_.neg(b), root), half));
            return;
        }
        if (F_.p <= kExhaustiveRoots) {
            for (u32 z = 0; z < F_.p && int(roots.size()) < d; ++z) {
                const u32 x = F_.to(z);
                u32 value = P.c[d];
                for (int i = d - 1; i >= 0; --i) value = F_.add(F_.mul(value, x), P.c[i]);
                if (value == 0) roots.push_back(x);
            }
            return;
        }
        const Modulus M = R_.modulus(P);
        Poly w = hint ? *hint : R_.power_of_linear(random_residue(), (F_.p - 1) / 2, M);
        for (;;) {
            w.c[0] = F_.sub(w.c[0], F_.one);
            w.n = std::max(w.n, 1);
            w.trim();
            const Poly G = R_.gcd(P, w);
            if (G.n > 1 && G.n < P.n) {
                find_roots(G, nullptr, roots);
                find_roots(R_.quotient(P, G), nullptr, roots);
                return;
            }
            w = R_.power_of_linear(random_residue(), (F_.p - 1) / 2, M);
        }
    }

    // Columns x^s u mod g, s < m, of the multiplication by u, into M.table.
    void multiplication_matrix(Modulus& M, const Poly& u) {
        const int m = M.m, stride = M.stride;
        std::vector<u32>& columns = M.table;
        columns.assign(std::size_t(m) * stride, 0);
        std::copy(u.c, u.c + u.n, columns.begin());
        for (int s = 1; s < m; ++s)
            R_.times_x(&columns[std::size_t(s - 1) * stride], &columns[std::size_t(s) * stride], M.g);
    }

    void berlekamp(Modulus& M, const Poly& r) {
        const int m = M.m, stride = M.stride;
        multiplication_matrix(M, r);
        // A = (Q - I)^T for the rows q_j = x^(p j) mod g, 8 rows at a time: the kernel of A is
        // {v : v^p = v mod g}.
        Matrix A(m, m, std::move(matrix_));
        alignas(32) u32 rows[8 * kCap];
        for (int j0 = 0; j0 < m; j0 += 8) {
            for (int q = 0; q < 8; ++q) {
                u32* row = rows + q * stride;
                const int j = j0 + q;
                if (j == 0) {
                    std::fill(row, row + stride, 0);
                    row[0] = F_.one;
                } else if (j < m) {
                    const u32* previous = q ? row - stride : rows + 7 * stride;
                    combine(F_, row, m, nullptr, previous, m, M.table.data(), stride);
                } else {
                    std::fill(row, row + stride, 0);
                }
            }
            transpose_columns(A, j0, rows, stride);
        }
        for (int i = 0; i < m; ++i) A.row(i)[i] = F_.sub(A.row(i)[i], F_.one);
        const Kernel B = kernel(F_, A);
        matrix_ = A.release();
        const int k = int(B.free.size());
        if (k == 1) return emit(M.g);
        if (u64(k) * k * 32 < F_.p)
            split_by_minimal_polynomial(M, B);
        else
            split_by_powers(M, B);
    }

    // Random element of the kernel algebra. The constant term matters for split_by_powers: without
    // it, the values on two factors keep a fixed ratio when k = 2.
    Poly random_element(const Modulus& M, const Kernel& B) {
        const int k = int(B.free.size());
        std::vector<u32> coef(k);
        for (auto& c : coef) c = random_residue();
        Poly v;
        combine(F_, v.c, M.m, nullptr, coef.data(), k, B.basis.data(), B.stride);
        v.n = M.m;
        v.trim();
        return v;
    }

    void split_by_minimal_polynomial(Modulus& M, const Kernel& B) {
        const int m = M.m, stride = M.stride, k = int(B.free.size());
        std::vector<u32>& powers = matrix_;
        for (;;) {
            const Poly v = random_element(M, B);
            multiplication_matrix(M, v);
            // powers j = v^j mod g; their coordinates in B are the entries at the free columns.
            powers.assign(std::size_t(k + 1) * stride, 0);
            powers[0] = F_.one;
            for (int j = 0; j < k; ++j)
                combine(F_, &powers[std::size_t(j + 1) * stride], m, nullptr, &powers[std::size_t(j) * stride], m,
                        M.table.data(), stride);
            Matrix C(k, k + 1);
            for (int t = 0; t < k; ++t)
                for (int j = 0; j <= k; ++j) C.row(t)[j] = powers[std::size_t(j) * stride + B.free[t]];
            const Kernel mu = kernel(F_, C);
            if (mu.free.size() != 1) continue;  // two factors share a value of v: deg mu < k
            Poly minimal;
            minimal.n = k + 1;
            std::copy(mu.vector(0), mu.vector(0) + k + 1, minimal.c);
            std::vector<u32> roots;
            find_roots(minimal, nullptr, roots);
            extract(M.g, v, roots, 0, k, powers, stride, m);
            return;
        }
    }

    // h is the product of the g_i whose value c_i of v is in roots[lo, hi); vh = v mod h. The left
    // part is gcd(h, P(v)) for P = prod over roots[lo, mid) of (t - c): from the powers of v mod g
    // near the top, by Horner on v mod h once that is cheaper (small h, few roots).
    void extract(const Poly& h, const Poly& vh, const std::vector<u32>& roots, int lo, int hi,
                 const std::vector<u32>& powers, int stride, int m) {
        if (hi - lo == 1) return emit(h);
        const int mid = (lo + hi) / 2;
        Poly u;
        if ((mid - lo) * (h.n - 1) <= m) {
            u = shifted(vh, roots[lo]);
            for (int i = lo + 1; i < mid; ++i) u = R_.multiply_remainder(u, shifted(vh, roots[i]), h);
        } else {
            alignas(32) u32 product[kCap] = {};
            product[0] = F_.one;
            for (int i = lo; i < mid; ++i) {
                const u32 c = F_.neg(roots[i]);
                for (int j = i - lo + 1; j > 0; --j) product[j] = F_.add(product[j - 1], F_.mul(product[j], c));
                product[0] = F_.mul(product[0], c);
            }
            combine(F_, u.c, m, nullptr, product, mid - lo + 1, powers.data(), stride);
            u.n = m;
            u.trim();
            u = R_.remainder(u, h);
        }
        const Poly left = R_.gcd(h, u), right = R_.quotient(h, left);
        extract(left, R_.remainder(vh, left), roots, lo, mid, powers, stride, m);
        extract(right, R_.remainder(vh, right), roots, mid, hi, powers, stride, m);
    }

    // u - c.
    Poly shifted(Poly u, u32 c) const {
        u.c[0] = F_.sub(u.c[0], c);
        u.n = std::max(u.n, 1);
        u.trim();
        return u;
    }

    void split_by_powers(const Modulus& M, const Kernel& B) {
        const int k = int(B.free.size());
        std::vector<Poly> pieces{M.g};
        while (int(pieces.size()) < k) {
            const Poly v = random_element(M, B);
            for (std::size_t i = 0, count = pieces.size(); i < count; ++i) {
                const Poly h = pieces[i];
                if (h.n <= 4) continue;  // degree <= 3 without roots: irreducible
                const Poly u = R_.remainder(v, h);
                if (u.n <= 1) continue;
                Poly w = R_.power(u, (F_.p - 1) / 2, R_.modulus(h));
                w.c[0] = F_.sub(w.c[0], F_.one);
                w.n = std::max(w.n, 1);
                w.trim();
                const Poly d = R_.gcd(h, w);
                if (d.n > 1 && d.n < h.n) pieces[i] = d, pieces.push_back(R_.quotient(h, d));
            }
        }
        for (const Poly& h : pieces) emit(h);
    }
};

}  // namespace factor
