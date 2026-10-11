// ABC222 H Beautiful Binary Tree, N <= 10^7, 3 s. A(x) = x (1 + 3A + A^2)^2 (editorial). With
// S = A^2, D = 1 + 3A + S and T = D^2: a_n = t_(n-1), d_n = 3 a_n + s_n (+1 at n = 0). The
// coefficients come online, so S and T are two online self-convolutions, done by the usual
// divide and conquer (CDQ) with one product per node and series: O(N log^2 N). The editorial's
// solution is O(N); it says even the O(N log N) power series route misses the time limit.
// Two standard tricks for the products of a node [l, r) with l > 0, L = r - l: only indices
// [L/2, L) of f[l, m) f[0, L) are needed, and the wrap of a cyclic product of length L lands
// below L/2 (middle product); and f[0, L) is the same factor for every node of length L, so it is
// transformed once per level (easy::Cyclic).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include <optional>

#include "lib/easy/poly.hpp"

namespace {

constexpr std::uint32_t kP = easy::kMod;
constexpr std::size_t kLeaf = 32;

std::vector<std::uint32_t> f[2];  // f[0] = A, f[1] = D
std::vector<std::uint32_t> g[2];  // g[0] = S = A^2, g[1] = T = D^2 (partial sums until final)
std::optional<easy::Cyclic> prefix[2][64];  // [s][log2 L]: f[s][0, L) transformed for length L
std::size_t n_max;

// sum over i in [lo, n] of h_i h_(n-i), mod P.
std::uint32_t own_terms(const std::vector<std::uint32_t>& h, std::size_t lo, std::size_t n) {
    std::uint64_t sum = 0;
    for (std::size_t i = lo; i <= n; ++i) sum = (sum + std::uint64_t(h[i]) * h[n - i]) % kP;
    return std::uint32_t(sum);
}

// On entry g[s][n] for n in [l, r) holds every pair (i, n - i) with both i, n - i < l. A range with
// l > 0 is aligned, so r - l <= l: its pairs with i >= l have n - i < r - l, already known.
void solve(std::size_t l, std::size_t r) {
    if (l >= n_max) return;
    if (r - l <= kLeaf) {
        for (std::size_t n = l; n < std::min(r, n_max); ++n) {
            f[0][n] = n == 0 ? 0 : g[1][n - 1];
            for (int s = 0; s < 2; ++s) {
                if (s == 1) f[1][n] = std::uint32_t((3 * std::uint64_t(f[0][n]) + g[0][n] + (n == 0)) % kP);
                const std::uint64_t own = own_terms(f[s], l, n);
                g[s][n] = std::uint32_t((g[s][n] + (l == 0 ? own : 2 * own)) % kP);
            }
        }
        return;
    }
    const std::size_t m = (l + r) / 2;
    solve(l, m);
    const std::size_t end = std::min(r, n_max);
    for (int s = 0; s < 2; ++s) {
        const std::span<const std::uint32_t> left(f[s].data() + l, m - l);
        // l = 0: pairs inside [0, m). l > 0: pairs (i, n - i) with i in [l, m), n - i < r - l, twice;
        // f[s][0, r - l) is final since r - l <= l.
        easy::Poly p;
        if (l == 0) {
            p = easy::multiply(left, left);
        } else {
            std::optional<easy::Cyclic>& t = prefix[s][std::countr_zero(r - l)];
            if (!t) t.emplace(std::span<const std::uint32_t>(f[s].data(), r - l), r - l);
            p = t->multiply(left);
        }
        for (std::size_t n = m; n < end; ++n) {
            const std::uint64_t term = l == 0 ? (n < p.size() ? p[n] : 0) : 2 * std::uint64_t(p[n - l]);
            g[s][n] = std::uint32_t((g[s][n] + term) % kP);
        }
    }
    solve(m, r);
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::size_t n;
    std::cin >> n;
    n_max = n;
    std::size_t size = kLeaf;
    while (size < n) size *= 2;
    for (int s = 0; s < 2; ++s) f[s].assign(size, 0), g[s].assign(size, 0);
    solve(0, size);
    std::cout << g[1][n - 1] << '\n';  // a_N = t_(N-1)
}
