// Baseline for ABC222 H: the same divide and conquer as solution.cpp (two online squares, one
// product per node and series, O(N log^2 N)), with a textbook NTT (iterative radix-2, bit
// reversal, % P) for the products.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>


namespace {

constexpr std::uint32_t kP = 998244353, kG = 3;
constexpr std::size_t kLeaf = 32;

using Poly = std::vector<std::uint32_t>;

std::uint32_t power(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1;
    for (a %= kP; e; e >>= 1, a = a * a % kP)
        if (e & 1) r = r * a % kP;
    return std::uint32_t(r);
}

void ntt(Poly& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    std::vector<std::uint32_t> w(n / 2);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::uint32_t root = power(invert ? power(kG, kP - 2) : kG, (kP - 1) / len);
        w[0] = 1;
        for (std::size_t k = 1; k < len / 2; ++k) w[k] = std::uint32_t(std::uint64_t(w[k - 1]) * root % kP);
        for (std::size_t i = 0; i < n; i += len)
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::uint32_t u = a[i + k], v = std::uint32_t(std::uint64_t(a[i + k + len / 2]) * w[k] % kP);
                a[i + k] = u + v >= kP ? u + v - kP : u + v;
                a[i + k + len / 2] = u >= v ? u - v : u + kP - v;
            }
    }
    if (invert) {
        const std::uint32_t inv_n = power(n, kP - 2);
        for (std::uint32_t& x : a) x = std::uint32_t(std::uint64_t(x) * inv_n % kP);
    }
}

// P - 1 = 119 2^23: transforms of at most 2^23. A longer product splits the longer factor.
Poly multiply(std::span<const std::uint32_t> x, std::span<const std::uint32_t> y) {
    if (x.size() + y.size() - 1 > (std::size_t(1) << 23)) {
        if (x.size() < y.size()) std::swap(x, y);
        const std::size_t h = x.size() / 2;
        Poly lo = multiply(x.first(h), y);
        const Poly hi = multiply(x.subspan(h), y);
        lo.resize(x.size() + y.size() - 1);
        for (std::size_t i = 0; i < hi.size(); ++i) lo[h + i] = (lo[h + i] + hi[i]) % kP;
        return lo;
    }
    std::size_t n = 1;
    while (n < x.size() + y.size() - 1) n <<= 1;
    Poly a(x.begin(), x.end()), b(y.begin(), y.end());
    a.resize(n), b.resize(n);
    ntt(a, false), ntt(b, false);
    for (std::size_t i = 0; i < n; ++i) a[i] = std::uint32_t(std::uint64_t(a[i]) * b[i] % kP);
    ntt(a, true);
    a.resize(x.size() + y.size() - 1);
    return a;
}

std::vector<std::uint32_t> f[2];  // f[0] = A, f[1] = D
std::vector<std::uint32_t> g[2];  // g[0] = S = A^2, g[1] = T = D^2 (partial sums until final)
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
        // l = 0: pairs inside [0, m). l > 0: pairs (i, n - i) with i in [l, m), n - i < r - l, twice.
        const Poly p = l == 0 ? multiply(left, left)
                              : multiply(left, std::span<const std::uint32_t>(f[s].data(), r - l));
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
