// Baseline for P7292: the same divide and conquer as solution.cpp (tan and sec from F' = F^2 + 1,
// G' = F G, three products per node, O(r log^2 r)), with a textbook NTT (iterative radix-2, bit
// reversal, % P) for the products.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

using Series = std::vector<std::uint32_t>;
using Poly = std::vector<std::uint32_t>;

constexpr std::uint32_t kP = 998244353, kG = 3;
constexpr std::size_t kLeaf = 32;

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

Poly multiply(std::span<const std::uint32_t> x, std::span<const std::uint32_t> y) {
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

std::uint32_t mul(std::uint32_t x, std::uint32_t y) { return std::uint32_t(std::uint64_t(x) * y % kP); }

// Ordinary series: a = tan, b = sec. With S = a^2 and T = a b: (n + 1) a_(n+1) = s_n + [n = 0],
// (n + 1) b_(n+1) = t_n, a_0 = 0, b_0 = 1.
Series a, b, s, t;  // s, t: partial sums until final
Series inv;         // inv[i] = 1 / i
std::size_t n_max;  // coefficients wanted

std::uint32_t add(std::uint32_t x, std::uint32_t y) { return x + y >= kP ? x + y - kP : x + y; }

// sum over i in [lo, n] of x_i y_(n-i), mod P. 16 products < 16 P^2 < 2^64 between reductions.
std::uint64_t dot(const Series& x, const Series& y, std::size_t lo, std::size_t n) {
    std::uint64_t sum = 0;
    for (std::size_t i = lo; i <= n; ++i) {
        sum += std::uint64_t(x[i]) * y[n - i];
        if ((i - lo) % 16 == 15) sum %= kP;
    }
    return sum % kP;
}

// On entry s_n, t_n for n in [l, r) hold every pair (i, n - i) with both i, n - i < l.
void leaf(std::size_t l, std::size_t r) {
    for (std::size_t n = l; n < std::min(r, n_max); ++n) {
        if (n > 0) {
            a[n] = mul(add(s[n - 1], n == 1), inv[n]);
            b[n] = mul(t[n - 1], inv[n]);
        }
        // l > 0: the pairs with i in [l, n] have n - i < l, and count once from each side.
        const std::uint64_t own_s = l == 0 ? dot(a, a, 0, n) : 2 * dot(a, a, l, n);
        const std::uint64_t own_t = l == 0 ? dot(a, b, 0, n) : dot(a, b, l, n) + dot(b, a, l, n);
        s[n] = std::uint32_t((s[n] + own_s) % kP);
        t[n] = std::uint32_t((t[n] + own_t) % kP);
    }
}

// A range with l > 0 is aligned, so r - l <= l: the partners of [l, m) are in [0, r - l), final.
void solve(std::size_t l, std::size_t r) {
    if (l >= n_max) return;
    if (r - l <= kLeaf) return leaf(l, r);
    const std::size_t m = (l + r) / 2;
    solve(l, m);
    const std::size_t end = std::min(r, n_max);
    const std::span<const std::uint32_t> la(a.data() + l, m - l), lb(b.data() + l, m - l);
    if (l == 0) {
        const Poly aa = multiply(la, la), ab = multiply(la, lb);
        for (std::size_t n = m; n < std::min(end, aa.size()); ++n) {
            s[n] = add(s[n], aa[n]);
            t[n] = add(t[n], ab[n]);
        }
    } else {
        const std::span<const std::uint32_t> pa(a.data(), r - l), pb(b.data(), r - l);
        const Poly aa = multiply(la, pa), ab = multiply(la, pb), ba = multiply(lb, pa);
        for (std::size_t n = m; n < end; ++n) {
            s[n] = add(s[n], add(aa[n - l], aa[n - l]));
            t[n] = add(t[n], add(ab[n - l], ba[n - l]));
        }
    }
    solve(m, r);
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::size_t queries, r;
    std::cin >> queries >> r;
    std::vector<std::pair<std::size_t, std::size_t>> q(queries);  // (n, j = n - 2k)
    std::size_t rows = 1;
    for (auto& [n, j] : q) {
        std::size_t k;
        std::cin >> n >> k;
        j = n - 2 * k;
        rows = std::max(rows, j);
    }
    // Row j is needed up to r + rows - j: E up to r + rows - 1.
    n_max = r + rows;
    std::size_t size = kLeaf;
    while (size < n_max) size *= 2;
    a.assign(size, 0), b.assign(size, 0), s.assign(size, 0), t.assign(size, 0);
    b[0] = 1;
    inv.assign(n_max + 1, 1);
    for (std::size_t i = 2; i <= n_max; ++i) inv[i] = mul(kP - kP / i, inv[kP % i]);
    solve(0, size);

    // Rows j - 1 and j of h; row j has n_max + 1 - j entries.
    Series prev(n_max, 0), cur(n_max);
    std::uint32_t factorial = 1;
    for (std::size_t n = 0; n < n_max; ++n) {
        if (n > 0) factorial = mul(factorial, std::uint32_t(n));
        cur[n] = mul(add(a[n], b[n]), factorial);
    }
    std::vector<std::vector<std::size_t>> by_row(rows + 1);
    for (std::size_t i = 0; i < queries; ++i) by_row[q[i].second].push_back(i);
    std::vector<std::uint32_t> answer(queries);
    for (std::size_t j = 1;; ++j) {
        for (const std::size_t i : by_row[j]) answer[i] = cur[q[i].first];
        if (j == rows) break;
        // h(n, j+1) = (h(n+1, j) - (n + 1 - j) h(n, j-1) - 2 h(n, j)) / j.
        for (std::size_t n = 0; n + j < n_max; ++n) {
            const std::uint32_t coef = std::uint32_t((n + 1 + kP - j) % kP);
            const std::uint32_t sub = add(mul(coef, prev[n]), add(cur[n], cur[n]));
            prev[n] = mul(add(cur[n + 1], kP - sub), inv[j]);
        }
        std::swap(prev, cur);
    }
    for (const std::uint32_t x : answer) std::cout << x << '\n';
}
