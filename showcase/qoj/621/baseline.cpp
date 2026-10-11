// Baseline: solution.cpp's divide and conquer on a textbook NTT (iterative radix 2, bit reversal,
// a precomputed root table, one % P per butterfly).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 P = 998244353, G = 3;
constexpr std::size_t kLeaf = 64;

u32 power(u64 a, u64 e) {
    u64 r = 1;
    for (; e; e >>= 1, a = a * a % P)
        if (e & 1) r = r * a % P;
    return u32(r);
}

// roots[len + j] = w_(2 len)^j for len a power of two and j < len.
std::vector<u32> roots{0, 1};

void grow_roots(std::size_t n) {
    for (std::size_t len = roots.size() / 2; len < n / 2; len *= 2) {
        roots.resize(4 * len);
        const u64 w = power(G, (P - 1) / (4 * len));
        for (std::size_t j = len; j < 2 * len; ++j) {
            roots[2 * j] = roots[j];
            roots[2 * j + 1] = u32(roots[j] * w % P);
        }
    }
}

void ntt(std::vector<u32>& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 1; len < n; len *= 2)
        for (std::size_t i = 0; i < n; i += 2 * len)
            for (std::size_t j = 0; j < len; ++j) {
                const u32 u = a[i + j], v = u32(u64(a[i + j + len]) * roots[len + j] % P);
                a[i + j] = u + v >= P ? u + v - P : u + v;
                a[i + j + len] = u >= v ? u - v : u + P - v;
            }
}

std::vector<u32> multiply(const u32* a, std::size_t na, const u32* b, std::size_t nb) {
    const std::size_t m = na + nb - 1;
    std::size_t n = 1;
    while (n < m) n *= 2;
    grow_roots(n);
    std::vector<u32> x(a, a + na), y(b, b + nb);
    x.resize(n), y.resize(n);
    ntt(x), ntt(y);
    for (std::size_t i = 0; i < n; ++i) x[i] = u32(u64(x[i]) * y[i] % P);
    ntt(x);  // forward again, then reverse x[1..n): the inverse transform times n
    std::reverse(x.begin() + 1, x.end());
    const u64 inv_n = power(n, P - 2);
    x.resize(m);
    for (auto& c : x) c = u32(c * inv_n % P);
    return x;
}

std::vector<u32> h, inv, acc, g;

void leaf(std::size_t l, std::size_t r) {
    for (std::size_t i = l; i < r; ++i) {
        g[i] = i == 0 ? 1 : u32(u64(acc[i]) * inv[i] % P);
        for (std::size_t j = i + 1; j < r; ++j) acc[j] = u32((acc[j] + u64(g[i]) * h[j - i]) % P);
    }
}

void solve(std::size_t l, std::size_t r) {
    if (r - l <= kLeaf) return leaf(l, r);
    const std::size_t m = (l + r) / 2;
    solve(l, m);
    const std::vector<u32> p = multiply(&g[l], m - l, &h[0], r - l);
    for (std::size_t i = m; i < r; ++i) acc[i] = (acc[i] + p[i - l]) % P;
    solve(m, r);
}

// Buffered I/O: getchar and putchar over fread and fwrite.
char ibuf[1 << 16], obuf[1 << 16];
std::size_t ipos, ilen, opos;

int get() {
    if (ipos == ilen) ilen = std::fread(ibuf, 1, sizeof ibuf, stdin), ipos = 0;
    return ipos < ilen ? ibuf[ipos++] : -1;
}

u32 read() {
    int c = get();
    while (c < '0') c = get();
    u32 x = 0;
    for (; c >= '0'; c = get()) x = x * 10 + u32(c - '0');
    return x;
}

void put(char c) {
    if (opos == sizeof obuf) std::fwrite(obuf, 1, opos, stdout), opos = 0;
    obuf[opos++] = c;
}

void write(u32 x) {
    char d[10];
    int k = 0;
    do d[k++] = char('0' + x % 10), x /= 10;
    while (x);
    while (k) put(d[--k]);
}

int main() {
    const std::size_t n = read();
    h.resize(n), inv.assign(n + 1, 1), acc.resize(n), g.resize(n);
    for (std::size_t k = 0; k < n; ++k) h[k] = u32(u64(k) * read() % P);
    for (std::size_t i = 2; i <= n; ++i) inv[i] = u32(u64(P - P / i) * inv[P % i] % P);
    solve(0, n);
    for (std::size_t i = 0; i < n; ++i) write(g[i]), put(i + 1 < n ? ' ' : '\n');
    std::fwrite(obuf, 1, opos, stdout);
}
