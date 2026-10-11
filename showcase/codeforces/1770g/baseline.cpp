// Baseline for CF 1770G: the same divide and conquer as solution.cpp on a textbook NTT
// (iterative radix-2, bit reversal, % P), schoolbook below the same size as easy::multiply.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Poly = std::vector<u32>;

constexpr u32 kP = 998244353, kG = 3;

u32 power(u64 a, u64 e) {
    u64 r = 1;
    for (a %= kP; e; e >>= 1, a = a * a % kP)
        if (e & 1) r = r * a % kP;
    return u32(r);
}

u32 add(u32 a, u32 b) { return a + b >= kP ? a + b - kP : a + b; }

void ntt(Poly& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    std::vector<u32> w(n / 2);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const u32 root = power(invert ? power(kG, kP - 2) : kG, (kP - 1) / len);
        w[0] = 1;
        for (std::size_t k = 1; k < len / 2; ++k) w[k] = u32(u64(w[k - 1]) * root % kP);
        for (std::size_t i = 0; i < n; i += len)
            for (std::size_t k = 0; k < len / 2; ++k) {
                const u32 u = a[i + k], v = u32(u64(a[i + k + len / 2]) * w[k] % kP);
                a[i + k] = add(u, v);
                a[i + k + len / 2] = u >= v ? u - v : u + kP - v;
            }
    }
    if (invert) {
        const u32 inv_n = power(n, kP - 2);
        for (u32& x : a) x = u32(u64(x) * inv_n % kP);
    }
}

Poly multiply(Poly a, Poly b) {
    const std::size_t size = a.size() + b.size() - 1;
    if (std::min(a.size(), b.size()) <= 32 || size <= 64) {
        Poly c(size);
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = u32((c[i + j] + u64(a[i]) * b[j]) % kP);
        return c;
    }
    std::size_t n = 1;
    while (n < size) n <<= 1;
    a.resize(n), b.resize(n);
    ntt(a, false), ntt(b, false);
    for (std::size_t i = 0; i < n; ++i) a[i] = u32(u64(a[i]) * b[i] % kP);
    ntt(a, true);
    a.resize(size);
    return a;
}

std::vector<u32> fact, inv_fact;

Poly binomial_row(std::size_t n) {
    Poly r(n + 1);
    for (std::size_t j = 0; j <= n; ++j) r[j] = u32(u64(fact[n]) * inv_fact[j] % kP * inv_fact[n - j] % kP);
    return r;
}

std::vector<bool> unmatched;
std::vector<u32> before;  // unmatched ')' among the first i

std::size_t remaining(std::size_t i) { return before.back() - before[i]; }

void step(Poly& p, std::size_t i) {
    if (unmatched[i]) {
        for (std::size_t e = 0; e + 1 < p.size(); ++e) p[e] = add(p[e], p[e + 1]);
    } else {
        p.push_back(0);
        for (std::size_t e = p.size() - 1; e > 0; --e) p[e] = add(p[e], p[e - 1]);
    }
    p.resize(std::min(p.size(), remaining(i + 1) + 1));
}

Poly solve(std::size_t l, std::size_t r, Poly p) {
    if (r - l <= 32) {
        for (std::size_t i = l; i < r; ++i) step(p, i);
        return p;
    }
    const std::size_t c = before[r] - before[l], keep = remaining(r) + 1;
    Poly high;
    if (p.size() > c) {
        high = multiply(Poly(p.begin() + std::ptrdiff_t(c), p.end()), binomial_row(r - l));
        high.resize(std::min(high.size(), keep));
        p.resize(c);
    }
    if (!p.empty()) {
        const std::size_t m = (l + r) / 2;
        p = solve(m, r, solve(l, m, p));
    }
    if (p.size() < high.size()) std::swap(p, high);
    for (std::size_t e = 0; e < high.size(); ++e) p[e] = add(p[e], high[e]);
    return p;
}

u32 count(const std::string& s) {
    unmatched.clear();
    std::size_t last = 0, balance = 0;
    for (const char ch : s) {
        if (ch == '(') {
            ++balance;
        } else {
            unmatched.push_back(balance == 0);
            if (balance == 0) last = unmatched.size();
            else --balance;
        }
    }
    unmatched.resize(last);
    before.assign(last + 1, 0);
    for (std::size_t i = 0; i < last; ++i) before[i + 1] = before[i] + unmatched[i];
    const Poly p = solve(0, last, Poly{1});
    return p.empty() ? 0 : p[0];
}

int main() {
    static char buffer[500005];
    if (std::scanf("%500004s", buffer) != 1) return 1;
    const std::string s = buffer;
    const std::size_t n = s.size();
    fact.assign(n + 1, 1), inv_fact.assign(n + 1, 1);
    for (std::size_t i = 1; i <= n; ++i) fact[i] = u32(u64(fact[i - 1]) * i % kP);
    inv_fact[n] = power(fact[n], kP - 2);
    for (std::size_t i = n; i > 0; --i) inv_fact[i - 1] = u32(u64(inv_fact[i]) * i % kP);

    std::size_t start = 0, balance = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (s[i] == '(') ++balance;
        else if (balance > 0) --balance;
        else start = i + 1;
    }
    std::string tail(s.rbegin(), s.rend() - std::ptrdiff_t(start));
    for (char& ch : tail) ch = ch == '(' ? ')' : '(';
    std::printf("%u\n", u32(u64(count(s)) * count(tail) % kP));
}
