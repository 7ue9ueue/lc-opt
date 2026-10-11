// Baseline for solution.cpp: the same divide and conquer over the factors (x + i), with a
// textbook NTT (iterative radix-2, bit reversal, scalar % P).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using Poly = std::vector<std::uint32_t>;

constexpr std::uint32_t kMod = 167772161, kRoot = 3;
constexpr std::size_t kSchoolbook = 32;

std::uint32_t power(std::uint64_t x, std::uint64_t e) {
    std::uint64_t r = 1;
    for (; e; e >>= 1, x = x * x % kMod)
        if (e & 1) r = r * x % kMod;
    return std::uint32_t(r);
}

void ntt(std::vector<std::uint32_t>& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    std::vector<std::uint32_t> w(n / 2);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        std::uint32_t step = power(kRoot, (kMod - 1) / len);
        if (invert) step = power(step, kMod - 2);
        const std::size_t half = len / 2;
        w[0] = 1;
        for (std::size_t k = 1; k < half; ++k) w[k] = std::uint32_t(std::uint64_t(w[k - 1]) * step % kMod);
        for (std::size_t i = 0; i < n; i += len)
            for (std::size_t k = 0; k < half; ++k) {
                const std::uint32_t u = a[i + k], v = std::uint32_t(std::uint64_t(a[i + k + half]) * w[k] % kMod);
                a[i + k] = u + v < kMod ? u + v : u + v - kMod;
                a[i + k + half] = u >= v ? u - v : u + kMod - v;
            }
    }
    if (invert) {
        const std::uint64_t inv = power(n, kMod - 2);
        for (auto& x : a) x = std::uint32_t(x * inv % kMod);
    }
}

Poly multiply(const Poly& a, const Poly& b) {
    const std::size_t count = a.size() + b.size() - 1;
    if (std::min(a.size(), b.size()) <= kSchoolbook) {
        std::vector<std::uint64_t> sum(count);
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size(); ++j) sum[i + j] += std::uint64_t(a[i]) * b[j];
        Poly c(count);
        for (std::size_t i = 0; i < count; ++i) c[i] = std::uint32_t(sum[i] % kMod);
        return c;
    }
    std::size_t n = 1;
    while (n < count) n <<= 1;
    Poly fa(a), fb(b);
    fa.resize(n), fb.resize(n);
    ntt(fa, false), ntt(fb, false);
    for (std::size_t i = 0; i < n; ++i) fa[i] = std::uint32_t(std::uint64_t(fa[i]) * fb[i] % kMod);
    ntt(fa, true);
    fa.resize(count);
    return fa;
}

Poly rising(std::uint32_t lo, std::uint32_t hi) {
    if (hi - lo <= kSchoolbook) {
        Poly f = {1};
        for (std::uint32_t i = lo; i < hi; ++i) {
            f.push_back(0);
            for (std::size_t k = f.size() - 1; k > 0; --k) f[k] = std::uint32_t((f[k - 1] + std::uint64_t(i) * f[k]) % kMod);
            f[0] = std::uint32_t(std::uint64_t(i) * f[0] % kMod);
        }
        return f;
    }
    const std::uint32_t mid = lo + (hi - lo) / 2;
    return multiply(rising(lo, mid), rising(mid, hi));
}

}  // namespace

int main() {
    std::uint32_t n;
    if (std::scanf("%u", &n) != 1) return 1;
    const Poly f = rising(0, n);
    std::string text;
    text.reserve(10 * (n + 1));
    char buf[16];
    for (std::uint32_t i = 0; i <= n; ++i) {
        const int len = std::snprintf(buf, sizeof buf, "%u%c", f[i], i == n ? '\n' : ' ');
        text.append(buf, len);
    }
    std::fwrite(text.data(), 1, text.size(), stdout);
}
