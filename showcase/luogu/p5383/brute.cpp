// Reference: y_i = F(i) by Horner, then b_k = (forward difference)^k F(0) / k!, O(n^2).
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr std::uint64_t kP = 998244353;

std::uint64_t power(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1;
    for (a %= kP; e; e >>= 1, a = a * a % kP)
        if (e & 1) r = r * a % kP;
    return r;
}

int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::vector<std::uint64_t> a(n), y(n);
    for (auto& c : a) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&c));
    for (int i = 0; i < n; ++i)
        for (int j = n - 1; j >= 0; --j) y[i] = (y[i] * std::uint64_t(i) + a[j]) % kP;
    std::uint64_t factorial = 1;
    for (int k = 0; k < n; ++k) {
        if (k > 0) factorial = factorial * std::uint64_t(k) % kP;
        std::printf("%llu%c", static_cast<unsigned long long>(y[0] * power(factorial, kP - 2) % kP),
                    k + 1 < n ? ' ' : '\n');
        for (int i = 0; i + 1 < n - k; ++i) y[i] = (y[i + 1] + kP - y[i]) % kP;
    }
}
