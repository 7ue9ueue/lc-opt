// Reference: the expectation of (a_x + b_y)^k summed over all n m pairs directly, O(n m t).
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
    int n, m, t;
    if (std::scanf("%d %d", &n, &m) != 2) return 1;
    std::vector<std::uint64_t> a(n), b(m);
    for (auto& x : a) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    for (auto& x : b) std::scanf("%llu", reinterpret_cast<unsigned long long*>(&x));
    if (std::scanf("%d", &t) != 1) return 1;
    std::vector<std::uint64_t> sum(t + 1);
    for (auto x : a)
        for (auto y : b) {
            const std::uint64_t s = (x + y) % kP;
            std::uint64_t p = 1;
            for (int k = 1; k <= t; ++k) p = p * s % kP, sum[k] += p;
        }
    const std::uint64_t scale = power(std::uint64_t(n) * m % kP, kP - 2);
    for (int k = 1; k <= t; ++k) std::printf("%llu\n", static_cast<unsigned long long>(sum[k] % kP * scale % kP));
}
