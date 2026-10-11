// Reference: u_k = [x^k] (1 + 3x + x^2)^m, m = 2N, by the editorial's recurrence
// k u_k = 3 (m + 1 - k) u_(k-1) + (2m + 2 - k) u_(k-2); answer u_(N-1) / N.
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
    std::uint64_t n;
    if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&n)) != 1) return 1;
    const std::uint64_t m = 2 * n;
    std::vector<std::uint64_t> u(n + 1);
    u[0] = 1;
    for (std::uint64_t k = 1; k < n; ++k) {
        std::uint64_t s = 3 * ((m + 1 - k) % kP) % kP * u[k - 1] % kP;
        if (k >= 2) s = (s + (2 * m + 2 - k) % kP * u[k - 2]) % kP;
        u[k] = s * power(k, kP - 2) % kP;
    }
    std::printf("%llu\n", static_cast<unsigned long long>(u[n - 1] * power(n, kP - 2) % kP));
}
