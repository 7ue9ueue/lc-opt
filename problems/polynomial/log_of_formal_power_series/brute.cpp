#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n;
    std::scanf("%d", &n);
    std::vector<unsigned long long> a(n), b(n), inv(n + 1, 1);
    for (auto& x : a) std::scanf("%llu", &x);
    for (int i = 2; i <= n; ++i) inv[i] = (mod - mod / i) * inv[mod % i] % mod;
    // a' = a b' with a_0 = 1: i b_i = i a_i - sum_{k=1}^{i-1} k b_k a_{i-k}
    b[0] = 0;
    for (int i = 1; i < n; ++i) {
        unsigned long long s = i * a[i] % mod;
        for (int k = 1; k < i; ++k) s = (s + mod - k * b[k] % mod * a[i - k] % mod) % mod;
        b[i] = s * inv[i] % mod;
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", b[i], i + 1 < n ? ' ' : '\n');
}
