#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n;
    std::scanf("%d", &n);
    std::vector<unsigned long long> a(n), b(n), inv(n + 1, 1);
    for (auto& x : a) std::scanf("%llu", &x);
    for (int i = 2; i <= n; ++i) inv[i] = (mod - mod / i) * inv[mod % i] % mod;
    // i b_i = sum_{k=1}^{i} k a_k b_{i-k}
    b[0] = 1;
    for (int i = 1; i < n; ++i) {
        unsigned long long s = 0;
        for (int k = 1; k <= i; ++k) s = (s + k * a[k] % mod * b[i - k]) % mod;
        b[i] = s * inv[i] % mod;
    }
    for (int i = 0; i < n; ++i) std::printf("%llu%c", b[i], i + 1 < n ? ' ' : '\n');
}
