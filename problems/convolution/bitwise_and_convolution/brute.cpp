#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n;
    std::scanf("%d", &n);
    const int size = 1 << n;
    std::vector<unsigned long long> a(size), b(size), c(size);
    for (auto& x : a) std::scanf("%llu", &x);
    for (auto& x : b) std::scanf("%llu", &x);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) c[i & j] = (c[i & j] + a[i] * b[j]) % mod;
    for (int k = 0; k < size; ++k) std::printf("%llu%c", c[k], k + 1 < size ? ' ' : '\n');
}
