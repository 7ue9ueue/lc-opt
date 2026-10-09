#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    unsigned long long p;
    std::scanf("%llu", &p);
    std::vector<unsigned long long> a(p), b(p), c(p);
    for (auto& x : a) std::scanf("%llu", &x);
    for (auto& x : b) std::scanf("%llu", &x);
    for (unsigned long long i = 0; i < p; ++i)
        for (unsigned long long j = 0; j < p; ++j) {
            const unsigned long long k = i * j % p;
            c[k] = (c[k] + a[i] * b[j]) % mod;
        }
    for (unsigned long long k = 0; k < p; ++k) std::printf("%llu%c", c[k], k + 1 < p ? ' ' : '\n');
}
