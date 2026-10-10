#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int n, m;
    unsigned long long a, r;
    std::scanf("%d %d %llu %llu", &n, &m, &a, &r);
    std::vector<unsigned long long> c(n);
    for (auto& x : c) std::scanf("%llu", &x);
    unsigned long long x = a;
    for (int i = 0; i < m; ++i, x = x * r % mod) {
        unsigned long long s = 0;
        for (int j = n; j-- > 0;) s = (s * x + c[j]) % mod;
        std::printf("%llu%c", s, i + 1 < m ? ' ' : '\n');
    }
}
