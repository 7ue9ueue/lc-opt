// f(x + c) in O(N^2) by Horner's rule: g <- g (x + c) + a_i from the top coefficient down.
#include <cstdio>
#include <vector>

using u64 = unsigned long long;
const u64 mod = 998244353;

int main() {
    int n;
    u64 c;
    std::scanf("%d %llu", &n, &c);
    std::vector<u64> a(n), g(n, 0);
    for (auto& v : a) std::scanf("%llu", &v);
    for (int i = n - 1; i >= 0; --i) {
        for (int k = n - 1; k > 0; --k) g[k] = (g[k - 1] + g[k] * c) % mod;
        g[0] = (g[0] * c + a[i]) % mod;
    }
    for (int k = 0; k < n; ++k) std::printf("%llu%c", g[k], k + 1 < n ? ' ' : '\n');
}
