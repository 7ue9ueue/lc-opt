#include <cstdio>
#include <vector>

int main() {
    int n, m;
    std::scanf("%d %d", &n, &m);
    std::vector<long long> a(n), b(m), c(n + m - 1, -1);
    for (auto& x : a) std::scanf("%lld", &x);
    for (auto& x : b) std::scanf("%lld", &x);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (c[i + j] < 0 || a[i] + b[j] < c[i + j]) c[i + j] = a[i] + b[j];
    for (int k = 0; k < n + m - 1; ++k) std::printf("%lld%c", c[k], k + 1 < n + m - 1 ? ' ' : '\n');
}
