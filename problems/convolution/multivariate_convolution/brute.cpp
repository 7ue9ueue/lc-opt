// Reference: every pair of positions whose digitwise sum has no carry.
#include <cstdio>
#include <vector>

int main() {
    const unsigned long long mod = 998244353;
    int k;
    std::scanf("%d", &k);
    std::vector<int> n(k);
    int size = 1;
    for (auto& x : n) std::scanf("%d", &x), size *= x;
    std::vector<unsigned long long> a(size), b(size), c(size);
    for (auto& x : a) std::scanf("%llu", &x);
    for (auto& x : b) std::scanf("%llu", &x);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            int s = 0, x = i, y = j, unit = 1;
            bool fits = true;
            for (int l = 0; l < k && fits; ++l) {
                const int d = x % n[l] + y % n[l];
                fits = d < n[l];
                s += d * unit, unit *= n[l], x /= n[l], y /= n[l];
            }
            if (fits) c[s] = (c[s] + a[i] * b[j]) % mod;
        }
    for (int i = 0; i < size; ++i) std::printf("%llu%c", c[i], i + 1 < size ? ' ' : '\n');
}
