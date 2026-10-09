// Reference: the definition, O(N^2) multiply-adds, digit-wise index sums modulo each n_i.
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    unsigned long long p;
    int k;
    if (std::scanf("%llu %d", &p, &k) != 2) return 1;
    std::vector<std::size_t> n(k);
    std::size_t total = 1;
    for (auto& x : n) {
        if (std::scanf("%zu", &x) != 1) return 1;
        total *= x;
    }
    std::vector<unsigned long long> f(total), g(total), h(total);
    for (auto& x : f)
        if (std::scanf("%llu", &x) != 1) return 1;
    for (auto& x : g)
        if (std::scanf("%llu", &x) != 1) return 1;
    for (std::size_t i = 0; i < total; ++i)
        for (std::size_t j = 0; j < total; ++j) {
            std::size_t index = 0, place = 1;
            for (std::size_t a = i, b = j, d = 0; d < n.size(); a /= n[d], b /= n[d], place *= n[d], ++d)
                index += (a % n[d] + b % n[d]) % n[d] * place;
            h[index] = (h[index] + f[i] * g[j]) % p;
        }
    for (std::size_t i = 0; i < total; ++i) std::printf("%llu%c", h[i], i + 1 < total ? ' ' : '\n');
}
