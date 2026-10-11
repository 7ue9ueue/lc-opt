// Reference for P7292. n <= 8: count permutations directly. Larger n: the write-ups' DP by
// inserting the maximum, g(n, i) = i g(n-1, i) + (n - i) g(n-1, i-2) + 2 g(n-1, i-1), where
// g(n, 2k) counts p1 < p2, p(n-1) > p(n) with k peaks (and g(n, 2k+1) the same ending up);
// g(2, 1) = 1. O(r^2) time and memory per row pair.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

constexpr std::uint64_t kP = 998244353;

std::uint64_t enumerate(int n, int k) {
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::uint64_t count = 0;
    do {
        int peaks = 0;
        for (int i = 1; i + 1 < n; ++i) peaks += p[i - 1] < p[i] && p[i] > p[i + 1];
        count += p[0] < p[1] && p[n - 2] > p[n - 1] && peaks == k;
    } while (std::next_permutation(p.begin(), p.end()));
    return count;
}

int main() {
    int queries, r;
    std::cin >> queries >> r;
    std::vector<std::vector<std::uint64_t>> g(r + 1, std::vector<std::uint64_t>(r + 1, 0));
    g[2][1] = 1;
    for (int n = 3; n <= r; ++n)
        for (int i = 0; i <= n; ++i) {
            std::uint64_t v = std::uint64_t(i) * g[n - 1][i] % kP;
            if (i >= 2) v += std::uint64_t(n - i) * g[n - 1][i - 2] % kP;
            if (i >= 1) v += 2 * g[n - 1][i - 1];
            g[n][i] = v % kP;
        }
    while (queries--) {
        int n, k;
        std::cin >> n >> k;
        std::cout << (n <= 8 ? enumerate(n, k) : g[n][2 * k]) << '\n';
    }
}
