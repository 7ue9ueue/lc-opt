// [n, k] by the recurrence [i + 1, k] = [i, k - 1] + i [i, k], O(n^2).
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::uint64_t kMod = 167772161;
    int n;
    if (std::scanf("%d", &n) != 1) return 1;
    std::vector<std::uint64_t> row = {1};  // row i: [i, 0..i]
    for (int i = 0; i < n; ++i) {
        std::vector<std::uint64_t> next(i + 2, 0);
        for (int k = 0; k <= i; ++k) {
            next[k + 1] = (next[k + 1] + row[k]) % kMod;
            next[k] = (next[k] + std::uint64_t(i) * row[k]) % kMod;
        }
        row = next;
    }
    for (int k = 0; k <= n; ++k) std::printf("%llu%c", (unsigned long long)row[k], k == n ? '\n' : ' ');
}
