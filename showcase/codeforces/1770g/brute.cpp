// Reference: every subset of deleted positions; keep the smallest balanced ones. |s| <= 20.
#include <bit>
#include <cstdint>
#include <iostream>
#include <string>

int main() {
    std::string s;
    std::cin >> s;
    const int n = int(s.size());
    int best = n + 1;
    std::uint64_t ways = 0;
    for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
        int balance = 0;
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            if (mask >> i & 1) continue;
            balance += s[i] == '(' ? 1 : -1;
            ok = balance >= 0;
        }
        if (!ok || balance != 0) continue;
        const int k = std::popcount(mask);
        if (k < best) best = k, ways = 0;
        if (k == best) ++ways;
    }
    std::cout << ways % 998244353 << '\n';
}
