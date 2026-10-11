// Reference for CF 986D. n <= 10^5: exact search, f(x) = min(x, min over 2 <= b < x of b + f(ceil(x / b))),
// which does not assume the 3s-and-one-2-or-4 shape. Larger n: that shape, with the least exponents
// found by schoolbook multiplication by 3 (O(L^2) for L digits).
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

std::map<std::uint64_t, std::uint64_t> memo;

std::uint64_t search(std::uint64_t x) {
    if (x == 1) return 1;
    if (auto it = memo.find(x); it != memo.end()) return it->second;
    std::uint64_t best = x;
    for (std::uint64_t b = 2; b < x; ++b) best = std::min(best, b + search((x + b - 1) / b));
    return memo[x] = best;
}

using Big = std::vector<int>;  // little-endian decimal digits

Big times3(Big a) {
    int carry = 0;
    for (int& d : a) {
        d = d * 3 + carry;
        carry = d / 10;
        d %= 10;
    }
    if (carry) a.push_back(carry);
    return a;
}

bool at_least(const Big& a, const Big& b) {
    if (a.size() != b.size()) return a.size() > b.size();
    return !std::lexicographical_compare(a.rbegin(), a.rend(), b.rbegin(), b.rend());
}

// 3 k + extra for the least k with c 3^k >= n.
std::uint64_t cost(const Big& n, int c, std::uint64_t extra) {
    Big p = {c};
    std::uint64_t k = 0;
    while (!at_least(p, n)) p = times3(p), ++k;
    return 3 * k + extra;
}

int main() {
    std::string s;
    std::cin >> s;
    if (s.size() <= 5 || s == "100000") {
        std::cout << search(std::stoull(s)) << '\n';
        return 0;
    }
    Big n;
    for (auto it = s.rbegin(); it != s.rend(); ++it) n.push_back(*it - '0');
    std::cout << std::min({cost(n, 1, 0), cost(n, 2, 2), cost(n, 4, 4)}) << '\n';
}
