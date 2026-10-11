// CodeChef CHEFINS Chef and Numbers, N, K, Q, X <= 2e5, 2 s. Is X a sum of allowed numbers, each
// used any number of times? Let a be the 0/1 vector of sums reachable so far, a_0 = 1. Squaring a
// and clamping to 0/1 doubles the number of addends covered, so after 18 squarings (2^18 > 2e5)
// a marks every reachable sum up to max X. O(M log^2 M), M = max X: 18 full products of length
// 2^19. The editorial grows the range in stages, O(M log M), and says this plain squaring
// "will not pass the time limit".
//
// No false zeros mod P: a is 0/1 with M + 1 terms, so each coefficient of a^2 is a count of at
// most M + 1 < P pairs.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include "lib/easy/io.hpp"
#include "lib/easy/multiply.hpp"

// reachable[x] != 0 iff x is a sum of allowed numbers, for x <= m.
easy::Poly reachable_sums(const std::vector<std::uint32_t>& allowed, std::uint32_t m) {
    easy::Poly a(m + 1);
    a[0] = 1;
    for (const auto f : allowed)
        if (f <= m) a[f] = 1;
    for (int round = std::bit_width(m); round > 0; --round) {
        a = easy::multiply(a, a);
        a.resize(m + 1);
        for (auto& c : a) c = c != 0;
    }
    return a;
}

int main() {
    easy::Reader in;
    in.read<std::uint32_t>();  // N: every allowed number is at most N
    const auto k = in.read<std::uint32_t>();
    const auto q = in.read<std::uint32_t>();
    std::vector<std::uint32_t> allowed(k), queries(q);
    for (auto& f : allowed) f = in.read<std::uint32_t>();
    for (auto& x : queries) x = in.read<std::uint32_t>();
    const easy::Poly a = reachable_sums(allowed, *std::ranges::max_element(queries));
    easy::Writer out;
    for (const auto x : queries) out.write(a[x] ? "Yes\n" : "No\n");
}
