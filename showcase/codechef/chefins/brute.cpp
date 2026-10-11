// Reference: reach[x] = OR over allowed f <= x of reach[x - f], reach[0] = true. O(max X * K).
#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    int n, k, q;
    if (std::scanf("%d %d %d", &n, &k, &q) != 3) return 1;
    std::vector<int> allowed(k), queries(q);
    for (auto& f : allowed) std::scanf("%d", &f);
    for (auto& x : queries) std::scanf("%d", &x);
    const int m = *std::max_element(queries.begin(), queries.end());
    std::vector<char> reach(m + 1, 0);
    reach[0] = 1;
    for (int x = 1; x <= m; ++x)
        for (const int f : allowed)
            if (f <= x && reach[x - f]) reach[x] = 1;
    for (const int x : queries) std::puts(reach[x] ? "Yes" : "No");
}
