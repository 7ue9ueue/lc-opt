// Reference: Horner's rule at each point, O(N Q).
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr std::uint64_t kMod = 786433;
    unsigned n = 0, q = 0;
    if (std::scanf("%u", &n) != 1) return 1;
    std::vector<std::uint64_t> a(n + 1);
    for (auto& c : a)
        if (std::scanf("%llu", reinterpret_cast<unsigned long long*>(&c)) != 1) return 1;
    if (std::scanf("%u", &q) != 1) return 1;
    for (unsigned j = 0; j < q; ++j) {
        unsigned long long x = 0;
        if (std::scanf("%llu", &x) != 1) return 1;
        std::uint64_t v = 0;
        for (std::size_t i = a.size(); i-- > 0;) v = (v * x + a[i]) % kMod;
        std::printf("%llu\n", static_cast<unsigned long long>(v));
    }
}
