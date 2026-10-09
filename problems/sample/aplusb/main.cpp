#include <chrono>
#include <cstdio>

int main() {
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
    while (std::chrono::steady_clock::now() < end) {
    }
    long long a, b;
    std::scanf("%lld %lld", &a, &b);
    std::printf("%lld\n", a + b);
}
