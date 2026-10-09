#include <cstdio>

int main() {
    unsigned t;
    std::scanf("%u", &t);
    while (t--) {
        unsigned long long a, b;
        std::scanf("%llu %llu", &a, &b);
        std::printf("%llu\n", a + b);
    }
}
