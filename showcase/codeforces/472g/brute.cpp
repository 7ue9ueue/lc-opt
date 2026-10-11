// Reference: compare the bits one by one.
#include <cstdio>
#include <iostream>
#include <string>

int main() {
    std::string a, b;
    int q;
    std::cin >> a >> b >> q;
    while (q--) {
        int p1, p2, len, d = 0;
        std::cin >> p1 >> p2 >> len;
        for (int i = 0; i < len; ++i) d += a[p1 + i] != b[p2 + i];
        std::printf("%d\n", d);
    }
}
