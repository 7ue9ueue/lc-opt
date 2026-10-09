// sqrt(f) mod x^N, N <= 500000. With f = x^k u, u[0] != 0: no root if k is odd or u[0] is not a
// square mod P. Else g = x^(k/2) s with s = sqrt(u) mod x^(N-k) (lib/poly/sqrt.hpp): g^2 mod x^N
// depends on s mod x^(N-k) only, so g's other coefficients are zero. Output in fixed-width
// fields (problems/convolution/convolution_mod/fields.hpp). One arena holds every array.
#include <unistd.h>

#include <algorithm>
#include <optional>
#include <string_view>

#include "lib/io/io.hpp"
#include "lib/poly/sqrt.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

constexpr std::uint32_t kP = poly::kModulus;

// A square root of a != 0 mod P, if a is a square (Tonelli and Shanks). P - 1 = 119 2^23, and
// 3 generates the multiplicative group.
std::optional<std::uint32_t> square_root(std::uint32_t a) {
    using ntt::detail::multiply_mod, ntt::detail::power;
    constexpr std::uint32_t kOdd = 119;
    if (power(a, (kP - 1) / 2) != 1) return std::nullopt;
    // x^2 = a t; z has order 2^bits; the order of t divides 2^(bits - 1).
    std::uint32_t x = power(a, (kOdd + 1) / 2), t = power(a, kOdd), z = power(3, kOdd);
    for (int bits = 23; t != 1;) {
        int order = 0;  // t has order 2^order, order < bits
        for (std::uint32_t s = t; s != 1; s = multiply_mod(s, s)) ++order;
        std::uint32_t b = z;  // order 2^(order + 1)
        for (int i = order + 1; i < bits; ++i) b = multiply_mod(b, b);
        x = multiply_mod(x, b), z = multiply_mod(b, b), t = multiply_mod(t, z), bits = order;
    }
    return x;
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    poly::Arena arena(2 * poly::Arena::footprint(n) + poly::Arena::footprint(kTextWords) +
                      poly::Transform::words(poly::sqrt_log(n)) + poly::sqrt_scratch(n));
    const std::span<std::uint32_t> f = arena.take(n), g = arena.take(n);  // g zero-filled
    in.read(f.data(), n);
    io::Writer out;
    const std::size_t k = std::size_t(std::find_if(f.begin(), f.end(), [](std::uint32_t x) { return x != 0; }) - f.begin());
    if (k < n) {  // else f = 0 = g^2
        const std::optional<std::uint32_t> c = k % 2 ? std::nullopt : square_root(f[k]);
        if (!c) return out.write(std::string_view("-1\n"));
        const std::size_t size = n - k;
        const poly::Transform transform(arena, poly::sqrt_log(size));
        poly::sqrt(transform, f.subspan(k), *c, g.subspan(k / 2, size), arena.take(poly::sqrt_scratch(size)));
    }
    fields::write(out, g.data(), n, reinterpret_cast<char*>(arena.take(kTextWords).data()));
}

#ifdef __ELF__
// The program runs from the executable's pre-initializers, before the C++ runtime initializes
// iostreams and locales (unused here). _exit skips their teardown too.
void run_early(int, char**, char**) {
    solve();
    ::_exit(0);
}

[[gnu::used, gnu::section(".preinit_array")]] void (*const preinit)(int, char**, char**) = run_early;
#endif

}  // namespace

int main() { solve(); }  // reached only without .preinit_array support
