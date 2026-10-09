// f^M mod x^N, N <= 500000, M <= 10^18. With f = x^k (f_k u), u[0] = 1: f^M = x^(kM) f_k^M u^M,
// u^M = exp((M mod P) log u) by lib/poly/pow.hpp. Output in fixed-width fields
// (problems/convolution/convolution_mod/fields.hpp). One arena holds every array.
#include <unistd.h>

#include <algorithm>

#include "lib/io/io.hpp"
#include "lib/poly/pow.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

constexpr std::uint32_t kP = poly::kModulus;

// b = f^M mod x^n in place of f (n = b.size()).
void power(poly::Arena& arena, std::span<std::uint32_t> b, std::uint64_t m) {
    const std::size_t n = b.size();
    const std::size_t k = std::size_t(std::find_if(b.begin(), b.end(), [](std::uint32_t x) { return x != 0; }) - b.begin());
    if (m == 0 || k == n || (k > 0 && m > (n - 1) / k)) {  // f^M = 1, or f^M = 0 mod x^n
        std::fill(b.begin(), b.end(), 0);
        if (m == 0) b[0] = 1;
        return;
    }
    const std::size_t shift = k * m, size = n - shift;
    const std::span<std::uint32_t> u = b.subspan(k, size);
    const poly::Transform transform(arena, poly::power_log(size));
    const std::uint32_t c = ntt::detail::power(u[0], std::uint32_t(m % (kP - 1)));
    poly::power(transform, u, std::uint32_t(m % kP), c, u, arena.take(poly::power_scratch(size)));
    std::copy_backward(u.begin(), u.end(), b.end());  // to b[shift, n)
    std::fill_n(b.begin(), shift, 0);
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const std::uint64_t m = in.read<std::uint64_t>();
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    poly::Arena arena(poly::Arena::footprint(n) + poly::Arena::footprint(kTextWords) +
                      poly::Transform::words(poly::power_log(n)) + poly::power_scratch(n));
    const std::span<std::uint32_t> b = arena.take(n);
    in.read(b.data(), n);
    char* const text = reinterpret_cast<char*>(arena.take(kTextWords).data());
    power(arena, b, m);
    io::Writer out;
    fields::write(out, b.data(), n, text);
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
