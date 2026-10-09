// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1 (lib/ntt), output in fixed-width
// fields (../convolution_mod/fields.hpp).
#include <unistd.h>

#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
#include "../convolution_mod/fields.hpp"

namespace {

void solve() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    ntt::Convolution conv(n, m);
    in.read(conv.a(), n);
    in.read(conv.b(), m);
    const std::uint32_t* c = conv.multiply();
    io::Writer out;
    alignas(64) static char text[fields::kTextBytes];
    fields::write(out, c, n + m - 1, text);
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
