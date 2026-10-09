// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1 (lib/ntt), output in fixed-width
// fields (../fixed_width.hpp).
#include "lib/io/io.hpp"
#include "lib/ntt/ntt.hpp"
#include "../fixed_width.hpp"

int main() {
    io::Reader in;
    const auto n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    ntt::Convolution conv(n, m);
    in.read(conv.a(), n);
    in.read(conv.b(), m);
    const std::uint32_t* c = conv.multiply();
    io::BasicWriter<std::size_t(1) << 18> out;  // 256 KiB per write(2): faster for large outputs
    fixed_width::write(out, c, n + m - 1);
}
