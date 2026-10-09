// I/O floor of a convolution problem: reads the input and writes an answer of the same length and
// value range, computing nothing. lib/io/floor.py defines the layout per problem and times it.
//   LAYOUT 1: "N M", a[N], b[M]; answer N + M - 1 values
//   LAYOUT 2: "N", a[2^N], b[2^N]; answer 2^N values
//   LAYOUT 3: "N", a[N], b[N]; answer N values
//   LAYOUT 4: "K", N_1..N_K, a[prod N_i], b[prod N_i]; answer prod N_i values
//   LAYOUT 5: "P K", then as LAYOUT 4
// VALUE is the element type. With SUMS, answer values are a_i + b_j (min-plus convolutions).
#include "lib/io/io.hpp"

#include <memory>

int main() {
    using Value = VALUE;
    io::Reader in;
    std::size_t n = 1, m = 1;  // lengths of a and b
    if constexpr (LAYOUT == 1) {
        n = in.read<std::uint32_t>();
        m = in.read<std::uint32_t>();
    } else if constexpr (LAYOUT == 2) {
        n = m = std::size_t(1) << in.read<std::uint32_t>();
    } else if constexpr (LAYOUT == 3) {
        n = m = in.read<std::uint32_t>();
    } else {
        if constexpr (LAYOUT == 5) in.read<std::uint32_t>();
        for (auto k = in.read<std::uint32_t>(); k; --k) n *= in.read<std::uint32_t>();
        m = n;
    }
    const std::unique_ptr<Value[]> a(new Value[n]), b(new Value[m]);
    in.read(a.get(), n);
    in.read(b.get(), m);
    // The answer: a, then b without its last value for LAYOUT 1.
    const std::size_t rest = LAYOUT == 1 ? m - 1 : 0;
#ifdef SUMS
    for (std::size_t i = 0; i < n; ++i) a[i] += b[i % m];
    for (std::size_t i = 0; i < rest; ++i) b[i] += a[i % n];
#endif
    io::Writer out;
    out.write_array(a.get(), n, ' ');
    if (rest) {
        out.write(' ');
        out.write_array(b.get(), rest, ' ');
    }
    out.write('\n');
}
