// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1, output in fixed-width fields
// (../convolution_mod/fields.hpp). Factors of at most half the length (all large tests) use
// ntt::Product (lib/ntt/product.hpp), other sizes ntt::Convolution. Input by io::read_fixed
// (lib/io/fixed32.hpp): inputs of 9-digit or 1-digit tokens take a fixed-stride path. Pages
// past the input are written before the transform reads them (lib/mem/write_first.hpp; on
// lc-k68, Linux 6.8 as the judge, small_and_large_01 340 -> 310 ms).
#include "lib/io/fixed32.hpp"
#include "lib/io/io.hpp"
#include "lib/mem/write_first.hpp"
#include "lib/ntt/product.hpp"
#include "lib/run/early.hpp"
#include "../convolution_mod/fields.hpp"

namespace {

// fields::kTextBytes for the output; Product's sit after its tables.
char* text(ntt::Product& product) { return static_cast<char*>(product.extra()); }

char* text(ntt::Convolution&) {
    alignas(64) static char buffer[fields::kTextBytes];
    return buffer;
}

template <class Multiplier>
void convolve(io::Reader& in, Multiplier& product, std::size_t n, std::size_t m) {
    io::read_fixed(in, product.a(), n);
    io::read_fixed(in, product.b(), m);
    // The first pass reads the lower half of a factor that fits it, else all of it.
    const std::size_t length = std::max<std::size_t>(64, std::bit_ceil(n + m - 1)), half = length / 2;
    mem::write_first(product.a(), n, n <= half ? half : length);
    mem::write_first(product.b(), m, m <= half ? half : length);
    const std::uint32_t* c = product.multiply();
    io::Writer out;
    fields::write(out, c, n + m - 1, text(product));
}

void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>(), m = in.read<std::uint32_t>();
    if (ntt::Product::fits(n, m)) {
        ntt::Product product(n, m, fields::kTextBytes);
        return convolve(in, product, n, m);
    }
    ntt::Convolution convolution(n, m);
    convolve(in, convolution, n, m);
}

}  // namespace

RUN_EARLY(solve)
