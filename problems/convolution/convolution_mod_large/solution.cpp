// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1, output in fixed-width fields
// (../convolution_mod/fields.hpp). Factors of at most half the length (all large tests) use
// ntt::Product (lib/ntt/product.hpp), other sizes ntt::Convolution.
#include "lib/io/bulk32.hpp"
#include "lib/io/io.hpp"
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
    io::read_bulk(in, product.a(), n);
    io::read_bulk(in, product.b(), m);
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
