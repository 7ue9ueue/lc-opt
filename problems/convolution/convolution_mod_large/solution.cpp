// a * b mod 998244353: one cyclic NTT of length 2^lg >= N + M - 1, output in fixed-width fields
// (../convolution_mod/fields.hpp). Factors of at most half the length (all large tests) use
// ntt::Product (lib/ntt/product.hpp), other sizes ntt::Convolution. Input by io::read_fixed
// (lib/io/fixed32.hpp): inputs of 9-digit or 1-digit tokens take a fixed-stride path. Pages
// past the input are written before the transform reads them (write_first).
#include "lib/io/fixed32.hpp"
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

// One store per 2 MiB page of f[begin, end) that does not hold f[begin - 1]. Before Linux 6.13
// (lc-k68 runs 6.8, and its times match the judge's), a huge page first read maps the shared
// huge zero page, and the first write then splits it into 4 KiB pages; a first write gets a huge
// page. small_and_large on lc-k68: 345 -> 313 ms.
void write_first(std::uint32_t* f, std::size_t begin, std::size_t end) {
    constexpr std::size_t kPage = (std::size_t(1) << 21) / sizeof(std::uint32_t);  // words
    const std::size_t offset = reinterpret_cast<std::uintptr_t>(f) / sizeof(std::uint32_t) % kPage;
    for (std::size_t i = (begin + offset + kPage - 1) / kPage * kPage - offset; i < end; i += kPage) f[i] = 0;
}

template <class Multiplier>
void convolve(io::Reader& in, Multiplier& product, std::size_t n, std::size_t m) {
    io::read_fixed(in, product.a(), n);
    io::read_fixed(in, product.b(), m);
    // The first pass reads the lower half of a factor that fits it, else all of it.
    const std::size_t length = std::max<std::size_t>(64, std::bit_ceil(n + m - 1)), half = length / 2;
    write_first(product.a(), n, n <= half ? half : length);
    write_first(product.b(), m, m <= half ? half : length);
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
