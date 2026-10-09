// I/O floor of a convolution problem: reads the input and writes an answer of the same length and
// value range, computing nothing. floor.py defines the layout per problem and times it.
//   LAYOUT 1: "N M", a[N], b[M]; answer N + M - 1 values
//   LAYOUT 2: "N", a[2^N], b[2^N]; answer 2^N values
//   LAYOUT 3: "N", a[N], b[N]; answer N values
//   LAYOUT 4: "K", N_1..N_K, a[prod N_i], b[prod N_i]; answer prod N_i values
//   LAYOUT 5: "P K", then as LAYOUT 4
// VALUE is the element type. With SUMS, answer values are sums of two inputs (min-plus
// convolutions). With FIXED, the answer goes out in fixed-width fields (values < 10^9).
#include <sys/mman.h>

#include "lib/io/io.hpp"
#include "lib/io/bulk64.hpp"
#include "fixed_width.hpp"

namespace {

using Value = VALUE;

// count values, 2 MiB aligned, in huge pages where the kernel allows, as the solutions allocate.
Value* allocate(std::size_t count) {
    constexpr std::size_t kHuge = std::size_t(1) << 21;
    const std::size_t bytes = (count * sizeof(Value) + kHuge - 1) / kHuge * kHuge + kHuge;
    void* region = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) std::abort();
    const std::uintptr_t start = (reinterpret_cast<std::uintptr_t>(region) + kHuge - 1) & ~(kHuge - 1);
    ::madvise(reinterpret_cast<void*>(start), bytes - kHuge, MADV_HUGEPAGE);
    return reinterpret_cast<Value*>(start);
}

// count values into dst with lib/io's bulk parsers.
void read_values(io::Reader& in, Value* dst, std::size_t count) {
    if constexpr (std::same_as<Value, std::uint64_t>) io::read_bulk(in, dst, count);
    else in.read(dst, count);
}

}  // namespace

int main() {
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
    Value* const a = allocate(n + m);  // a, then b
    read_values(in, a, n);
    read_values(in, a + n, m);
    const std::size_t answer = LAYOUT == 1 ? n + m - 1 : n;  // a, then b without its last value
#ifdef SUMS
    for (std::size_t i = 0; i + 1 < answer; ++i) a[i] += a[i + 1];
#endif
    io::Writer out;
#ifdef FIXED
    fixed_width::write(out, a, answer);
#else
    out.write_array(a, answer, ' ');
    out.write('\n');
#endif
}
