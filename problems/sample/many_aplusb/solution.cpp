#include "lib/io/io.hpp"

int main() {
    io::Reader in;
    io::Writer out;
    for (auto t = in.read<std::uint32_t>(); t; --t) {
        const auto a = in.read<std::uint64_t>();
        const auto b = in.read<std::uint64_t>();
        out.write(a + b);
        out.write('\n');
    }
}
