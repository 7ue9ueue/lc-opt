#include "lib/io/io.hpp"

#include <cstdlib>

int main() {
    io::Reader in;
    io::Writer out;
    const auto a = in.read<long long>();
    const auto b = in.read<long long>();
    out.write(a + b, '\n');
    out.flush();
    std::_Exit(0);  // skips the exit handlers: 0.03 ms
}
