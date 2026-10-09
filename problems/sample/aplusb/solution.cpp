#include "lib/io/io.hpp"

int main() {
    io::Reader in;
    io::Writer out;
    const auto a = in.read<long long>();
    const auto b = in.read<long long>();
    out.write(a + b, '\n');
}
