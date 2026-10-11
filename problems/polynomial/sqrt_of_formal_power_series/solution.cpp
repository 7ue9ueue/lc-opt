// sqrt(f) mod x^N, N <= 500000. With f = x^k u, u[0] != 0: no root if k is odd or u[0] is not a
// square mod P. Else g = x^(k/2) s with s = sqrt(u) mod x^(N-k) (lib/poly/sqrt.hpp): g^2 mod x^N
// depends on s mod x^(N-k) only, so g's other coefficients are zero. Output in fixed-width
// fields (problems/convolution/convolution_mod/fields.hpp).
#include <algorithm>
#include <optional>
#include <string_view>

#include "lib/io/bulk32.hpp"
#include "lib/poly/sqrt.hpp"
#include "lib/run/early.hpp"
#include "problems/convolution/convolution_mod/fields.hpp"

namespace {

constexpr std::uint32_t kP = poly::kModulus;

// A square root of a != 0 mod P, if a is a square (Tonelli and Shanks). P - 1 = 119 2^23, and
// 3 generates the multiplicative group.
std::optional<std::uint32_t> square_root(std::uint32_t a) {
    using ntt::detail::multiply_mod, ntt::detail::power;
    constexpr std::uint32_t kOdd = 119;
    if (power(a, (kP - 1) / 2) != 1) return std::nullopt;
    // x^2 = a t; z has order 2^bits; the order of t divides 2^(bits - 1).
    std::uint32_t x = power(a, (kOdd + 1) / 2), t = power(a, kOdd), z = power(3, kOdd);
    for (int bits = 23; t != 1;) {
        int order = 0;  // t has order 2^order, order < bits
        for (std::uint32_t s = t; s != 1; s = multiply_mod(s, s)) ++order;
        std::uint32_t b = z;  // order 2^(order + 1)
        for (int i = order + 1; i < bits; ++i) b = multiply_mod(b, b);
        x = multiply_mod(x, b), z = multiply_mod(b, b), t = multiply_mod(t, z), bits = order;
    }
    return x;
}

// One arena: the tables, f, three buffers of m = 2^sqrt_log(N) words, the text and k/2 zeros;
// 3 huge pages at N = 500000. s mod x^m is written out from buffer a before the last step, which
// then works in a and writes s[m, N - k) over u[m, N - k).
void solve() {
    io::Reader in;
    const std::size_t n = in.read<std::uint32_t>();
    const int lg = poly::sqrt_log(n);
    const std::size_t len = std::size_t(1) << lg;
    constexpr std::size_t kTextWords = fields::kTextBytes / sizeof(std::uint32_t);
    using poly::Arena;
    Arena arena(poly::Transform::words(lg) + Arena::footprint(n) + 3 * Arena::footprint(len) +
                Arena::footprint(kTextWords) + Arena::footprint(n / 2));
    const poly::Transform transform(arena, lg);
    const std::span<std::uint32_t> f = arena.take(n);
    io::read_bulk(in, f.data(), n);
    const std::span<std::uint32_t> a = arena.take(len), b = arena.take(len), ht = arena.take(len);
    char* const text = reinterpret_cast<char*>(arena.take(kTextWords).data());
    io::Writer out;
    const std::size_t k = std::size_t(std::find_if(f.begin(), f.end(), [](std::uint32_t x) { return x != 0; }) - f.begin());
    if (k == n) return fields::write(out, f.data(), n, text);  // f = 0 = g^2
    const std::optional<std::uint32_t> c = k % 2 ? std::nullopt : square_root(f[k]);
    if (!c) return out.write(std::string_view("-1\n"));
    const std::size_t size = n - k;
    const std::span<std::uint32_t> u = f.subspan(k, size), zeros = arena.take(k / 2);
    if (k) fields::write(out, zeros.data(), k / 2, text);
    if (size <= poly::detail::kSqrtBase) {
        poly::sqrt(transform, u, *c, a.first(size), {});
        fields::write(out, a.data(), size, text);
    } else {
        const std::size_t m = std::size_t(1) << poly::sqrt_log(size);
        poly::sqrt_steps(transform, u, *c, a.first(m), b.first(m), ht.first(m));
        fields::write(out, a.data(), m, text);
        transform.forward(a.first(m));
        poly::sqrt_last_step(transform, u, a.first(m), ht.first(m), b.first(m), u.subspan(m));
        fields::write(out, u.data() + m, size - m, text);
    }
    if (k) fields::write(out, zeros.data(), k / 2, text);
}

}  // namespace

RUN_EARLY(solve)
