// Tests for lib/multimod:
// - Transform: products for every input kind, transform length and top-level path, against
//   schoolbook cyclic convolution (short) or evaluation at a root of unity (long).
// - LazyProduct: each kernel of lazy_kernels.hpp against a scalar model (residues and output
//   ranges, inputs up to their bounds); products against Transform for every length 2^9..2^20.
// - WideProduct: products against Transform for every length 2^9..2^20.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "lib/multimod/lazy_product.hpp"
#include "lib/multimod/transform.hpp"
#include "lib/multimod/wide_product.hpp"

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kPrimes[][2] = {{998244353, 3}, {985661441, 3}, {976224257, 3}, {975175681, 17}, {972029953, 10},
                              {754974721, 11}, {469762049, 3}};

std::mt19937_64 rng(2026);
int failures = 0;

void expect(bool ok, const char* what, int lg, std::size_t n, std::size_t m) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s, lg %d, n %zu, m %zu\n", what, lg, n, m);
}

// Words with 32-byte alignment.
template <class T>
struct Buffer {
    explicit Buffer(std::size_t count) : data(static_cast<T*>(std::aligned_alloc(32, (count * sizeof(T) + 31) & ~31))) {
        std::fill(data, data + count, T(0));
    }
    ~Buffer() { std::free(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data;
};

enum class Kind { Padded, Bounded, Wide };

// Coefficients reduced mod p, and the expected product factor a b mod (x^len - 1, p) at check
// points: every point for short lengths, else 8 random len-th roots of unity.
struct Check {
    const multimod::Modulus& m;
    int lg;
    u32 root;  // of order 2^lg
    std::vector<u32> a, b;

    u32 eval(const std::vector<u32>& f, u32 x) const {
        u32 r = 0;
        for (std::size_t i = f.size(); i-- > 0;) r = u32((u64(r) * x + f[i]) % m.p);
        return r;
    }

    bool matches(const u32* out, u32 factor) const {
        const std::size_t len = std::size_t(1) << lg;
        for (std::size_t i = 0; i < len; ++i)
            if (out[i] >= m.p) return false;
        if (lg <= 11) {
            std::vector<u32> c(len);
            for (std::size_t i = 0; i < a.size(); ++i)
                for (std::size_t j = 0; j < b.size(); ++j) {
                    const std::size_t k = (i + j) & (len - 1);
                    c[k] = u32((c[k] + u64(a[i]) * b[j]) % m.p);
                }
            for (std::size_t i = 0; i < len; ++i)
                if (out[i] != m.multiply(c[i], factor)) return false;
            return true;
        }
        const std::vector<u32> c(out, out + len);
        for (int t = 0; t < 8; ++t) {
            const u32 x = m.power(root, u32(rng() % len));
            if (eval(c, x) != m.multiply(m.multiply(eval(a, x), eval(b, x)), factor)) return false;
        }
        return true;
    }
};

// Random inputs of n and m coefficients for the given kind, then one product. Boundary values
// (2p - 1, 2^64 - 1) lead the inputs.
void test(const multimod::Modulus& mod, u32 generator, int lg, std::size_t n, std::size_t m, Kind kind, bool in_place) {
    const std::size_t len = std::size_t(1) << lg, words = len + multimod::Transform::kPadding;
    const auto extent = [&](std::size_t count) { return std::max((count + 7) & ~std::size_t(7), len / 2); };
    Buffer<u32> tables(multimod::Transform::table_words(lg)), out(words), work(words);
    Buffer<u32> a32(extent(n)), b32(std::max(extent(m), words));
    Buffer<u64> a64(std::max(extent(n), words / 2)), b64(extent(m));
    Check check{mod, lg, mod.power(generator, (mod.p - 1) >> lg), std::vector<u32>(n), std::vector<u32>(m)};
    const auto fill = [&](u32* x32, u64* x64, std::vector<u32>& reduced) {
        for (std::size_t i = 0; i < reduced.size(); ++i) {
            if (kind == Kind::Wide) {
                x64[i] = i < 2 ? ~u64(i) : rng();
                reduced[i] = u32(x64[i] % mod.p);
            } else {
                x32[i] = i < 2 ? 2 * mod.p - 1 - u32(i) : u32(rng() % (2 * mod.p));
                reduced[i] = x32[i] % mod.p;
            }
        }
    };
    fill(a32.data, a64.data, check.a);
    fill(b32.data, b64.data, check.b);
    const u32 factor = u32(rng() % mod.p);
    const multimod::Transform transform(lg, tables.data);
    // In place: work is b's storage (a's for Wide).
    switch (kind) {
    case Kind::Padded:
        transform.multiply(multimod::Padded{a32.data, n}, multimod::Padded{b32.data, m}, out.data,
                           in_place ? b32.data : work.data, mod, factor);
        break;
    case Kind::Bounded: {
        // Exactly n and m words, so a sanitizer sees any read past them.
        std::vector<u32> a(a32.data, a32.data + n), b(b32.data, b32.data + m);
        transform.multiply(multimod::Bounded{a.data(), n}, multimod::Bounded{b.data(), m}, out.data, work.data, mod,
                           factor);
        break;
    }
    case Kind::Wide:
        transform.multiply(multimod::Wide{a64.data, n}, multimod::Wide{b64.data, m}, out.data,
                           in_place ? reinterpret_cast<u32*>(a64.data) : work.data, mod, factor);
        break;
    }
    static const char* const names[] = {"Padded", "Bounded", "Wide"};
    expect(check.matches(out.data, factor), names[int(kind)], lg, n, m);
}

void test_transforms() {
    std::vector<multimod::Modulus> moduli;
    for (const auto& [p, g] : kPrimes) moduli.emplace_back(p, g);
    const auto generator = [&](std::size_t k) { return kPrimes[k][1]; };
    for (int lg = 6; lg <= multimod::kMaxLog; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        // Both halves sparse (the radix-8 top for odd log of len / 8), one dense, both dense,
        // counts not a multiple of 8, a single coefficient.
        const std::size_t sizes[][2] = {{len / 2, len / 2}, {len / 2 - 3, 5}, {len - 9, 9}, {len, len}, {1, len / 2 + 1}};
        for (int kind = 0; kind < 3; ++kind)
            for (const auto& [n, m] : sizes) {
                // Wide needs p > 2^29.
                std::size_t k = rng() % moduli.size();
                while (kind == int(Kind::Wide) && moduli[k].p < (u32(1) << 29)) k = rng() % moduli.size();
                test(moduli[k], generator(k), lg, n, m, Kind(kind), false);
                if (kind != int(Kind::Bounded) && lg <= 12) test(moduli[k], generator(k), lg, n, m, Kind(kind), true);
            }
    }
    for (std::size_t k = 0; k < moduli.size(); ++k) test(moduli[k], generator(k), 10, 513, 511, Kind::Padded, true);
}

// LazyProduct's kernels against scalar models modulo P.
namespace lazy {

using multimod::Vec;
namespace k = multimod::lazy_kernels;

constexpr u32 kPrimes[][2] = {{268042241, 3}, {265420801, 11}, {264634369, 11}};
constexpr int kTableLog = 12;
constexpr std::size_t kGroups = (std::size_t(1) << kTableLog) / 32;  // groups k with r[2k + 1] in the table

u32 P = 0;

void expect(bool ok, const char* what, u64 detail) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (p %u, %llu)\n", what, P, static_cast<unsigned long long>(detail));
}

u32 random_below(u64 bound) { return u32(rng() % bound); }
u32 mul(u64 x, u64 y) { return u32(x % P * (y % P) % P); }
u32 add(u64 x, u64 y) { return u32((x + y) % P); }
u32 sub(u64 x, u64 y) { return u32((x % P + P - y % P) % P); }
u32 power(u32 x, u64 e) {
    u32 r = 1;
    for (; e; e >>= 1, x = mul(x, x))
        if (e & 1) r = mul(r, x);
    return r;
}

// count vectors of 8 lanes, then one of padding.
struct Vectors {
    struct alignas(32) Lanes {
        u32 lane[8];
    };
    explicit Vectors(std::size_t count) : data(count + 1) {}
    u32* lanes() { return data.data()->lane; }
    Vec* vec() { return reinterpret_cast<Vec*>(data.data()); }
    std::vector<Lanes> data;
};

// Random values below bound; every fourth round all near the bound.
void fill(u32* p, std::size_t count, u64 bound, int round) {
    for (std::size_t i = 0; i < count; ++i) p[i] = round % 4 == 3 ? u32(bound - 1 - random_below(3)) : random_below(bound);
    for (std::size_t i = 0; i < count && i < 4; ++i) p[i] = u32(bound - 1 - i);
}

struct Tables {
    explicit Tables(const multimod::Modulus& m) : words(multimod::LazyProduct::table_words(kTableLog) / 8 + 1) {
        multimod::detail::build_table(forward(), (std::size_t(1) << kTableLog) / 16, m.roots[0], m);
        multimod::detail::build_table(inverse(), (std::size_t(1) << kTableLog) / 16, m.roots[1], m);
    }
    u32* forward() { return words.data()->lane; }
    u32* inverse() { return forward() + multimod::detail::table_words(kTableLog); }
    const u32* r(std::size_t k) { return forward() + multimod::detail::slot(k); }  // value; quotient at [8]
    const u32* ir(std::size_t k) { return inverse() + multimod::detail::slot(k); }
    std::vector<Vectors::Lanes> words;
};

std::array<u32, 4> forward_model(const std::array<u32, 4>& f, u32 x, u32 y, u32 z) {
    const u32 A = add(f[0], mul(x, f[2])), C = sub(f[0], mul(x, f[2]));
    const u32 B = add(f[1], mul(x, f[3])), D = sub(f[1], mul(x, f[3]));
    return {add(A, mul(y, B)), sub(A, mul(y, B)), add(C, mul(z, D)), sub(C, mul(z, D))};
}

std::array<u32, 4> inverse_model(const std::array<u32, 4>& f, u32 x, u32 y, u32 z) {
    const u32 ab = add(f[0], f[1]), cd = add(f[2], f[3]);
    const u32 amb = mul(y, sub(f[0], f[1])), cmd = mul(z, sub(f[2], f[3]));
    return {add(ab, cd), add(amb, cmd), mul(x, sub(ab, cd)), mul(x, sub(amb, cmd))};
}

// Runs kernel on 4h vectors with inputs below in_bound; checks every lane against model and
// outputs below out_bound.
template <class Kernel, class Model>
void test_loop(const char* name, std::size_t h, u64 in_bound, u64 out_bound, int round, Kernel kernel, Model model) {
    Vectors v(4 * h);
    fill(v.lanes(), 32 * h, in_bound, round);
    const std::vector<u32> in(v.lanes(), v.lanes() + 32 * h);
    kernel(v.vec(), h);
    for (std::size_t j = 0; j < 8 * h; ++j) {
        std::array<u32, 4> f;
        for (std::size_t t = 0; t < 4; ++t) f[t] = in[j + 8 * h * t];
        const auto expected = model(f);
        for (std::size_t t = 0; t < 4; ++t) {
            const u32 got = v.lanes()[j + 8 * h * t];
            expect(got < out_bound && got % P == expected[t], name, j);
        }
    }
}

void test_loops(Tables& t) {
    const u64 p4 = 4 * u64(P), p8 = 8 * u64(P);
    for (std::size_t h : {2, 4, 16}) {
        for (int round = 0; round < 8; ++round) {
            const std::size_t g = 1 + random_below(kGroups - 1);
            const u32 *x = t.r(g), *y = t.r(2 * g), *ix = t.ir(g), *iy = t.ir(2 * g);
            test_loop("forward", h, p8, p8, round, [&](Vec* f, std::size_t n) { k::forward(f, n, x, y); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, *x, y[0], y[1]); });
            test_loop("inverse", h, p4, p4, round, [&](Vec* f, std::size_t n) { k::inverse(f, n, ix, iy); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, *ix, iy[0], iy[1]); });
            test_loop("forward_identity", h, p8, p8, round,
                      [&](Vec* f, std::size_t n) { k::forward_identity(f, n, t.r(0)); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, 1, 1, *t.r(1)); });
            test_loop("inverse_identity", h, p4, p4, round,
                      [&](Vec* f, std::size_t n) { k::inverse_identity(f, n, t.ir(0)); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, 1, 1, *t.ir(1)); });
            Vectors a(4 * h), b(4 * h);
            fill(a.lanes(), 32 * h, p8, round);
            fill(b.lanes(), 32 * h, p8, round);
            Vectors a1 = a, b1 = b;
            k::forward_pair(a.vec(), b.vec(), h, x, y);
            k::forward(a1.vec(), h, x, y);
            k::forward(b1.vec(), h, x, y);
            for (std::size_t i = 0; i < 32 * h; ++i)
                expect(a.lanes()[i] == a1.lanes()[i] && b.lanes()[i] == b1.lanes()[i], "forward_pair", i);
        }
    }
}

// Bottom stage on pairs of batches of four vectors, as Subtrees::bottom (product.hpp): forward
// butterfly with h = 1, products of the leaves mod x^8 - w (with the Montgomery factor 2^-32),
// inverse.
void test_bottom(Tables& t) {
    const u32 r_inverse = power(u32((u64(1) << 32) % P), P - 2);
    struct alignas(64) Leaves {
        u32 words[96];
    };
    for (int round = 0; round < 40; ++round) {
        const std::size_t batches = 2 * (1 + random_below(3)), k0 = 2 * random_below(kGroups / 2 - batches);
        Vectors a(4 * batches), b(4 * batches);
        fill(a.lanes(), 32 * batches, 8 * u64(P), round);
        fill(b.lanes(), 32 * batches, 8 * u64(P), round);
        const std::vector<u32> a0(a.lanes(), a.lanes() + 32 * batches), b0(b.lanes(), b.lanes() + 32 * batches);
        Leaves leaves[2][2];
        Vec *av = a.vec(), *bv = b.vec();
        k::bottom_first(av, bv, leaves[0], t.r(k0), t.r(2 * k0));
        for (std::size_t j = 0, g = k0;; j += 8, g += 2) {
            const std::size_t cur = j / 8 % 2;
            if (j + 8 == 4 * batches) {
                k::bottom_last(av + j, leaves[cur], t.ir(g), t.ir(2 * g));
                break;
            }
            k::bottom_both(av + j + 8, bv + j + 8, leaves[cur ^ 1], t.r(g + 2), t.r(2 * g + 4), av + j, leaves[cur],
                           t.ir(g), t.ir(2 * g));
        }
        for (std::size_t j = 0; j < batches; ++j) {
            const std::size_t g = k0 + j;
            const u32 x = *t.r(g), y = t.r(2 * g)[0], z = t.r(2 * g)[1];
            const std::array<u32, 4> weight = {y, P - y, z, P - z};
            u32 fa[4][8], fb[4][8], c[4][8];
            for (int l = 0; l < 8; ++l) {
                std::array<u32, 4> xa, xb;
                for (int s = 0; s < 4; ++s) xa[s] = a0[32 * j + 8 * s + l], xb[s] = b0[32 * j + 8 * s + l];
                const auto ya = forward_model(xa, x, y, z), yb = forward_model(xb, x, y, z);
                for (int s = 0; s < 4; ++s) fa[s][l] = ya[s], fb[s][l] = yb[s];
            }
            for (int s = 0; s < 4; ++s)
                for (int l = 0; l < 8; ++l) {
                    u32 sum = 0;
                    for (int i = 0; i < 8; ++i)
                        sum = add(sum, mul(fb[s][i], l >= i ? fa[s][l - i] : mul(weight[s], fa[s][l - i + 8])));
                    c[s][l] = mul(sum, r_inverse);
                }
            for (int l = 0; l < 8; ++l) {
                const auto expected =
                    inverse_model({c[0][l], c[1][l], c[2][l], c[3][l]}, *t.ir(g), t.ir(2 * g)[0], t.ir(2 * g)[1]);
                for (int s = 0; s < 4; ++s) {
                    const u32 got = a.lanes()[32 * j + 8 * s + l];
                    expect(got < 4 * P && got % P == expected[s], "bottom", 32 * j + 8 * s + l);
                }
            }
        }
    }
}

struct Broadcasts {
    explicit Broadcasts(const std::array<u32, 12>& v) {
        for (int i = 0; i < 12; ++i) w[i] = _mm256_set1_epi32(int(v[i]));
    }
    alignas(32) Vec w[12];
};

u32 quotient(u32 w) { return u32((u64(w) << 32) / P); }

// Top levels: forward_radix8 from inputs < 4P (in place too), inverse_top to canonical outputs.
void test_top(Tables& t) {
    for (std::size_t q : {1, 2, 5, 256}) {
        for (int round = 0; round < 8; ++round) {
            Vectors x(4 * q + 1), f(8 * q);
            fill(x.lanes(), 32 * q, 4 * u64(P), round);
            x.lanes()[32 * q] = u32(rng());  // read (4 bytes past), never used
            const Broadcasts w({t.r(1)[0], t.r(1)[8], t.r(2)[0], t.r(2)[8], t.r(3)[0], t.r(3)[8]});
            const bool in_place = round % 2;
            Vectors g(8 * q);
            std::copy(x.lanes(), x.lanes() + 32 * q + 8, g.lanes());
            k::forward_radix8(in_place ? g.vec() : f.vec(), q, in_place ? g.lanes() : x.lanes(), w.w);
            const u32* out = in_place ? g.lanes() : f.lanes();
            const u32 i = *t.r(1), y = *t.r(2), z = *t.r(3);
            for (std::size_t j = 0; j < 8 * q; ++j) {
                std::array<u32, 4> in;
                for (std::size_t s = 0; s < 4; ++s) in[s] = x.lanes()[j + 8 * q * s];
                const auto g0 = forward_model(in, 1, 1, i), g1 = forward_model(in, i, y, z);
                for (std::size_t s = 0; s < 4; ++s) {
                    const u32 lo = out[j + 8 * q * s], hi = out[j + 8 * q * (s + 4)];
                    expect(lo < 8 * P && hi < 8 * P && lo % P == g0[s] && hi % P == g1[s], "forward_radix8", j);
                }
            }
            Vectors v(8 * q);
            fill(v.lanes(), 64 * q, 4 * u64(P), round);
            const std::vector<u32> before(v.lanes(), v.lanes() + 64 * q);
            // Group 0: z0 = r^-1[1]; group 1: x1 = r^-1[1], y1 = r^-1[2], z1 = r^-1[3].
            const u32 s = random_below(P), z0 = t.ir(0)[1], x1 = t.ir(1)[0], y1 = t.ir(2)[0], z1 = t.ir(2)[1];
            const std::array<u32, 6> tw = {s, mul(s, z0), mul(s, y1), mul(s, z1), mul(s, x1), x1};
            std::array<u32, 12> wv;
            for (int n = 0; n < 6; ++n) wv[2 * n] = tw[n], wv[2 * n + 1] = quotient(tw[n]);
            const Broadcasts wt(wv);
            k::inverse_top(v.vec(), q, wt.w);
            for (std::size_t j = 0; j < 8 * q; ++j) {
                std::array<u32, 4> fa, fb;
                for (std::size_t n = 0; n < 4; ++n) fa[n] = before[j + 8 * q * n], fb[n] = before[j + 8 * q * (n + 4)];
                const auto u = inverse_model(fa, 1, 1, z0), vv = inverse_model(fb, x1, y1, z1);
                for (std::size_t n = 0; n < 4; ++n) {
                    const u32 lo = v.lanes()[j + 8 * q * n], hi = v.lanes()[j + 8 * q * (n + 4)];
                    expect(lo == mul(s, add(u[n], vv[n])) && hi == mul(s, sub(u[n], vv[n])), "inverse_top", j);
                }
            }
        }
    }
}

// Ways to place the arrays of LazyProduct::multiply.
enum class Layout { kSeparate, kWorkInB, kOutInA };

// LazyProduct against Transform (inputs reduced mod p), for factors of n and m words: kind 0-3
// below 10^9 + 7, kind 4 at the bound 4p.
void test_product(const multimod::Modulus& m, int lg, std::size_t n, std::size_t m_count, int kind, Layout layout) {
    const std::size_t len = std::size_t(1) << lg, half = len / 2;
    Vectors x(len / 8 + 2), y(len / 8 + 2), out(len / 8 + 2), work(len / 8 + 2), ra(len / 8 + 2), rb(len / 8 + 2),
        ref(len / 8 + 2), tables(multimod::LazyProduct::table_words(lg) / 8 + 1);
    // kOutInA: a and b are the halves of x, out is x. Otherwise a is x and b is y; the upper halves
    // hold garbage that must not be read.
    u32* a = x.lanes();
    u32* b = layout == Layout::kOutInA ? x.lanes() + half : y.lanes();
    for (std::size_t i = 0; i < len + 8; ++i) x.lanes()[i] = u32(rng()), y.lanes()[i] = u32(rng());
    const auto value = [&](std::size_t i) -> u32 {
        switch (kind) {
        case 0: return random_below(1000000007);
        case 1: return 1000000006;
        case 2: return i % 3 ? 1000000006 : 0;
        case 3: return random_below(10);
        default: return 4 * m.p - 1 - random_below(3);
        }
    };
    for (std::size_t i = 0; i < half; ++i) a[i] = i < n ? value(i) : 0;
    for (std::size_t i = 0; i < half; ++i) b[i] = i < m_count ? value(i) : 0;
    for (std::size_t i = 0; i < half; ++i) ra.lanes()[i] = a[i] % m.p, rb.lanes()[i] = b[i] % m.p;
    const u32 factor = random_below(m.p);
    const multimod::Transform reference(lg, tables.lanes());
    reference.multiply(multimod::Padded{ra.lanes(), n}, multimod::Padded{rb.lanes(), m_count}, ref.lanes(), work.lanes(),
                       m, factor);
    const multimod::LazyProduct product(lg, tables.lanes());
    u32* o = layout == Layout::kOutInA ? x.lanes() : out.lanes();
    u32* w = layout == Layout::kWorkInB ? b : work.lanes();
    product.multiply(a, b, o, w, m, factor);
    bool ok = true;
    for (std::size_t i = 0; i < len; ++i) ok &= o[i] == ref.lanes()[i];
    expect(ok, "LazyProduct", u64(lg) * 1000000 + n);
}

void run() {
    for (const auto& [p, g] : kPrimes) {
        P = p;
        const multimod::Modulus m(p, g);
        m.select();
        k::k4P = _mm256_set1_epi32(int(4 * p)), k::k8P = _mm256_set1_epi32(int(8 * p));
        Tables t(m);
        test_loops(t);
        test_bottom(t);
        test_top(t);
        for (int lg = 9; lg <= multimod::kMaxLog; ++lg) {
            const std::size_t half = std::size_t(1) << (lg - 1);
            for (const Layout layout : {Layout::kSeparate, Layout::kWorkInB, Layout::kOutInA}) {
                test_product(m, lg, half, half, lg % 4, layout);
                test_product(m, lg, half - 3, 1 + random_below(half), 0, layout);
            }
            test_product(m, lg, half, half, 1, Layout::kOutInA);
            test_product(m, lg, half, half - 5, 4, Layout::kSeparate);
            if (lg <= 12) test_product(m, lg, 1, 1, 2, Layout::kSeparate);
        }
    }
}

}  // namespace lazy

// WideProduct against Transform on the same factors of n and m 64-bit words. kind 0: random,
// 1: 2^64 - 1, 2: 2^64 - 1 and zeros. b holds exactly half the length, so a sanitizer sees any read
// past it; a holds garbage from there up (work_in_a: work is a's storage).
void test_wide(const multimod::Modulus& mod, int lg, std::size_t n, std::size_t m, int kind, bool work_in_a) {
    const std::size_t len = std::size_t(1) << lg, half = len / 2, words = len + multimod::WideProduct::kPadding;
    Buffer<u64> a(words / 2), b(half), ra(half), rb(half);
    Buffer<u32> tables(multimod::WideProduct::table_words(lg)), out(words), work(words), ref(words);
    const auto value = [&](std::size_t i) -> u64 {
        switch (kind) {
        case 0: return rng();
        case 1: return ~u64(0);
        default: return i % 3 ? ~u64(0) : 0;
        }
    };
    for (std::size_t i = 0; i < half; ++i) ra.data[i] = a.data[i] = i < n ? value(i) : 0;
    for (std::size_t i = 0; i < half; ++i) rb.data[i] = b.data[i] = i < m ? value(i) : 0;
    for (std::size_t i = half; i < words / 2; ++i) a.data[i] = rng();
    const u32 factor = u32(rng() % mod.p);
    const multimod::Transform reference(lg, tables.data);
    reference.multiply(multimod::Wide{ra.data, n}, multimod::Wide{rb.data, m}, ref.data, work.data, mod, factor);
    const multimod::WideProduct product(lg, tables.data);
    u32* w = work_in_a ? reinterpret_cast<u32*>(a.data) : work.data;
    product.multiply(multimod::Wide{a.data, n}, multimod::Wide{b.data, m}, out.data, w, mod, factor);
    expect(std::equal(out.data, out.data + len, ref.data), "WideProduct", lg, n, m);
}

void test_wide_products() {
    for (int lg = 9; lg <= multimod::kMaxLog; ++lg) {
        const std::size_t half = std::size_t(1) << (lg - 1);
        for (int kind = 0; kind < 3; ++kind) {
            // Primes above 2^29: the first six.
            const auto& [p, g] = kPrimes[rng() % 6];
            const multimod::Modulus mod(p, g);
            test_wide(mod, lg, half, half, kind, false);
            test_wide(mod, lg, half - 3, 1 + rng() % half, kind, true);
            if (lg <= 12) test_wide(mod, lg, 1, 1, kind, false);
        }
    }
}

}  // namespace

int main() {
    test_transforms();
    lazy::run();
    test_wide_products();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
