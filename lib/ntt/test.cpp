// Tests for lib/ntt: each kernel against a scalar model (residues and output ranges), products
// against schoolbook multiplication and, for long ones, evaluation at random points.
#include "lib/ntt/ntt.hpp"
#include "lib/ntt/product.hpp"

#include <cstdio>
#include <random>
#include <vector>

namespace {

using ntt::kernels::Vec;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u8 = unsigned char;

constexpr u32 P = ntt::kModulus;

std::mt19937_64 rng(2026);
int failures = 0;

void expect(bool ok, const char* what, u64 detail = 0) {
    if (ok) return;
    if (++failures <= 20) std::printf("FAIL: %s (%llu)\n", what, static_cast<unsigned long long>(detail));
}

u32 random_below(u64 bound) { return u32(rng() % bound); }

u32 mul(u32 x, u32 y) { return u32(u64(x) * y % P); }
u32 add(u32 x, u32 y) { return u32((u64(x) + y) % P); }
u32 sub(u32 x, u32 y) { return u32((u64(x) % P + P - y % P) % P); }
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

void fill(u32* p, std::size_t count, u64 bound) {
    for (std::size_t i = 0; i < count; ++i) p[i] = random_below(bound);
    // Boundary values in the first lanes.
    for (std::size_t i = 0; i < count && i < 4; ++i) p[i] = u32(bound - 1 - i);
}

// Twiddle tables for transforms of length 2^lg, 32-byte aligned.
struct Tables {
    explicit Tables(int lg) : forward_(ntt::detail::table_words(lg) / 8), inverse_(forward_.size()) {
        ntt::detail::build_table(forward(), (std::size_t(1) << lg) / 16, ntt::detail::kRoots[0]);
        ntt::detail::build_table(inverse(), (std::size_t(1) << lg) / 16, ntt::detail::kRoots[1]);
    }
    std::size_t words() const { return 8 * forward_.size(); }
    const u32* forward() const { return forward_.data()->lane; }
    const u32* inverse() const { return inverse_.data()->lane; }
    const u32* r(std::size_t k) const { return forward() + ntt::detail::slot(k); }  // value; quotient at [8]
    const u32* ir(std::size_t k) const { return inverse() + ntt::detail::slot(k); }

private:
    u32* forward() { return forward_.data()->lane; }
    u32* inverse() { return inverse_.data()->lane; }
    std::vector<Vectors::Lanes> forward_, inverse_;
};

constexpr int kTableLog = 12;
constexpr std::size_t kGroups = (std::size_t(1) << kTableLog) / 32;  // groups k with r[2k + 1] in the table

void test_tables() {
    const Tables t(kTableLog);
    const u32 i4 = power(3, (P - 1) / 4);
    expect(*t.r(0) == 1 && *t.r(1) == i4, "table: r[0] = 1, r[1] = sqrt(-1)");
    for (std::size_t k = 1; k < kGroups; ++k) {
        expect(mul(*t.r(2 * k), *t.r(2 * k)) == *t.r(k), "table: r[2k]^2 = r[k]", k);
        expect(*t.r(2 * k + 1) == mul(*t.r(2 * k), i4), "table: r[2k + 1] = r[2k] sqrt(-1)", k);
    }
    for (std::size_t k = 0; k < 2 * kGroups; ++k) {
        expect(mul(*t.r(k), *t.ir(k)) == 1, "table: inverse", k);
        for (const u32* e : {t.r(k), t.ir(k)}) expect(e[0] < P && e[8] == u32((u64(e[0]) << 32) / P), "table: quotient", k);
    }
}

// Forward butterfly of one lane: a + x c + y (b + x d), ..., mod P.
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
void test_loop(const char* name, std::size_t h, u64 in_bound, u64 out_bound, Kernel kernel, Model model) {
    Vectors v(4 * h);
    fill(v.lanes(), 32 * h, in_bound);
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

void test_loops() {
    const Tables t(kTableLog);
    for (std::size_t h : {2, 4, 16}) {
        for (int round = 0; round < 8; ++round) {
            const std::size_t k = 1 + random_below(kGroups - 1);
            const u32 *x = t.r(k), *y = t.r(2 * k), *ix = t.ir(k), *iy = t.ir(2 * k);
            test_loop("forward", h, 4 * u64(P), 4 * u64(P),
                      [&](Vec* f, std::size_t n) { ntt::kernels::forward(f, n, x, y); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, *x, y[0], y[1]); });
            test_loop("inverse", h, 2 * u64(P), 2 * u64(P),
                      [&](Vec* f, std::size_t n) { ntt::kernels::inverse(f, n, ix, iy); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, *ix, iy[0], iy[1]); });
            test_loop("forward_identity", h, 4 * u64(P), 4 * u64(P),
                      [&](Vec* f, std::size_t n) { ntt::kernels::forward_identity(f, n, t.r(0)); },
                      [&](const std::array<u32, 4>& f) { return forward_model(f, 1, 1, *t.r(1)); });
            test_loop("inverse_identity", h, 2 * u64(P), 2 * u64(P),
                      [&](Vec* f, std::size_t n) { ntt::kernels::inverse_identity(f, n, t.ir(0)); },
                      [&](const std::array<u32, 4>& f) { return inverse_model(f, 1, 1, *t.ir(1)); });
            // forward_pair: forward() on two arrays.
            Vectors a(4 * h), b(4 * h);
            fill(a.lanes(), 32 * h, 4 * u64(P));
            fill(b.lanes(), 32 * h, 4 * u64(P));
            Vectors a1 = a, b1 = b;
            ntt::kernels::forward_pair(a.vec(), b.vec(), h, x, y);
            ntt::kernels::forward(a1.vec(), h, x, y);
            ntt::kernels::forward(b1.vec(), h, x, y);
            for (std::size_t i = 0; i < 32 * h; ++i)
                expect(a.lanes()[i] == a1.lanes()[i] && b.lanes()[i] == b1.lanes()[i], "forward_pair", i);
        }
        // scale_radix2: s (f0 + f1), s (f0 - f1), canonical.
        Vectors v(2 * h);
        fill(v.lanes(), 16 * h, 2 * u64(P));
        const std::vector<u32> in(v.lanes(), v.lanes() + 16 * h);
        alignas(32) u32 s[16] = {};
        s[1] = random_below(P);
        s[9] = u32((u64(s[1]) << 32) / P);
        ntt::kernels::scale_radix2(v.vec(), h, s);
        for (std::size_t j = 0; j < 8 * h; ++j) {
            expect(v.lanes()[j] == mul(s[1], add(in[j], in[j + 8 * h])), "scale_radix2 sum", j);
            expect(v.lanes()[j + 8 * h] == mul(s[1], sub(in[j], in[j + 8 * h])), "scale_radix2 difference", j);
        }
    }
}

// Bottom stage on batches of four vectors: forward butterfly with h = 1, products of the leaves
// mod x^8 - w (with the Montgomery factor 2^-32), inverse butterfly.
void test_bottom() {
    const Tables t(kTableLog);
    const u32 r_inverse = power(u32((u64(1) << 32) % P), P - 2);
    for (int round = 0; round < 50; ++round) {
        const std::size_t batches = 1 + random_below(4), k0 = random_below(kGroups - batches);
        Vectors a(4 * batches), b(4 * batches);
        fill(a.lanes(), 32 * batches, 4 * u64(P));
        fill(b.lanes(), 32 * batches, 4 * u64(P));
        const std::vector<u32> a0(a.lanes(), a.lanes() + 32 * batches), b0(b.lanes(), b.lanes() + 32 * batches);
        const auto weights = [&](std::size_t k) {
            const u32* y = t.r(2 * k);
            return std::array<u32, 4>{y[0], P - y[0], y[1], P - y[1]};
        };
        // As ntt::detail::Recursion::bottom().
        struct alignas(64) Leaves {
            u32 words[96];
        } leaves[2];
        alignas(32) u32 w[2][8];
        const auto load_weights = [&](std::size_t k, u32* out) {
            const auto v = weights(k);
            for (int i = 0; i < 4; ++i) out[i] = v[i], out[4 + i] = u32((u64(v[i]) << 32) / P);
        };
        Vec *av = a.vec(), *bv = b.vec();
        load_weights(k0, w[0]);
        ntt::kernels::bottom_first(av, bv, &leaves[0], t.r(k0), t.r(2 * k0), w[0]);
        for (std::size_t j = 0; j < batches; ++j) {
            const std::size_t k = k0 + j, cur = j % 2;
            if (j + 1 == batches) {
                ntt::kernels::bottom_last(av + 4 * j, &leaves[cur], t.ir(k), t.ir(2 * k));
                break;
            }
            load_weights(k + 1, w[cur ^ 1]);
            ntt::kernels::bottom_both(av + 4 * j + 4, bv + 4 * j + 4, &leaves[cur ^ 1], t.r(k + 1), t.r(2 * k + 2),
                                      w[cur ^ 1], av + 4 * j, &leaves[cur], t.ir(k), t.ir(2 * k));
        }
        for (std::size_t j = 0; j < batches; ++j) {
            const std::size_t k = k0 + j;
            const u32 x = *t.r(k), y = t.r(2 * k)[0], z = t.r(2 * k)[1];
            u32 fa[4][8], fb[4][8], c[4][8];
            for (int l = 0; l < 8; ++l) {
                std::array<u32, 4> xa, xb;
                for (int s = 0; s < 4; ++s) xa[s] = a0[32 * j + 8 * s + l], xb[s] = b0[32 * j + 8 * s + l];
                const auto ya = forward_model(xa, x, y, z), yb = forward_model(xb, x, y, z);
                for (int s = 0; s < 4; ++s) fa[s][l] = ya[s], fb[s][l] = yb[s];
            }
            const auto wt = weights(k);
            for (int s = 0; s < 4; ++s)
                for (int l = 0; l < 8; ++l) {
                    u32 sum = 0;
                    for (int i = 0; i < 8; ++i)
                        sum = add(sum, mul(fb[s][i], l >= i ? fa[s][l - i] : mul(wt[s], fa[s][l - i + 8])));
                    c[s][l] = mul(sum, r_inverse);
                }
            for (int l = 0; l < 8; ++l) {
                const auto expected = inverse_model({c[0][l], c[1][l], c[2][l], c[3][l]}, *t.ir(k), t.ir(2 * k)[0],
                                                    t.ir(2 * k)[1]);
                for (int s = 0; s < 4; ++s) {
                    const u32 got = a.lanes()[32 * j + 8 * s + l];
                    expect(got < 2 * P && got % P == expected[s], "bottom", 32 * j + 8 * s + l);
                }
            }
        }
    }
}

template <class Multiplier>
std::vector<u32> product(const std::vector<u32>& a, const std::vector<u32>& b) {
    Multiplier conv(a.size(), b.size());
    std::copy(a.begin(), a.end(), conv.a());
    std::copy(b.begin(), b.end(), conv.b());
    const u32* c = conv.multiply();
    return {c, c + a.size() + b.size() - 1};
}

std::vector<u32> schoolbook(const std::vector<u32>& a, const std::vector<u32>& b) {
    std::vector<u64> c(a.size() + b.size() - 1);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + u64(a[i]) * b[j]) % P;
    return {c.begin(), c.end()};
}

u32 evaluate(const std::vector<u32>& p, u32 x) {
    u32 r = 0;
    for (std::size_t i = p.size(); i--;) r = add(mul(r, x), p[i]);
    return r;
}

std::vector<u32> random_poly(std::size_t n, int kind) {
    std::vector<u32> p(n);
    for (auto& x : p) x = kind == 0 ? random_below(P) : kind == 1 ? P - 1 : kind == 2 ? 0 : random_below(10);
    return p;
}

template <class Multiplier = ntt::Convolution>
void test_product(std::size_t n, std::size_t m, int kind, bool exact) {
    const auto a = random_poly(n, kind), b = random_poly(m, kind);
    const auto c = product<Multiplier>(a, b);
    bool ok = c.size() == n + m - 1;
    for (const u32 x : c) ok &= x < P;
    if (exact) {
        ok &= c == schoolbook(a, b);
    } else {
        for (int i = 0; i < 3; ++i) {
            const u32 x = random_below(P);
            ok &= evaluate(c, x) == mul(evaluate(a, x), evaluate(b, x));
        }
    }
    expect(ok, "product", n * 100000000 + m);
}

void test_products() {
    for (std::size_t n = 1; n <= 40; ++n)
        for (std::size_t m = 1; m <= 40; m += 3) test_product(n, m, 0, true);
    for (int round = 0; round < 60; ++round) test_product(1 + random_below(1500), 1 + random_below(1500), round % 4, true);
    // Every transform length, sparse and dense first levels.
    for (int lg = 6; lg <= ntt::kMaxLog; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const bool exact = lg <= 11;
        test_product(len / 2, len / 2, lg % 2, exact);
        test_product(len / 2 + 1, len / 2 - 1, 0, exact);
        test_product(len - 1, 2, 0, exact);
        if (lg <= 22) test_product(len / 4 + 1, 1, 3, exact);
    }
}

// ntt::Product with garbage in the factors' upper halves, which it must not read.
void test_upper_halves_unread(std::size_t n, std::size_t m) {
    const auto a = random_poly(n, 0), b = random_poly(m, 0);
    ntt::Product p(n, m);
    std::copy(a.begin(), a.end(), p.a());
    std::copy(b.begin(), b.end(), p.b());
    const std::size_t half = p.length() / 2;
    for (std::size_t i = half; i < 2 * half; ++i) p.a()[i] = u32(rng()), p.b()[i] = u32(rng());
    const u32* c = p.multiply();
    expect(std::vector<u32>(c, c + n + m - 1) == schoolbook(a, b), "upper halves unread", n * 100000000 + m);
}

// ntt::Product: the bounds of fits(), the extra bytes, and every length it takes (both top levels:
// radix 4 at odd lg, radix 8 and the fused inverse top at even lg).
void test_product_class() {
    using ntt::Product;
    expect(Product::fits(256, 256) && !Product::fits(257, 255) && !Product::fits(128, 128), "fits: lg 9");
    expect(Product::fits(2, 256) && !Product::fits(2, 257) && !Product::fits(1, 300) && !Product::fits(0, 300),
           "fits: small factor");
    const std::size_t half = std::size_t(1) << (ntt::kMaxLog - 1);
    expect(Product::fits(half, half) && !Product::fits(half + 1, half - 1), "fits: lg 25");
    {
        constexpr std::size_t kExtra = 100000;
        Product p(300, 300, kExtra);
        auto* extra = static_cast<unsigned char*>(p.extra());
        bool zero = reinterpret_cast<std::uintptr_t>(extra) % 64 == 0;
        for (std::size_t i = 0; i < kExtra; ++i) zero &= extra[i] == 0, extra[i] = u8(i * 7);
        std::fill(p.a(), p.a() + 300, 1), std::fill(p.b(), p.b() + 300, 1);
        const u32* c = p.multiply();
        bool kept = c[0] == 1 && c[299] == 300 && c[598] == 1;
        for (std::size_t i = 0; i < kExtra; ++i) kept &= extra[i] == u8(i * 7);
        expect(zero && kept, "extra bytes: aligned, zeroed, kept");
    }
    for (int lg = 9; lg <= 11; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        test_upper_halves_unread(len / 2, len / 2);
        test_upper_halves_unread(len / 2 - 3, len / 4 + 9);
    }
    for (int lg = 9; lg <= ntt::kMaxLog; ++lg) {
        const std::size_t len = std::size_t(1) << lg;
        const bool exact = lg <= 11;
        test_product<Product>(len / 2, len / 2, lg % 2, exact);
        test_product<Product>(len / 2, 2, 0, exact);
        if (lg <= 22) {
            test_product<Product>(len / 4 + 1, len / 4 + 1, 3, exact);
            test_product<Product>(len / 2 - 5, len / 4 + 7, 0, exact);
        }
    }
}

}  // namespace

int main() {
    test_tables();
    test_loops();
    test_bottom();
    test_products();
    test_product_class();
    if (failures) {
        std::printf("%d failures\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
