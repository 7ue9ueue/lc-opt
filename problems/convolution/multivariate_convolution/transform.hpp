// Cyclic NTT mod 998244353 of length 2^lg, 7 <= lg <= 22, with forward transform, pointwise
// products and inverse transform as separate steps (lib/ntt fuses them at its leaves).
//
//   split::Transform t(lg);
//   t.run(base, stride, inputs, outputs, count, load, leaf, store);   // see run()
//
// Inputs are zero above half the length, so the transform holds products of two inputs without
// wrap-around. Points come in a fixed order shared by every transform of the same length.
//
// Levels: node k of size 2d is a polynomial mod x^2d - r[k]^2, split by r[k] into nodes 2k and
// 2k + 1 (r: lib/ntt's twiddle table). The top level is radix 2 or 4 and fused with load and
// store; then lib/ntt's radix-4 kernels, depth first, down to nodes of 8 vectors; a node of 8
// vectors takes three levels across vectors, an 8 x 8 transpose (lane i = leaf 8k + i) and three
// levels with per-lane twiddles, down to single points.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/ntt/ntt.hpp"

namespace split {

using Vec = __m256i;

namespace detail {

using ntt::detail::slot;
using std::uint32_t;
constexpr uint32_t kP = ntt::kernels::kP;

inline Vec all(uint32_t x) { return _mm256_set1_epi32(int(x)); }

// A twiddle per lane: values w < P and Shoup quotients floor(w 2^32 / P), the odd lanes' quotients
// shifted down.
struct Twiddle {
    Vec w, q, q_odd;
};

// One twiddle for all lanes, from a table entry (value, quotient 8 words on).
inline Twiddle broadcast(const uint32_t* entry) {
    const Vec q = all(entry[8]);
    return {all(entry[0]), q, q};
}

// x w mod P in [0, 2P), for any x < 2^32.
inline Vec multiply(Vec x, const Twiddle& t) {
    const Vec even = _mm256_srli_epi64(_mm256_mul_epu32(x, t.q), 32);
    const Vec odd = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), t.q_odd);
    const Vec q = _mm256_blend_epi32(even, odd, 0xAA);
    return _mm256_sub_epi32(_mm256_mullo_epi32(x, t.w), _mm256_mullo_epi32(q, all(kP)));
}

// x < 4P -> x mod 2P.
inline Vec reduce2(Vec x) { return _mm256_min_epu32(x, _mm256_sub_epi32(x, all(2 * kP))); }

// lo, hi < 4P -> lo + w hi, lo - w hi, both < 4P.
inline void forward_butterfly(Vec& lo, Vec& hi, const Twiddle& t) {
    const Vec x = reduce2(lo), y = multiply(hi, t);
    lo = _mm256_add_epi32(x, y);
    hi = _mm256_sub_epi32(_mm256_add_epi32(x, all(2 * kP)), y);
}

// x, y < 2P -> x + y, (x - y) w, both < 2P.
inline void inverse_butterfly(Vec& x, Vec& y, const Twiddle& t) {
    const Vec s = reduce2(_mm256_add_epi32(x, y));
    y = multiply(_mm256_sub_epi32(_mm256_add_epi32(x, all(2 * kP)), y), t);
    x = s;
}

// v[i] lane c <-> v[c] lane i.
inline void transpose(Vec* v) {
    Vec t[8], u[8];
#pragma GCC unroll 8
    for (int i = 0; i < 8; i += 2) {
        t[i] = _mm256_unpacklo_epi32(v[i], v[i + 1]);
        t[i + 1] = _mm256_unpackhi_epi32(v[i], v[i + 1]);
    }
#pragma GCC unroll 8
    for (int i = 0; i < 8; i += 4)
#pragma GCC unroll 8
        for (int j = 0; j < 2; ++j) {
            u[i + j] = _mm256_unpacklo_epi64(t[i + j], t[i + j + 2]);
            u[i + j + 2] = _mm256_unpackhi_epi64(t[i + j], t[i + j + 2]);
        }
    // u[0..3]: lanes (c, c + 4) of rows 0-3 for c = 0, 2, 1, 3 order; u[4..7] likewise for rows 4-7.
    const int order[4] = {0, 2, 1, 3};
#pragma GCC unroll 8
    for (int j = 0; j < 4; ++j) {
        v[order[j]] = _mm256_permute2x128_si256(u[j], u[j + 4], 0x20);
        v[order[j] + 4] = _mm256_permute2x128_si256(u[j], u[j + 4], 0x31);
    }
}

}  // namespace detail

class Transform {
public:
    explicit Transform(int lg) : lg_(lg), nv_(std::size_t(1) << (lg - 3)) {
        using namespace detail;
        const std::size_t entries = 4 * nv_;  // r[j] for j < 4 nv: the deepest level splits node 4 nv - 1
        for (int d = 0; d < 2; ++d) {
            std::vector<uint32_t>& t = table_[d];
            t.assign(slot(entries) + 32, 0);
            ntt::detail::build_table(aligned(t), entries, ntt::detail::kRoots[d]);
        }
        // Per-lane twiddles of the last two levels of the bottom node k: r[16k + 2i + p] (p < 2),
        // then r[32k + 4i + p] (p < 4), for lanes i < 8; each as 8 values and 8 quotients.
        for (int d = 0; d < 2; ++d) {
            std::vector<uint32_t>& lanes = lanes_[d];
            const uint32_t* t = aligned(table_[d]);
            lanes.assign(nv_ / 8 * kLaneWords + 32, 0);
            uint32_t* out = aligned(lanes);
            for (std::size_t k = 0; k < nv_ / 8; ++k, out += kLaneWords) {
                for (int p = 0; p < 2; ++p)
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t* e = t + slot(16 * k + 2 * i + p);
                        out[16 * p + i] = e[0], out[16 * p + 8 + i] = e[8];
                    }
                for (int p = 0; p < 4; ++p)
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t* e = t + slot(32 * k + 4 * i + p);
                        out[32 + 16 * p + i] = e[0], out[32 + 16 * p + 8 + i] = e[8];
                    }
            }
        }
    }

    std::size_t vectors() const { return nv_; }

    // Arrays i < inputs at base + i * stride (stride >= vectors() + 1, 32-byte aligned). Input i is
    // load(i, j) for vectors j < count <= vectors() / 2 (canonical), zero above. The transforms run
    // depth first, all arrays together: at each node of 8 vectors, after the inputs' forward
    // transforms, leaf(offset) replaces the points of arrays i < outputs at vectors
    // [offset, offset + 8) by values < 2P (typically products of input points; inputs' points are
    // canonical). Then the inverse transforms of outputs i: store(i, j, x) for j < count, x < 4P,
    // x = 2^lg times coefficient vector j.
    template <class Load, class Leaf, class Store>
    void run(Vec* base, std::size_t stride, std::size_t inputs, std::size_t outputs, std::size_t count, Load load,
             Leaf leaf, Store store) const {
        using namespace detail;
        const Vec zero = _mm256_setzero_si256(), p2 = all(2 * kP);
        const Batch batch{base, stride, inputs, outputs};
        if ((lg_ - 6) % 2) {  // radix 2: both nodes get the lower half
            const std::size_t h = nv_ / 2;
            for (std::size_t i = 0; i < inputs; ++i) {
                Vec* a = base + i * stride;
                for (std::size_t j = 0; j < h; ++j) a[j] = a[j + h] = j < count ? load(i, j) : zero;
            }
            node(batch, 0, h, 0, leaf);
            node(batch, h, h, 1, leaf);
            for (std::size_t i = 0; i < outputs; ++i) {
                const Vec* a = base + i * stride;
                for (std::size_t j = 0; j < count; ++j) store(i, j, _mm256_add_epi32(a[j], a[j + h]));
            }
            return;
        }
        const std::size_t h = nv_ / 4;
        {
            const Twiddle z = broadcast(roots() + slot(1));
            for (std::size_t i = 0; i < inputs; ++i) {
                Vec* a = base + i * stride;
                for (std::size_t j = 0; j < h; ++j) {
                    const Vec x = j < count ? load(i, j) : zero, y = j + h < count ? load(i, j + h) : zero;
                    const Vec zy = multiply(y, z);
                    a[j] = _mm256_add_epi32(x, y), a[j + h] = _mm256_sub_epi32(_mm256_add_epi32(x, p2), y);
                    a[j + 2 * h] = _mm256_add_epi32(x, zy), a[j + 3 * h] = _mm256_sub_epi32(_mm256_add_epi32(x, p2), zy);
                }
            }
        }
        for (std::size_t t = 0; t < 4; ++t) node(batch, t * h, h, t, leaf);
        const Twiddle z = broadcast(inverse_roots() + slot(1));
        for (std::size_t i = 0; i < outputs; ++i) {
            const Vec* a = base + i * stride;
            for (std::size_t j = 0; j < h && j < count; ++j) {
                const Vec p0 = a[j], p1 = a[j + h], q0 = a[j + 2 * h], q1 = a[j + 3 * h];
                const Vec ab = reduce2(_mm256_add_epi32(p0, p1)), cd = reduce2(_mm256_add_epi32(q0, q1));
                store(i, j, _mm256_add_epi32(ab, cd));
                if (j + h < count) {
                    const Vec amb = reduce2(_mm256_sub_epi32(_mm256_add_epi32(p0, p2), p1));
                    const Vec cmd = multiply(_mm256_sub_epi32(_mm256_add_epi32(q0, p2), q1), z);
                    store(i, j + h, _mm256_add_epi32(amb, cmd));
                }
            }
        }
    }

private:
    static constexpr std::size_t kLaneWords = 6 * 16;  // per bottom node: 2 + 4 twiddle vectors with quotients

    struct Batch {
        Vec* base;
        std::size_t stride, inputs, outputs;
    };

    static std::uint32_t* aligned(std::vector<std::uint32_t>& v) {
        return reinterpret_cast<std::uint32_t*>((reinterpret_cast<std::uintptr_t>(v.data()) + 31) & ~std::uintptr_t(31));
    }
    static const std::uint32_t* aligned(const std::vector<std::uint32_t>& v) {
        return aligned(const_cast<std::vector<std::uint32_t>&>(v));
    }
    const std::uint32_t* roots() const { return aligned(table_[0]); }
    const std::uint32_t* inverse_roots() const { return aligned(table_[1]); }

    // Node k: vectors [offset, offset + nvs) of every array (nvs / 8 a power of 4).
    template <class Leaf>
    void node(const Batch& b, std::size_t offset, std::size_t nvs, std::size_t k, Leaf& leaf) const {
        using ntt::detail::slot;
        if (nvs == 8) {
            for (std::size_t i = 0; i < b.inputs; ++i) forward_bottom(b.base + i * b.stride + offset, k);
            leaf(offset);
            for (std::size_t i = 0; i < b.outputs; ++i) inverse_bottom(b.base + i * b.stride + offset, k);
            return;
        }
        const std::size_t h = nvs / 4;
        for (std::size_t i = 0; i < b.inputs; ++i) {
            Vec* a = b.base + i * b.stride + offset;
            if (k == 0)
                ntt::kernels::forward_identity(a, h, roots());
            else
                ntt::kernels::forward(a, h, roots() + slot(k), roots() + slot(2 * k));
        }
        for (std::size_t t = 0; t < 4; ++t) node(b, offset + t * h, h, 4 * k + t, leaf);
        for (std::size_t i = 0; i < b.outputs; ++i) {
            Vec* a = b.base + i * b.stride + offset;
            if (k == 0)
                ntt::kernels::inverse_identity(a, h, inverse_roots());
            else
                ntt::kernels::inverse(a, h, inverse_roots() + slot(k), inverse_roots() + slot(2 * k));
        }
    }

    // Per-lane twiddle p of a lanes_ block: values at 16 p, quotients 8 words on.
    static detail::Twiddle lane_twiddle(const std::uint32_t* block, int p) {
        const std::uint32_t* e = block + 16 * p;
        return {_mm256_load_si256(reinterpret_cast<const Vec*>(e)), _mm256_load_si256(reinterpret_cast<const Vec*>(e + 8)),
                _mm256_loadu_si256(reinterpret_cast<const Vec*>(e + 9))};
    }

    // Node k of 8 vectors (values < 4P) to its 64 points, canonical.
    void forward_bottom(Vec* a, std::size_t k) const {
        using namespace detail;
        const std::uint32_t* r = roots();
        Vec v[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) v[i] = a[i];
        {
            const Twiddle t = broadcast(r + slot(k));
#pragma GCC unroll 8
            for (int i = 0; i < 4; ++i) forward_butterfly(v[i], v[i + 4], t);
        }
#pragma GCC unroll 8
        for (int p = 0; p < 2; ++p) {
            const Twiddle t = broadcast(r + slot(2 * k + p));
#pragma GCC unroll 8
            for (int i = 0; i < 2; ++i) forward_butterfly(v[4 * p + i], v[4 * p + i + 2], t);
        }
#pragma GCC unroll 8
        for (int p = 0; p < 4; ++p) forward_butterfly(v[2 * p], v[2 * p + 1], broadcast(r + slot(4 * k + p)));
        transpose(v);
        {
            const std::uint32_t* e = r + slot(8 * k);
            const Twiddle t{_mm256_load_si256(reinterpret_cast<const Vec*>(e)),
                            _mm256_load_si256(reinterpret_cast<const Vec*>(e + 8)),
                            _mm256_loadu_si256(reinterpret_cast<const Vec*>(e + 9))};
#pragma GCC unroll 8
            for (int c = 0; c < 4; ++c) forward_butterfly(v[c], v[c + 4], t);
        }
        const std::uint32_t* block = aligned(lanes_[0]) + k * kLaneWords;
#pragma GCC unroll 8
        for (int p = 0; p < 2; ++p) {
            const Twiddle t = lane_twiddle(block, p);
#pragma GCC unroll 8
            for (int c = 0; c < 2; ++c) forward_butterfly(v[4 * p + c], v[4 * p + c + 2], t);
        }
#pragma GCC unroll 8
        for (int p = 0; p < 4; ++p) forward_butterfly(v[2 * p], v[2 * p + 1], lane_twiddle(block, 2 + p));
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) {
            const Vec x = reduce2(v[i]);
            a[i] = _mm256_min_epu32(x, _mm256_sub_epi32(x, all(kP)));
        }
    }

    // Points (< 2P) of node k back to its 8 vectors times 64, < 2P.
    void inverse_bottom(Vec* a, std::size_t k) const {
        using namespace detail;
        const std::uint32_t* r = inverse_roots();
        Vec v[8];
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) v[i] = a[i];
        const std::uint32_t* block = aligned(lanes_[1]) + k * kLaneWords;
#pragma GCC unroll 8
        for (int p = 0; p < 4; ++p) inverse_butterfly(v[2 * p], v[2 * p + 1], lane_twiddle(block, 2 + p));
#pragma GCC unroll 8
        for (int p = 0; p < 2; ++p) {
            const Twiddle t = lane_twiddle(block, p);
#pragma GCC unroll 8
            for (int c = 0; c < 2; ++c) inverse_butterfly(v[4 * p + c], v[4 * p + c + 2], t);
        }
        {
            const std::uint32_t* e = r + slot(8 * k);
            const Twiddle t{_mm256_load_si256(reinterpret_cast<const Vec*>(e)),
                            _mm256_load_si256(reinterpret_cast<const Vec*>(e + 8)),
                            _mm256_loadu_si256(reinterpret_cast<const Vec*>(e + 9))};
#pragma GCC unroll 8
            for (int c = 0; c < 4; ++c) inverse_butterfly(v[c], v[c + 4], t);
        }
        transpose(v);
#pragma GCC unroll 8
        for (int p = 0; p < 4; ++p) inverse_butterfly(v[2 * p], v[2 * p + 1], broadcast(r + slot(4 * k + p)));
#pragma GCC unroll 8
        for (int p = 0; p < 2; ++p) {
            const Twiddle t = broadcast(r + slot(2 * k + p));
#pragma GCC unroll 8
            for (int i = 0; i < 2; ++i) inverse_butterfly(v[4 * p + i], v[4 * p + i + 2], t);
        }
        {
            const Twiddle t = broadcast(r + slot(k));
#pragma GCC unroll 8
            for (int i = 0; i < 4; ++i) inverse_butterfly(v[i], v[i + 4], t);
        }
#pragma GCC unroll 8
        for (int i = 0; i < 8; ++i) a[i] = v[i];
    }

    int lg_;
    std::size_t nv_;
    std::vector<std::uint32_t> table_[2];  // forward, inverse: r[j] for j < 4 nv, lib/ntt's block layout
    std::vector<std::uint32_t> lanes_[2];  // forward, inverse: per-lane twiddles of each bottom node
};

}  // namespace split
