#!/usr/bin/env python3
"""Generate kernels.hpp: the transform's two lowest radix-4 levels and the column levels of pruned
transforms as GCC inline assembly loops.

Uses lib/ntt/gen_kernels.py's dataflow graphs, Zen 3 list scheduler and register allocator.
  forward_h4, inverse_h4: the level h = 4 over consecutive groups, one group (4 butterflies) per
    iteration; twiddle addresses from the group index, slot(k) = k + (k & ~7).
  forward_bottom, inverse_bottom: the level h = 1 (the leaves), 8 groups per iteration; their
    twiddles are one table block of x and two of y, z. Forward outputs are canonical.
  forward_columns, inverse_columns (and *_even_columns for width 1): lib/ntt's radix-4 loops of one
    group on the columns j < h with (j & width) == 0, for pruned transforms: the same scheduled
    body, the stride h separate from the count.
  forward_top8_lower, forward_top8_upper: the forward radix-8 top level of transforms of 2 * 4^j
    vectors (the radix-2 level and each half's first radix-4 group) for a source in one half, in
    place, two columns per iteration.
Arithmetic and ranges as lib/ntt's kernels (lib/ntt/notes.md); knobs in lib/poly/notes.md.

Usage: gen_kernels.py [--check]   (--check: exit 1 if kernels.hpp differs from the output)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('ntt_gen_kernels', HERE.parent / 'ntt' / 'gen_kernels.py')
ntt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ntt)

OUT = HERE / 'kernels.hpp'
Value, Graph, Knobs = ntt.Value, ntt.Graph, ntt.Knobs

# Chosen by timing generated families on lc-amd (EPYC 7B13), 256-vector tiles in L1.
H4_KNOBS = Knobs(window=16)
BOTTOM_KNOBS = Knobs(window=40, margin=1)


def twiddles(g, x, y, i=0):
    """Group i of a block: x entry i at x; y, z entries 2i, 2i + 1 of the blocks at y."""
    yd = 64 * (i // 4) + 8 * (i % 4)
    return {'WX': g.scalar(x, 4 * i), 'QX': g.scalar(x, 32 + 4 * i), 'WY': g.scalar(y, yd), 'QY': g.scalar(y, yd + 32),
            'WZ': g.scalar(y, yd + 4), 'QZ': g.scalar(y, yd + 36)}


def forward(g, tw, quarters, canonical):
    """Forward butterfly on the vectors at byte offsets quarters of a, in place, values < 4P."""
    f = [g.at('a', q) for q in quarters]
    a, b = g.low(f[0]), g.low(f[1])
    cc = g.shoup(f[2], tw['WX'], tw['QX'], odd=g.at('a', quarters[2] + 4))
    dd = g.shoup(f[3], tw['WX'], tw['QX'], odd=g.at('a', quarters[3] + 4))
    ac = g.low(g.add(a, cc))
    amc = g.low_signed(g.op('vpsubd', a, cc))
    bd, bmd = g.add(b, dd), g.diff(b, dd)
    y = g.shoup(bd, tw['WY'], tw['QY'])
    z = g.shoup(bmd, tw['WZ'], tw['QZ'])
    for q, v in zip(quarters, [g.add(ac, y), g.diff(ac, y), g.add(amc, z), g.diff(amc, z)]):
        g.store(g.canonical(v) if canonical else v, 'a', None, q)


def inverse(g, tw, quarters, source):
    """Inverse butterfly from the vectors at quarters of source into those of a, values < 2P."""
    f = [g.at(source, q) for q in quarters]
    ab, cd = g.low(g.add(f[0], f[1])), g.low(g.add(f[2], f[3]))
    amb = g.shoup(g.diff(f[0], f[1]), tw['WY'], tw['QY'])
    cmd = g.shoup(g.diff(f[2], f[3]), tw['WZ'], tw['QZ'])
    g.store(g.low(g.add(ab, cd)), 'a', None, quarters[0])
    g.store(g.low(g.add(amb, cmd)), 'a', None, quarters[1])
    g.store(g.shoup(g.diff(ab, cd), tw['WX'], tw['QX']), 'a', None, quarters[2])
    g.store(g.shoup(g.diff(amb, cmd), tw['WX'], tw['QX']), 'a', None, quarters[3])


def scheduled(build, knobs):
    g = Graph({'P': Value('creg', reg=15), '2P': Value('creg', reg=14)})
    build(g)
    seq = ntt.schedule(g.ops, 14, knobs)
    reg = ntt.allocate(seq, {14, 15})
    return [ntt.emit(o, reg) for o in seq]


def h4_body(kind):
    def build(g):
        tw = twiddles(g, 'tx', 'ty')
        for j in range(4):
            quarters = [32 * (j + 4 * s) for s in range(4)]
            if kind == 'forward':
                forward(g, tw, quarters, canonical=False)
            else:
                inverse(g, tw, quarters, 'a')
    return scheduled(build, H4_KNOBS)


def bottom_body(kind):
    def build(g):
        for i in range(8):
            quarters = [128 * i + 32 * t for t in range(4)]
            if kind == 'forward':
                forward(g, twiddles(g, 'x', 'y', i), quarters, canonical=True)
            else:
                inverse(g, twiddles(g, 'x', 'y', i), quarters, 'from')
    return scheduled(build, BOTTOM_KNOBS)


HEAD = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14', '.p2align 5', '1:']
CLOBBERS = ntt.CLOBBERS + ', "cc"'


def h4_function(kind):
    # tx = table + 4 slot(k), ty = table + 4 slot(2k)
    slots = ['mov %[k], %[tx]', 'and $-8, %[tx]', 'add %[k], %[tx]', 'lea (%[table],%[tx],4), %[tx]',
             'lea (%[k],%[k]), %[ty]', 'and $-8, %[ty]', 'lea (%[ty],%[k],2), %[ty]', 'lea (%[table],%[ty],4), %[ty]']
    tail = ['add $512, %[a]', 'inc %[k]', 'cmp %[end], %[a]', 'jne 1b']
    doc = {'forward': 'Forward radix-4 butterflies at h = 4 of count >= 1 groups from group k (16 vectors each),\n'
                      'values < 4P; table: the twiddle table.',
           'inverse': 'Inverse radix-4 butterflies at h = 4 of count >= 1 groups from group k, values < 2P;\n'
                      'table: the inverse twiddle table.'}[kind]
    lines = [f'// {line}' for line in doc.split('\n')]
    lines += [f'[[gnu::noinline]] inline void {kind}_h4(Vec* a, std::size_t count, const std::uint32_t* table, std::size_t k) {{',
              '    Vec* const end = a + 16 * count;', '    const std::uint32_t *tx, *ty;', '    asm volatile(']
    lines += ntt.asm_lines(HEAD + slots + h4_body(kind) + tail)
    lines += ['        : [a] "+r"(a), [k] "+r"(k), [tx] "=&r"(tx), [ty] "=&r"(ty)',
              '        : [end] "r"(end), [table] "r"(table), [P] "m"(kP), [P2] "m"(k2P)',
              f'        : {CLOBBERS});', '}']
    return lines


def bottom_function(kind):
    tail = ['add $1024, %[a]'] + (['add $1024, %[from]'] if kind == 'inverse' else [])
    tail += ['add $64, %[x]', 'add $128, %[y]', 'cmp %[end], %[a]', 'jne 1b']
    doc = {'forward': 'Forward radix-4 butterflies at h = 1 of count groups from group k (4 vectors each),\n'
                      'canonical outputs: the leaves. count and k multiples of 8; x, y: slots k and 2k of the table.',
           'inverse': 'Inverse radix-4 butterflies at h = 1 of the count groups at from, into a; values < 2P.\n'
                      'count and k multiples of 8; x, y: slots k and 2k of the inverse table. from may be a.'}[kind]
    params = 'Vec* a, std::size_t count, const std::uint32_t* x, const std::uint32_t* y'
    outputs = '[a] "+r"(a), [x] "+r"(x), [y] "+r"(y)'
    if kind == 'inverse':
        params += ', const Vec* from'
        outputs += ', [from] "+r"(from)'
    lines = [f'// {line}' for line in doc.split('\n')]
    lines += [f'[[gnu::noinline]] inline void {kind}_bottom({params}) {{', '    Vec* const end = a + 4 * count;', '    asm volatile(']
    lines += ntt.asm_lines(HEAD + bottom_body(kind) + tail)
    lines += [f'        : {outputs}', '        : [end] "r"(end), [P] "m"(kP), [P2] "m"(k2P)',
              f'        : {CLOBBERS});', '}']
    return lines


HEADER = '''\
// Generated by lib/poly/gen_kernels.py. Do not edit.
// AVX2 loops of the transform's lowest radix-4 levels and of pruned transforms' column levels as
// inline assembly, scheduled for Zen 3.
// Table layout and arithmetic as lib/ntt's kernels (lib/ntt/notes.md). The forward loops read 4
// bytes past their last vector.
#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace poly::kernels {

using Vec = __m256i;

inline constexpr std::uint32_t kP = 998244353, k2P = 2 * kP;
'''


def columns_body(loop, spacing):
    """lib/ntt's loop body (two butterflies, scheduled with its knobs), the second butterfly spacing
    bytes after the first. spacing = 32 gives lib/ntt's own body."""
    consts = {'P': Value('creg', reg=15), '2P': Value('creg', reg=14)}
    for i, name in enumerate('XYZ'):
        consts['W' + name] = Value('cmem', mem=32 * i)
        consts['Q' + name] = Value('creg', reg=11 + i)
    ops = []
    for k in range(2):
        g = Graph(consts, disp=spacing * k)
        loop.graph(g)
        ops += g.ops
    seq = ntt.schedule(ops, 11, loop.knobs)
    reg = ntt.allocate(seq, {11, 12, 13, 14, 15})
    return [ntt.emit(o, reg) for o in seq]


def columns_function(kind, even):
    """Columns j < h with (j & width) == 0 of one radix-4 group, two per iteration: consecutive
    (width >= 2, chunks of width columns, then a skip of width) or, for width 1, j and j + 2."""
    loop = next(l for l in ntt.LOOPS if l.name == kind)
    asm = ntt.loop_asm(loop)
    head = asm[:asm.index('.p2align 5')]
    if even:
        asm = head + ['.p2align 5', '1:'] + columns_body(loop, 64) + ['add $128, %[a]', 'cmp %[end], %[a]', 'jne 1b']
    else:
        asm = head + ['1:', 'lea (%[a],%[width]), %[stop]', '.p2align 5', '2:'] + columns_body(loop, 32)
        asm += ['add $64, %[a]', 'cmp %[stop], %[a]', 'jne 2b', 'add %[width], %[a]', 'cmp %[end], %[a]', 'jne 1b']
    what = {'forward': 'Forward radix-4 butterflies, values < 4P', 'inverse': 'Inverse radix-4 butterflies, values < 2P'}[kind]
    if even:
        name, doc = f'{kind}_even_columns', [f'// {what}, on the columns j < h of one group with j even (inputs a[j + t h],',
                                             '// t < 4). h a multiple of 4; x, y as for lib/ntt\'s kernels (slots k and 2k).']
        params, width = 'Vec* a, std::size_t h', []
    else:
        name, doc = f'{kind}_columns', [f'// {what}, on the columns j < h of one group with (j & width) == 0 (inputs',
                                        '// a[j + t h], t < 4). width >= 2 a power of two, h a multiple of 2 width; x, y as for',
                                        '// lib/ntt\'s kernels (slots k and 2k).']
        params, width = 'Vec* a, std::size_t h, std::size_t width', [', [width] "r"(32 * width)']
    lines = doc + [f'[[gnu::noinline]] inline void {name}({params}, const std::uint32_t* x, const std::uint32_t* y) {{',
                   '    alignas(32) Vec w[3];  // twiddle values x, y, z; their quotients stay in ymm11-13',
                   '    Vec* const end = a + h;']
    if not even:
        lines += ['    Vec* stop;']
    lines += ['    asm volatile(']
    lines += ntt.asm_lines(asm)
    lines += ['        : [a] "+r"(a)' + ('' if even else ', [stop] "=&r"(stop)'),
              '        : [end] "r"(end), [h] "r"(32 * h), [h3] "r"(96 * h)' + ''.join(width) + ', [y] "r"(y), [x] "r"(x), [w] "r"(w),',
              '          [P] "m"(kP), [P2] "m"(k2P)',
              f'        : {CLOBBERS});', '}']
    return lines


def top8_forward(g, disp, upper):
    """Column of the radix-8 forward top level, in place: the source (canonical) in the quarters of
    a (lower half) or b (upper half). Inputs are loaded once into registers before any store."""
    c = g.c
    f = [g.materialize(Value('mem', mem=('b' if upper else 'a', t, disp))) for t in range(4)]
    h = [g.op('vpsubd', c['P'], x) for x in f] if upper else f  # the upper half's source, <= P
    s0, s1 = g.add(f[0], f[2]), g.add(f[1], f[3])
    d0 = g.op('vpsubd', g.add(f[0], c['P']), f[2])
    id1 = g.shoup(g.op('vpsubd', g.add(f[1], c['P']), f[3]), c['WX'], c['QX'])
    for t, v in enumerate([g.add(s0, s1), g.diff(s0, s1), g.add(d0, id1), g.diff(d0, id1)]):
        g.store(v, 'a', t, disp)
    ig2, ig3 = g.shoup(h[2], c['WX'], c['QX']), g.shoup(h[3], c['WX'], c['QX'])
    u, v = g.low(g.add(h[0], ig2)), g.low(g.diff(h[0], ig2))
    yu, zv = g.shoup(g.add(h[1], ig3), c['WY'], c['QY']), g.shoup(g.diff(h[1], ig3), c['WZ'], c['QZ'])
    for t, x in enumerate([g.add(u, yu), g.diff(u, yu), g.add(v, zv), g.diff(v, zv)]):
        g.store(x, 'b', t, disp)


TOP8_KNOBS = Knobs(window=28)


def top8_function(variant):
    """forward_top8_lower or forward_top8_upper: two columns j < q per iteration; a and b point to
    columns 0 and 4q (the halves), strides q vectors."""
    consts = {'P': Value('creg', reg=15), '2P': Value('creg', reg=14)}
    for i, name in enumerate('XYZ'):
        consts['W' + name] = Value('cmem', mem=32 * i)
        consts['Q' + name] = Value('creg', reg=11 + i)
    ops = []
    for disp in (0, 32):
        g = Graph(consts)
        top8_forward(g, disp, variant == 'upper')
        ops += g.ops
    seq = ntt.schedule(ops, 11, TOP8_KNOBS)
    reg = ntt.allocate(seq, {11, 12, 13, 14, 15})
    head = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14']
    for i in range(3):  # entries 1, 2, 3 of the table: values on the stack, quotients in ymm11-13
        head += [f'vbroadcastss {4 * (i + 1)}(%[t]), %%ymm0', f'vmovdqa %%ymm0, {32 * i}(%[w])',
                 f'vbroadcastss {4 * (i + 1) + 32}(%[t]), %%ymm{11 + i}']
    body = head + ['.p2align 5', '1:'] + [ntt.emit(o, reg) for o in seq]
    body += ['add $64, %[a]', 'add $64, %[b]', 'cmp %[end], %[a]', 'jne 1b']
    what = {'lower': 'the lower half (a[j + t q], t < 4)', 'upper': 'the upper half (a[j + (4 + t) q], t < 4)'}[variant]
    lines = [f'// Forward radix-8 top level of a transform of 8q vectors, in place, for a source in {what},',
             '// canonical; the other half is not read. The radix-2 level and the first radix-4 group of each',
             '// half: group 0 of the lower, group 1 of the upper. Outputs a[j + t q], t < 8, < 4P. q even;',
             '// table: the twiddle table.',
             f'[[gnu::noinline]] inline void forward_top8_{variant}(Vec* a, std::size_t q, const std::uint32_t* table) {{',
             '    alignas(32) Vec w[3];  // twiddle values of entries 1, 2, 3; their quotients stay in ymm11-13',
             '    Vec* b = a + 4 * q;', '    Vec* const end = a + q;', '    asm volatile(']
    lines += ntt.asm_lines(body)
    lines += ['        : [a] "+r"(a), [b] "+r"(b)',
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [t] "r"(table), [w] "r"(w), [P] "m"(kP),',
              '          [P2] "m"(k2P)',
              f'        : {CLOBBERS});', '}']
    return lines


def generate():
    lines = HEADER.split('\n')
    for kind in ('forward', 'inverse'):
        lines += h4_function(kind) + ['']
    for kind in ('forward', 'inverse'):
        lines += bottom_function(kind) + ['']
    for kind in ('forward', 'inverse'):
        for even in (False, True):
            lines += columns_function(kind, even) + ['']
    for variant in ('lower', 'upper'):
        lines += top8_function(variant) + ['']
    lines += ['}  // namespace poly::kernels']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = generate()
    if args.check:
        if OUT.read_text() != text:
            print(f'{OUT} is stale: run {Path(__file__).name}')
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
