#!/usr/bin/env python3
"""Generate kernels.hpp: the transform's two lowest radix-4 levels as GCC inline assembly loops.

Uses lib/ntt/gen_kernels.py's dataflow graphs, Zen 3 list scheduler and register allocator.
  forward_h4, inverse_h4: the level h = 4 over consecutive groups, one group (4 butterflies) per
    iteration; twiddle addresses from the group index, slot(k) = k + (k & ~7).
  forward_bottom, inverse_bottom: the level h = 1 (the leaves), 8 groups per iteration; their
    twiddles are one table block of x and two of y, z. Forward outputs are canonical.
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
// AVX2 loops of the transform's lowest radix-4 levels as inline assembly, scheduled for Zen 3.
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


def generate():
    lines = HEADER.split('\n')
    for kind in ('forward', 'inverse'):
        lines += h4_function(kind) + ['']
    for kind in ('forward', 'inverse'):
        lines += bottom_function(kind) + ['']
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
