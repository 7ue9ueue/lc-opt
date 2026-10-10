#!/usr/bin/env python3
"""Generate bottom.hpp: the NTT's bottom stage for this problem, two groups per asm statement.

Built from lib/ntt/gen_kernels.py (same dataflow graphs, scheduler and register allocator), with
three changes measured on Zen 3 (notes.md):
- Two batches (radix-4 groups with h = 1) per statement. The caller passes an even group index,
  so the second batch's twiddles are in the same 8-entry table block: x + 4 and y + 8 bytes.
- No leaf weight array: the leaf moduli x^8 - w_t of group k have w_t = y, -y, z, -z with y, z the
  group's own forward twiddles r[2k], r[2k + 1]. The windows of t = 1, 3 hold P - canonical(y A_t)
  (in (0, P]) instead of a product with P - y.
- Leaf products in broadcast-reuse form: each coefficient of B is broadcast once and multiplies
  two windows read as memory operands.
The functions are always inlined, so the operands stay in registers between statements.

Usage: gen_bottom.py [--check]   (--check: exit 1 if bottom.hpp differs from the output)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'lib' / 'ntt'))
import gen_kernels as gk  # noqa: E402

OUT = HERE / 'bottom.hpp'
BATCHES = 2
LEAVES = 384  # leaf buffer bytes per batch: window[4][16], then coefficients[4][8]
KNOBS = {'first': gk.Knobs(window=12),
         'last': gk.Knobs(seed=7, margin=2, jitter=3.0, window=12),
         'both': gk.Knobs(window=20)}
SHIFT = -0.05  # 'both': the first stage's ops start this fraction earlier in source order
ARGS = {'first': ['a', 'b', 'buf', 'x', 'y'], 'last': ['out', 'cur', 'ix', 'iy'],
        'both': ['a', 'b', 'buf', 'x', 'y', 'out', 'cur', 'ix', 'iy']}
DOC = {
    'first': 'Forward half of the bottom stage for the two batches at a, b; fills buf.',
    'last': 'Leaf products from cur and the inverse butterflies; writes the two batches at out.',
    'both': 'last() for the current two batches interleaved with first() for the next two.',
}


class Batch(gk.Graph):
    """A graph whose addresses move by a fixed offset per base: batch m of a statement."""

    def __init__(self, consts):
        super().__init__(consts, share_broadcasts=False)
        self.offset = {}

    def at(self, base, disp, fold=True):
        return super().at(base, disp + self.offset.get(base, 0), fold)

    def scalar(self, base, disp):
        return super().scalar(base, disp + self.offset.get(base, 0))

    def store(self, v, base, index, disp):
        super().store(v, base, index, disp + self.offset.get(base, 0))


def first(g):
    """Forward butterflies of a and b; leaf buffer rows [w_t A_t, A_t] and B_t, all in [0, P]."""
    for t, v in enumerate(gk.forward_h1(g, 'a', 'x', 'y')):
        a = g.canonical(v)
        g.store(a, 'buf', None, 64 * t + 32)
        w, q = (0, 32) if t < 2 else (4, 36)  # y or z and its quotient
        wa = g.shoup(a, g.scalar('y', w), g.scalar('y', q))
        wa = g.op('vpminud', wa, g.op('vpsubd', wa, g.c['P']))
        if t % 2:
            wa = g.op('vpsubd', g.c['P'], wa)  # (P - w) A = P - w A
        g.store(wa, 'buf', None, 64 * t)
    for t, v in enumerate(gk.forward_h1(g, 'b', 'x', 'y')):
        g.store(g.canonical(v), 'buf', None, 256 + 32 * t)


def last(g):
    """Leaf t: e (even lanes) and o (odd lanes) sum b_j times the window 8 - j and 9 - j words in;
    sums < 8 P^2, then Montgomery reduction and the inverse butterfly as gen_kernels.bottom_last."""
    c = g.c
    leaves = []
    for t in range(4):
        e = o = None
        for j in range(8):
            b = g.materialize(g.scalar('cur', 256 + 32 * t + 4 * j))
            pe = g.op('vpmuludq', g.at('cur', 64 * t + 4 * (8 - j)), b)
            po = g.op('vpmuludq', g.at('cur', 64 * t + 4 * (9 - j)), b)
            e = pe if e is None else g.op('vpaddq', e, pe)
            o = po if o is None else g.op('vpaddq', o, po)
        e = g.op('vpaddq', e, g.op('vpmuludq', g.op('vpmuludq', e, c['NI']), c['P']))
        o = g.op('vpaddq', o, g.op('vpmuludq', g.op('vpmuludq', o, c['NI']), c['P']))
        leaves.append(g.low(g.op('vpblendd', g.op('vpsrlq', e, imm=32), o, imm=0xAA)))
    tw = {'WX': g.scalar('ix', 0), 'QX': g.scalar('ix', 32), 'WY': g.scalar('iy', 0), 'QY': g.scalar('iy', 32),
          'WZ': g.scalar('iy', 4), 'QZ': g.scalar('iy', 36)}
    r = leaves
    ab, cd = g.low(g.add(r[0], r[1])), g.low(g.add(r[2], r[3]))
    amb = g.shoup(g.diff(r[0], r[1]), tw['WY'], tw['QY'])
    cmd = g.shoup(g.diff(r[2], r[3]), tw['WZ'], tw['QZ'])
    g.store(g.low(g.add(ab, cd)), 'out', None, 0)
    g.store(g.low(g.add(amb, cmd)), 'out', None, 32)
    g.store(g.shoup(g.diff(ab, cd), tw['WX'], tw['QX']), 'out', None, 64)
    g.store(g.shoup(g.diff(amb, cmd), tw['WX'], tw['QX']), 'out', None, 96)


def body(part):
    g = Batch({'P': gk.Value('creg', reg=15), '2P': gk.Value('creg', reg=14), 'NI': gk.Value('creg', reg=13)})
    if part in ('last', 'both'):
        for m in range(BATCHES):
            g.offset = {'cur': LEAVES * m, 'out': 128 * m, 'ix': 4 * m, 'iy': 8 * m}
            last(g)
    split = len(g.ops)
    if part in ('first', 'both'):
        for m in range(BATCHES):
            g.offset = {'a': 128 * m, 'b': 128 * m, 'buf': LEAVES * m, 'x': 4 * m, 'y': 8 * m}
            first(g)
    ops = g.ops
    if part == 'both':  # interleave the two stages' source orders proportionally
        older, newer = ops[:split], ops[split:]
        keyed = [(i / len(older), 0, i, o) for i, o in enumerate(older)]
        keyed += [(j / len(newer) + SHIFT, 1, j, o) for j, o in enumerate(newer)]
        ops = [o for *_, o in sorted(keyed, key=lambda t: t[:3])]
    seq = gk.schedule(ops, 13, KNOBS[part])
    reg = gk.allocate(seq, {13, 14, 15})
    head = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14', 'vpbroadcastd %[NI], %%ymm13']
    return head + [gk.emit(o, reg) for o in seq]


def function(part):
    args = ARGS[part]
    params = ', '.join(f'{"void*" if a in ("buf", "out") else "const void*"} {a}' for a in args)
    lines = [f'// {DOC[part]}', f'[[gnu::always_inline]] inline void {part}({params}) {{', '    asm volatile(']
    lines += gk.asm_lines(body(part))
    lines += ['        :', '        : ' + ', '.join(f'[{a}] "r"({a})' for a in args) + ',',
              '          [P] "m"(kP), [P2] "m"(k2P), [NI] "m"(kNI)',
              '        : ' + gk.CLOBBERS + ');', '}']
    return lines


HEADER = '''\
// Generated by gen_bottom.py. Do not edit.
// The NTT's bottom stage, two radix-4 groups with h = 1 per statement (lib/ntt/kernels.hpp has one).
// A batch of four vectors is a group; each output vector v_t is a leaf, a polynomial mod x^8 - w_t
// with w_t = y, -y, z, -z from the group's twiddles y = r[2k], z = r[2k + 1]. buf, cur: leaf
// buffers, 384 bytes per batch, 64-byte aligned. x, y, ix, iy: twiddles of the first batch, whose
// group index is even.
#pragma once

#include "lib/ntt/kernels.hpp"

namespace bottom_kernels {

using ntt::kernels::k2P;
using ntt::kernels::kNI;
using ntt::kernels::kP;
'''


def generate():
    lines = HEADER.split('\n')
    for part in ('first', 'last', 'both'):
        lines += function(part) + ['']
    lines += ['}  // namespace bottom_kernels']
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
