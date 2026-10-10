#!/usr/bin/env python3
"""Generate this problem's asm kernels: bottom.hpp (the NTT's bottom stage, two groups per asm
statement) and top.hpp (the inverse top level with the scale).

Built from lib/ntt/gen_kernels.py (same dataflow graphs, scheduler and register allocator).
bottom.hpp has three changes measured on Zen 3 (notes.md):
- Two batches (radix-4 groups with h = 1) per statement. The caller passes an even group index,
  so the second batch's twiddles are in the same 8-entry table block: x + 4 and y + 8 bytes.
- No leaf weight array: the leaf moduli x^8 - w_t of group k have w_t = y, -y, z, -z with y, z the
  group's own forward twiddles r[2k], r[2k + 1]. The windows of t = 1, 3 hold P - canonical(y A_t)
  (in (0, P]) instead of a product with P - y.
- Leaf products in broadcast-reuse form: each coefficient of B is broadcast once and multiplies
  two windows read as memory operands.
Its functions are always inlined, so the operands stay in registers between statements.
top.hpp fuses lib/ntt's inverse_identity, inverse (group 1) and scale_radix2 into one pass, with
the scale folded into the twiddles: 5 more Shoup products per column instead of 8.

Usage: gen_asm.py [--check]   (--check: exit 1 if a header differs from the output)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'lib' / 'ntt'))
import gen_kernels as gk  # noqa: E402

OUT = HERE / 'bottom.hpp'
TOP_OUT = HERE / 'top.hpp'
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


# Inverse top of a transform of 8q vectors: blocks t < 4 (a) are the identity group's quarters,
# blocks 4 + t (b = a + 4q) group 1's; a column is vector j of each block. With twiddles z0 (group 0),
# x1, y1, z1 (group 1) and the scale s, values < 2P in:
#   U = s * inverse_identity(f), V = s * inverse(g); out_t = canonical(U_t + V_t),
#   out_{4+t} = canonical(U_t - V_t + 2P),
# where s is folded into the twiddles: U uses s, s z0; V uses s, s y1, s z1, s x1 and x1.
TOP_TWIDDLES = ['S', 'SZ0', 'SY1', 'SZ1', 'SX1', 'X1']


def top_column(ga, gb):
    c = ga.c
    shoup = lambda g, x, name: g.shoup(x, c['W' + name], c['Q' + name])
    f = [ga.quarter(i) for i in range(4)]
    sab, samb = shoup(ga, ga.add(f[0], f[1]), 'S'), shoup(ga, ga.diff(f[0], f[1]), 'S')
    scd, scmd = shoup(ga, ga.add(f[2], f[3]), 'S'), shoup(ga, ga.diff(f[2], f[3]), 'SZ0')
    u = [ga.low(ga.add(sab, scd)), ga.low(ga.add(samb, scmd)),
         ga.low_signed(ga.op('vpsubd', sab, scd)), ga.low_signed(ga.op('vpsubd', samb, scmd))]
    g = [gb.quarter(i) for i in range(4)]
    ab, cd = gb.low(gb.add(g[0], g[1])), gb.low(gb.add(g[2], g[3]))
    amb, cmd = shoup(gb, gb.diff(g[0], g[1]), 'SY1'), shoup(gb, gb.diff(g[2], g[3]), 'SZ1')
    v = [shoup(gb, gb.add(ab, cd), 'S'), gb.low(gb.add(amb, cmd)),
         shoup(gb, gb.diff(ab, cd), 'SX1'), shoup(gb, gb.diff(amb, cmd), 'X1')]
    for t in range(4):
        ga.store_quarter(ga.canonical(ga.add(u[t], v[t])), t)
        gb.store_quarter(gb.canonical(gb.diff(u[t], v[t])), t)


def top_body():
    consts = {'P': gk.Value('creg', reg=15), '2P': gk.Value('creg', reg=14)}
    for i, name in enumerate(TOP_TWIDDLES):  # w[2i] value, w[2i + 1] quotient, broadcast
        consts['W' + name] = gk.Value('cmem', mem=64 * i)
        consts['Q' + name] = gk.Value('cmem', mem=64 * i + 32)
    ga, gb = gk.Graph(consts, array='a'), gk.Graph(consts, array='b')
    gb.ops = ga.ops  # one column: two address bases, one op list
    top_column(ga, gb)
    seq = gk.schedule(ga.ops, 14, gk.Knobs())
    reg = gk.allocate(seq, {14, 15})
    head = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14', '.p2align 5', '1:']
    tail = ['add $32, %[a]', 'add $32, %[b]', 'cmp %[end], %[a]', 'jne 1b']
    return head + [gk.emit(o, reg) for o in seq] + tail


TOP_HEADER = '''\
// Generated by gen_asm.py. Do not edit.
// Inverse top level of a transform of 8q vectors (lib/ntt runs it as inverse_identity, inverse and
// scale_radix2): blocks a[tq, (t+1)q), t < 4, hold the identity group, t >= 4 group 1, values < 2P.
// Writes the canonical scaled result. w: twiddle vectors s, s z0, s y1, s z1, s x1, x1, each
// followed by its Shoup quotient (gen_asm.py).
#pragma once

#include "lib/ntt/kernels.hpp"

namespace top_kernels {

using ntt::kernels::k2P;
using ntt::kernels::kP;
using ntt::kernels::Vec;

inline void inverse_top(Vec* a, std::size_t q, const Vec* w) {
    Vec* b = a + 4 * q;
    Vec* const end = a + q;
    asm volatile(
'''


def generate_top():
    lines = TOP_HEADER.split('\n')[:-1]
    lines += gk.asm_lines(top_body())
    lines += ['        : [a] "+r"(a), [b] "+r"(b)',
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [w] "r"(w), [P] "m"(kP), [P2] "m"(k2P)',
              '        : ' + gk.CLOBBERS + ', "cc");', '}', '', '}  // namespace top_kernels']
    return '\n'.join(lines) + '\n'


HEADER = '''\
// Generated by gen_asm.py. Do not edit.
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
    outputs = {OUT: generate(), TOP_OUT: generate_top()}
    if args.check:
        stale = [path.name for path, text in outputs.items() if not path.exists() or path.read_text() != text]
        if stale:
            print(f'{", ".join(stale)} stale: run {Path(__file__).name}')
            return 1
        return 0
    for path, text in outputs.items():
        path.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
