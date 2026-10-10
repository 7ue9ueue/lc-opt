#!/usr/bin/env python3
"""Generate kernels.hpp, the asm kernels of product.hpp: lib/ntt's kernels (ntt::Product's bottom
stage and inverse top) rewritten for primes P < 2^28 with lazier reductions.

With 16P < 2^32 every value may grow to 16P, so most min(x, x - kP) steps of lib/ntt's kernels
drop out (ranges below; lib/ntt/notes.md has the arithmetic). Forward kernels take and return
values < 8P, inverse ones < 4P. The graphs, scheduler, register allocator and knobs are
lib/ntt's (gen_kernels.py, gen_product_kernels.py); 4P and 8P are memory operands.

Usage: gen_kernels.py [--check]   (--check: exit 1 if kernels.hpp differs from the output)
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'kernels.hpp'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(ROOT / 'lib/ntt'))
gk = load('gen_kernels', ROOT / 'lib/ntt/gen_kernels.py')
gpk = load('gen_product_kernels', ROOT / 'lib/ntt/gen_product_kernels.py')

# ------------------------------------------------------------------------------------- arithmetic


def reduce(g, x, bound):
    """x < 2 bound -> x mod bound, < bound: min(x, x - bound)."""
    return g.op('vpminud', x, g.op('vpsubd', x, g.c[bound]))


def bias(g, x, y, b):
    """x - y + b for y <= b."""
    return g.op('vpsubd', g.op('vpaddd', x, g.c[b]), y)


def forward_lazy(g, f, x, y, odd):
    """Radix-4 forward butterfly on f[0..3] < 8P with twiddles x, y, z (name -> (value, quotient)):
    returns outputs < 8P. odd: values with the odd lanes of f[2], f[3] in their even lanes."""
    a = reduce(g, f[0], '4P')  # < 4P
    cc = g.shoup(f[2], *x, odd=odd[0])  # < 2P
    dd = g.shoup(f[3], *x, odd=odd[1])
    ac, amc = g.add(a, cc), g.diff(a, cc)  # < 6P
    bd, bmd = g.add(f[1], dd), g.diff(f[1], dd)  # < 10P
    yb, zb = g.shoup(bd, *y[0]), g.shoup(bmd, *y[1])  # < 2P
    return [g.add(ac, yb), g.diff(ac, yb), g.add(amc, zb), g.diff(amc, zb)]


def inverse_lazy(g, f, x, y):
    """Radix-4 inverse butterfly on f[0..3] < 4P: outputs < 4P."""
    ab, cd = g.add(f[0], f[1]), g.add(f[2], f[3])  # < 8P
    amb = g.shoup(bias(g, f[0], f[1], '4P'), *y[0])  # < 2P
    cmd = g.shoup(bias(g, f[2], f[3], '4P'), *y[1])
    s = reduce(g, reduce(g, g.add(ab, cd), '8P'), '4P')  # < 16P -> < 4P
    return [s, g.add(amb, cmd), g.shoup(bias(g, ab, cd, '8P'), *x), g.shoup(g.diff(amb, cmd), *x)]


def twiddles(c):
    return (c['WX'], c['QX']), ((c['WY'], c['QY']), (c['WZ'], c['QZ']))


def forward_loop(g):
    x, y = twiddles(g.c)
    f = [g.quarter(i) for i in range(4)]
    for t, v in enumerate(forward_lazy(g, f, x, y, (g.quarter(2, 4), g.quarter(3, 4)))):
        g.store_quarter(v, t)


def inverse_loop(g):
    x, y = twiddles(g.c)
    for t, v in enumerate(inverse_lazy(g, [g.quarter(i) for i in range(4)], x, y)):
        g.store_quarter(v, t)


def forward_identity_loop(g):
    """Group 0 (x = y = 1, z = r[1]): f < 8P -> < 8P."""
    c = g.c
    f = [g.quarter(i) for i in range(4)]
    a, cc = reduce(g, f[0], '4P'), reduce(g, f[2], '4P')
    ac, amc = reduce(g, g.add(a, cc), '4P'), reduce(g, bias(g, a, cc, '4P'), '4P')
    bd = reduce(g, reduce(g, g.add(f[1], f[3]), '8P'), '4P')  # < 16P -> < 4P
    zb = g.shoup(bias(g, f[1], f[3], '8P'), c['WZ'], c['QZ'])
    outs = [g.add(ac, bd), bias(g, ac, bd, '4P'), g.add(amc, zb), g.diff(amc, zb)]
    for t, v in enumerate(outs):
        g.store_quarter(v, t)


def inverse_identity_loop(g):
    """Group 0: f < 4P -> < 4P."""
    c = g.c
    f = [g.quarter(i) for i in range(4)]
    ab, cd = reduce(g, g.add(f[0], f[1]), '4P'), reduce(g, g.add(f[2], f[3]), '4P')
    amb = reduce(g, bias(g, f[0], f[1], '4P'), '4P')
    cmd = g.shoup(bias(g, f[2], f[3], '4P'), c['WZ'], c['QZ'])
    outs = [g.add(ab, cd), g.add(amb, cmd), bias(g, ab, cd, '4P'), g.diff(amb, cmd)]
    for t, v in enumerate(outs):
        g.store_quarter(reduce(g, v, '4P'), t)


# ------------------------------------------------------------------------------------- emission

def operand(v, reg):
    if v.kind == 'cmem' and isinstance(v.mem, str):  # named constant operand
        return f'%[{v.mem}]'
    return gk.operand(v, reg)


def emit(o, reg):
    """gk.emit with named constant operands."""
    if o.store is not None:
        return f'vmovdqa {operand(o.srcs[0], reg)}, {gk.address(o.store)}'
    dst = f'%%ymm{reg[o.dst]}'
    if o.mnemonic in gk.LOADS:
        return f'{o.mnemonic} {operand(o.srcs[0], reg)}, {dst}'
    if o.mnemonic == 'vpsrlq':
        return f'vpsrlq ${o.imm}, {operand(o.srcs[0], reg)}, {dst}'
    x, y = o.srcs  # dst = x OP y is "op y, x, dst"; only y may be in memory
    if o.mnemonic in gk.COMMUTATIVE and x.in_memory:
        x, y = y, x
    imm = f'${o.imm}, ' if o.imm is not None else ''
    return f'{o.mnemonic} {imm}{operand(y, reg)}, {operand(x, reg)}, {dst}'


def base_consts():
    return {'P': gk.Value('creg', reg=15), '2P': gk.Value('creg', reg=14),
            '4P': gk.Value('cmem', mem='P4'), '8P': gk.Value('cmem', mem='P8')}


CONSTANT_INPUTS = '[P] "m"(kP), [P2] "m"(k2P), [P4] "m"(k4P), [P8] "m"(k8P)'
HEAD = ['vpbroadcastd %[P], %%ymm15', 'vpbroadcastd %[P2], %%ymm14']

# ------------------------------------------------------------------------------------- loops

LOOPS = [
    # name, graph, twiddles, pair, knobs (lib/ntt's), doc
    ('forward', forward_loop, 'xyz', False, gk.Knobs(),
     'Forward radix-4 butterflies j < h of one group on a[j + t h], t < 4; values < 8P.\n'
     'x, y: twiddles r[k], r[2k] in the block table; z = r[2k + 1] follows y. h even.'),
    ('forward_pair', forward_loop, 'xyz', True, gk.Knobs(seed=129, margin=1, loads_per_cycle=3),
     'forward() on a and b with the same twiddles, one butterfly of each per step.'),
    ('inverse', inverse_loop, 'xyz', False, gk.Knobs(seed=138, load_latency=16),
     'Inverse radix-4 butterflies j < h of one group, values < 4P; inverse twiddles. h even.'),
    ('forward_identity', forward_identity_loop, 'z', False, gk.Knobs(),
     'forward() for group 0 (x = y = 1); z = y[1]: pass the table start. h even.'),
    ('inverse_identity', inverse_identity_loop, 'z', False, gk.Knobs(),
     'inverse() for group 0; z = y[1]: pass the inverse table start. h even.'),
]


def loop_function(name, graph, kind, pair, knobs, doc):
    consts = base_consts()
    reserved = {14, 15}
    if kind == 'z':
        consts.update(WZ=gk.Value('creg', reg=12), QZ=gk.Value('creg', reg=13))
        reserved |= {12, 13}
    else:
        for i, n in enumerate('XYZ'):
            consts['W' + n] = gk.Value('cmem', mem=32 * i)
            consts['Q' + n] = gk.Value('creg', reg=11 + i)
            reserved.add(11 + i)
    ops = []
    for k in range(2):
        g = gk.Graph(consts, disp=0 if pair else 32 * k, array='b' if pair and k else 'a')
        graph(g)
        ops += g.ops
    seq = gk.schedule(ops, 16 - len(reserved), knobs)
    reg = gk.allocate(seq, reserved)
    head = list(HEAD)
    if kind == 'z':
        head += ['vbroadcastss 4(%[y]), %%ymm12', 'vbroadcastss 36(%[y]), %%ymm13']
    else:
        for i, (src, disp) in enumerate([('x', 0), ('y', 0), ('y', 4)]):
            head += [f'vbroadcastss {gk.address((src, None, disp))}, %%ymm0',
                     f'vmovdqa %%ymm0, {gk.address(("w", None, 32 * i))}',
                     f'vbroadcastss {gk.address((src, None, disp + 32))}, %%ymm{11 + i}']
    tail = (['add $32, %[b]'] if pair else []) + [f'add ${32 if pair else 64}, %[a]', 'cmp %[end], %[a]', 'jne 1b']
    body = head + ['.p2align 5', '1:'] + [emit(o, reg) for o in seq] + tail
    args = 'Vec* a, Vec* b, std::size_t h' if pair else 'Vec* a, std::size_t h'
    twiddle_args = 'const std::uint32_t* x, const std::uint32_t* y' if kind == 'xyz' else 'const std::uint32_t* y'
    outputs = '[a] "+r"(a)' + (', [b] "+r"(b)' if pair else '')
    inputs = '[end] "r"(end), [h] "r"(32 * h), [h3] "r"(96 * h), [y] "r"(y)'
    if kind == 'xyz':
        inputs += ', [x] "r"(x), [w] "r"(w)'
    lines = [f'// {line}' for line in doc.split('\n')]
    lines += [f'inline void {name}({args}, {twiddle_args}) {{']
    if kind == 'xyz':
        lines += ['    alignas(32) Vec w[3];  // twiddle values x, y, z; their quotients stay in ymm11-13']
    lines += ['    Vec* const end = a + h;', '    asm volatile(']
    lines += gk.asm_lines(body)
    lines += [f'        : {outputs}', f'        : {inputs},', f'          {CONSTANT_INPUTS}',
              '        : ' + gk.CLOBBERS + ', "cc");', '}', '']
    return lines


# ------------------------------------------------------------------------------------- bottom

# lib/ntt's knobs, but 'both' from timing 100 knob sets on EPYC 7B13 (118.3 cycles per group;
# lib/ntt's set deadlocks the scheduler with these graphs, and window 16 gives 122.9).
BOTTOM_KNOBS = dict(gpk.KNOBS, both=gk.Knobs(margin=1, load_latency=6, window=20))

def forward_h1(g, array, x, y):
    tw = lambda base, d: (g.scalar(base, d), g.scalar(base, d + 32))
    f = [g.at(array, 32 * t) for t in range(4)]
    return forward_lazy(g, f, tw(x, 0), (tw(y, 0), tw(y, 4)), (g.at(array, 68), g.at(array, 100)))


def bottom_first(g):
    """Forward butterflies of a and b (inputs < 8P); leaf rows [w_t A_t, A_t] and B_t, all <= 2P.
    w_t = y, -y, z, -z: rows t = 1, 3 hold 2P - (y A_t mod 2P)."""
    for t, v in enumerate(forward_h1(g, 'a', 'x', 'y')):
        a = reduce(g, reduce(g, v, '4P'), '2P')
        g.store(a, 'buf', None, 64 * t + 32)
        w, q = (0, 32) if t < 2 else (4, 36)
        wa = g.shoup(a, g.scalar('y', w), g.scalar('y', q))
        if t % 2:
            wa = g.op('vpsubd', g.c['2P'], wa)
        g.store(wa, 'buf', None, 64 * t)
    for t, v in enumerate(forward_h1(g, 'b', 'x', 'y')):
        g.store(reduce(g, reduce(g, v, '4P'), '2P'), 'buf', None, 256 + 32 * t)


def bottom_last(g):
    """Leaf products: sums of 8 products < 4P^2 (< 2^61), Montgomery reduction to < 3P, then the
    inverse butterfly (inputs < 4P) into out."""
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
        leaves.append(g.op('vpblendd', g.op('vpsrlq', e, imm=32), o, imm=0xAA))
    tw = lambda base, d: (g.scalar(base, d), g.scalar(base, d + 32))
    outs = inverse_lazy(g, leaves, tw('ix', 0), (tw('iy', 0), tw('iy', 4)))
    for t, v in enumerate(outs):
        g.store(v, 'out', None, 32 * t)


def bottom_function(part):
    consts = base_consts()
    consts['NI'] = gk.Value('creg', reg=13)
    g = gpk.Batch(consts)
    if part in ('last', 'both'):
        for m in range(gpk.BATCHES):
            g.offset = {'cur': gpk.LEAVES * m, 'out': 128 * m, 'ix': 4 * m, 'iy': 8 * m}
            bottom_last(g)
    split = len(g.ops)
    if part in ('first', 'both'):
        for m in range(gpk.BATCHES):
            g.offset = {'a': 128 * m, 'b': 128 * m, 'buf': gpk.LEAVES * m, 'x': 4 * m, 'y': 8 * m}
            bottom_first(g)
    ops = g.ops
    if part == 'both':  # interleave the two stages' source orders proportionally
        older, newer = ops[:split], ops[split:]
        keyed = [(i / len(older), 0, i, o) for i, o in enumerate(older)]
        keyed += [(j / len(newer) + gpk.SHIFT, 1, j, o) for j, o in enumerate(newer)]
        ops = [o for *_, o in sorted(keyed, key=lambda t: t[:3])]
    seq = gk.schedule(ops, 13, BOTTOM_KNOBS[part])
    reg = gk.allocate(seq, {13, 14, 15})
    body = HEAD + ['vpbroadcastd %[NI], %%ymm13'] + [emit(o, reg) for o in seq]
    args = gpk.ARGS[part]
    params = ', '.join(f'{"void*" if a in ("buf", "out") else "const void*"} {a}' for a in args)
    lines = [f'// {gpk.DOC[part]}', f'[[gnu::always_inline]] inline void bottom_{part}({params}) {{',
             '    asm volatile(']
    lines += gk.asm_lines(body)
    lines += ['        :', '        : ' + ', '.join(f'[{a}] "r"({a})' for a in args) + ',',
              f'          {CONSTANT_INPUTS}, [NI] "m"(kNI)', '        : ' + gk.CLOBBERS + ');', '}', '']
    return lines


# ------------------------------------------------------------------------------------- top levels

TOP_TWIDDLES = ['S', 'SZ0', 'SY1', 'SZ1', 'SX1', 'X1']


def top_column(ga, gb):
    """gen_product_kernels.top_column for inputs < 4P."""
    c = ga.c
    shoup = lambda g, x, name: g.shoup(x, c['W' + name], c['Q' + name])
    f = [ga.quarter(i) for i in range(4)]
    sab, samb = shoup(ga, ga.add(f[0], f[1]), 'S'), shoup(ga, bias(ga, f[0], f[1], '4P'), 'S')
    scd, scmd = shoup(ga, ga.add(f[2], f[3]), 'S'), shoup(ga, bias(ga, f[2], f[3], '4P'), 'SZ0')
    u = [ga.low(ga.add(sab, scd)), ga.low(ga.add(samb, scmd)),
         ga.low_signed(ga.op('vpsubd', sab, scd)), ga.low_signed(ga.op('vpsubd', samb, scmd))]  # < 2P
    g = [gb.quarter(i) for i in range(4)]
    ab, cd = gb.add(g[0], g[1]), gb.add(g[2], g[3])  # < 8P
    amb, cmd = shoup(gb, bias(gb, g[0], g[1], '4P'), 'SY1'), shoup(gb, bias(gb, g[2], g[3], '4P'), 'SZ1')
    v = [shoup(gb, gb.add(ab, cd), 'S'), gb.low(gb.add(amb, cmd)),
         shoup(gb, bias(gb, ab, cd, '8P'), 'SX1'), shoup(gb, gb.diff(amb, cmd), 'X1')]  # < 2P
    for t in range(4):
        ga.store_quarter(ga.canonical(ga.add(u[t], v[t])), t)
        gb.store_quarter(gb.canonical(gb.diff(u[t], v[t])), t)


def top_function():
    consts = base_consts()
    for i, name in enumerate(TOP_TWIDDLES):  # w[2i] value, w[2i + 1] quotient, broadcast
        consts['W' + name] = gk.Value('cmem', mem=64 * i)
        consts['Q' + name] = gk.Value('cmem', mem=64 * i + 32)
    ga, gb = gk.Graph(consts, array='a'), gk.Graph(consts, array='b')
    gb.ops = ga.ops  # one column: two address bases, one op list
    top_column(ga, gb)
    seq = gk.schedule(ga.ops, 14, gk.Knobs())
    reg = gk.allocate(seq, {14, 15})
    body = HEAD + ['.p2align 5', '1:'] + [emit(o, reg) for o in seq]
    body += ['add $32, %[a]', 'add $32, %[b]', 'cmp %[end], %[a]', 'jne 1b']
    lines = [
        '// Inverse top level of a transform of 8q vectors (inverse_identity, inverse and a radix-2 level',
        '// with the scale, in one pass): blocks a[tq, (t+1)q), t < 4, hold the identity group, t >= 4',
        '// group 1, values < 4P. Writes the canonical scaled result. w: twiddle vectors s, s z0, s y1,',
        '// s z1, s x1, x1, each followed by its Shoup quotient.',
        'inline void inverse_top(Vec* a, std::size_t q, const Vec* w) {',
        '    Vec* b = a + 4 * q;', '    Vec* const end = a + q;', '    asm volatile(']
    lines += gk.asm_lines(body)
    lines += ['        : [a] "+r"(a), [b] "+r"(b)',
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [w] "r"(w),',
              f'          {CONSTANT_INPUTS}', '        : ' + gk.CLOBBERS + ', "cc");', '}', '']
    return lines


RADIX8_CONSTANTS = ['WI', 'QI', 'WY', 'QY', 'WZ', 'QZ']


def radix8_column(g):
    """First level from x < 4P: the radix-4 groups 0 (twiddles 1, 1, i) and 1 (i, y, z) of
    lib/multimod's forward_radix8; outputs < 8P."""
    c = g.c
    f = [g.quarter(t) for t in range(4)]
    i, y, z = (c['WI'], c['QI']), (c['WY'], c['QY']), (c['WZ'], c['QZ'])
    g0, g1 = reduce(g, g.add(f[0], f[2]), '4P'), reduce(g, g.add(f[1], f[3]), '4P')  # < 4P
    h0 = reduce(g, bias(g, f[0], f[2], '4P'), '4P')
    ih1 = g.shoup(bias(g, f[1], f[3], '4P'), *i)
    outs = [g.add(g0, g1), bias(g, g0, g1, '4P'), g.add(h0, ih1), g.diff(h0, ih1)]
    if2, if3 = g.shoup(f[2], *i, odd=g.quarter(2, 4)), g.shoup(f[3], *i, odd=g.quarter(3, 4))
    u0, v0 = g.add(f[0], if2), g.diff(f[0], if2)  # < 6P
    yu1, zv1 = g.shoup(g.add(f[1], if3), *y), g.shoup(g.diff(f[1], if3), *z)
    outs += [g.add(u0, yu1), g.diff(u0, yu1), g.add(v0, zv1), g.diff(v0, zv1)]
    for k, v in enumerate(outs):
        g.store(v, 'a' if k < 4 else 'b', k % 4, 0)


def radix8_function():
    consts = base_consts()
    consts.update({name: gk.Value('cmem', mem=32 * i) for i, name in enumerate(RADIX8_CONSTANTS)})
    g = gk.Graph(consts, array='x')
    radix8_column(g)
    seq = gk.schedule(g.ops, 14, gk.Knobs())
    reg = gk.allocate(seq, {14, 15})
    body = HEAD + ['.p2align 5', '1:'] + [emit(o, reg) for o in seq]
    body += ['add $32, %[x]', 'add $32, %[a]', 'add $32, %[b]', 'cmp %[end], %[a]', 'jne 1b']
    lines = [
        '// First level of a factor of 64q words whose upper half is zero: reads x[0, 32q), words < 4P,',
        '// and writes the first radix-4 group of both halves (twiddles 1, 1, r[1] and r[1], r[2], r[3]),',
        '// f[0, 8q), values < 8P. f may be x\'s storage: each column is read before it is written.',
        '// Reads 4 bytes past x[32q). x and f 32-byte aligned, q >= 1. w: broadcast twiddles r[1],',
        '// r[2], r[3], each followed by its Shoup quotient.',
        'inline void forward_radix8(Vec* f, std::size_t q, const std::uint32_t* x, const Vec* w) {',
        '    Vec* b = f + 4 * q;', '    Vec* const end = f + q;', '    asm volatile(']
    lines += gk.asm_lines(body)
    lines += ['        : [a] "+r"(f), [b] "+r"(b), [x] "+r"(x)',
              '        : [end] "r"(end), [h] "r"(32 * q), [h3] "r"(96 * q), [w] "r"(w),',
              f'          {CONSTANT_INPUTS}', '        : ' + gk.CLOBBERS + ', "cc");', '}', '']
    return lines


HEADER = f'''\
// Generated by {HERE.relative_to(ROOT)}/gen_kernels.py. Do not edit.
// AVX2 kernels of product.hpp: lib/ntt's transform kernels for a prime P < 2^28 set at run time
// (lib/multimod's kP, k2P, kNI, and k4P, k8P here), with lazier reductions: forward kernels take
// and return values < 8P, inverse kernels < 4P. A twiddle w < P comes with its Shoup quotient
// 8 words later (block table layout). Forward kernels read 4 bytes past their last vector.
#pragma once

#include <cstddef>
#include <cstdint>

#include "lib/multimod/kernels.hpp"

namespace lazy::kernels {{

using multimod::kernels::k2P;
using multimod::kernels::kNI;
using multimod::kernels::kP;
using multimod::kernels::Vec;

inline Vec k4P, k8P;  // broadcast 4P and 8P
'''


def generate() -> str:
    lines = HEADER.split('\n')
    for loop in LOOPS:
        lines += loop_function(*loop)
    lines += ['// Bottom stage, ntt::Product\'s: two radix-4 groups with h = 1 per statement. A batch of four',
              '// vectors is a group; each output vector v_t is a leaf, a polynomial mod x^8 - w_t with',
              '// w_t = y, -y, z, -z from the group\'s twiddles y = r[2k], z = r[2k + 1]. buf, cur: leaf',
              '// buffers, 384 bytes per batch, 64-byte aligned. x, y, ix, iy: twiddles of the first batch,',
              '// whose group index is even. Inputs < 8P, outputs < 4P.', '']
    for part in ('first', 'last', 'both'):
        lines += bottom_function(part)
    lines += top_function() + radix8_function()
    lines += ['}  // namespace lazy::kernels']
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = generate()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f'{OUT} is stale: run {Path(__file__).name}')
            return 1
        return 0
    OUT.write_text(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
