# polynomial_interpolation_on_geometric_sequence

N <= 2^19 values y_i at the distinct points a r^i mod 998244353; print the N coefficients of the
f of degree < N through them. 5 s. Slowest tests: max_random_00..02 and y0_00 (N = 2^19),
random_00/01 and pow_rN_equal_1_01 (N > 2^18). near_pow2 (N = 2^18 - 2 .. 2^18 + 2) takes the
same transform length, 2^20. nar_0 has N <= 1 (a, r may be 0), small N <= 5.

Best judged: ours, 15 ms: [409510](https://judge.yosupo.jp/submission/409510) (`main.cpp` of #239),
clean (`tools/spikes.py`).
Record when opened (issue #78): 90 ms.

## Design

- Newton's form on the nodes x_i = a r^i. With t(k) = k (k - 1) / 2 and
  Q_k = (1 - r)(1 - r^2)...(1 - r^k) (nonzero for k < N, as the nodes are distinct):
  - divided differences: prod_(j <= k, j != i) (x_i - x_j) = a^k (-1)^i r^(t(i) + i (k - i))
    Q_i Q_(k-i), and t(i) + i (k - i) = t(k) - t(k - i), so d_k = a^-k r^-t(k) E_k with
    E = Y K mod x^N, Y_i = (-1)^i y_i / Q_i, K_j = r^t(j) / Q_j;
  - Newton basis: prod_(j<k) (x - a r^j) = sum_l (-a)^(k-l) r^t(k-l) [k, l]_r x^l (q-binomial
    theorem), [k, l]_r = Q_k / (Q_l Q_(k-l)), so c_l = (-a)^-l / Q_l sum_(k >= l) D_k K_(k-l)
    with D_k = (-1)^k Q_k r^-t(k) E_k.
  Both steps are the lower half of a product with the same K: with D reversed, one cyclic
  convolution of length L = 2^lg >= 2N each, and the second reuses K's transform. Five
  transforms of length 2^20 at the maximum. Checked first in Python against O(N^2) Lagrange,
  r^N = 1 included (scratch `proto.py`).
- Counted against the evaluation route (weights y_i / M'(x_i) in closed form, power sums by a
  chirp z middle product, then the product with the master polynomial M): two products as well,
  but no shared factor, so six transforms.
- Transforms: `ntt::Product`'s layout (`lib/ntt/product.hpp`), as in
  multipoint_evaluation_on_geometric_sequence (#76): `ntt::detail::forward_radix8` for factors in
  the lower half, the subtrees with `product_kernels`' bottom stage, the halves' top inverse
  groups, and the last radix-2 level folded into the next scan. The subtrees are a local copy of
  `ntt::detail::Subtrees` with one option: `Subtrees<false>` skips b's forward levels, as b keeps
  K's transform down to the level above the bottom stage, which only reads b. L = 2^lg >= 2N with
  lg even (2^20 also for N <= 2^18).
- Scans: Q, 1/Q, the chirps and the geometric factors are products along k. 32 lanes (4 vectors)
  each walk a chunk of C positions (C a multiple of 16, C / 16 odd); a step is
  x <- x (g + o) / 2^32 (Montgomery) and g <- g w (Shoup), and every 8 steps an 8 x 8 transpose
  puts each lane's 8 values in natural order. Five scans: lane totals of Q (forward, no output),
  1/Q for Y (backward, in place over y), K (backward), D (forward, reads E's halves, writes D
  reversed) and c (backward, reads G's halves reversed). Backward scans restart the lane that
  crosses N at position N - 1, so padding terms (where 1 - r^k may be 0) do no harm.
- N = 0: an empty line. N <= 32: Lagrange in O(N^2) (covers a = 0 and r = 0, only possible for
  N <= 2).
- `lib/io` input (`io::read_bulk`) into the first buffer; output in 10-byte fixed-width fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific). One mapping in huge pages
  for the three buffers, the tables and the text. The program runs from `.preinit_array` and ends
  with `_exit`.

## Floor

`lc-amd`, whole process, judge flags, max_random_00..02 (scratch `timeit.py`: judge.py's build and
runner without the checker): read N + 3 numbers with `io::read_bulk` and write N values with
`fields.hpp`, nothing else (scratch `floor.cpp`): 4.99 ms median (11 rounds), 5.12 (21 rounds,
busier VM).

## Log

- 2026-10-10, claude (round 1): first solution. All on `lc-amd`, judge flags, max_random_00
  unless noted. Phases: medians of 7-9 runs of a probe with `CLOCK_MONOTONIC` stamps (stdout to
  /dev/null). Scratch files: `lc-opt-explore/polynomial_interpolation_on_geometric_sequence/`.
  - Phases (ms): parse 0.82, tables 0.08, scans 0.13 (totals) + 0.38 (Y) + 0.34 (K) + 0.51 (D)
    + 0.45 (c), radix-8 passes 0.12 (a) + 0.17 (b) + 0.12 (d), product 1: subtrees 1.89 + 1.91,
    top inverse groups 0.05 + 0.07; product 2: 1.48 + 1.50, 0.05 + 0.07; format and write (to
    /dev/null) 0.34. 10.55 in process. Skipping b's forward levels saves 0.82 ms of 3.80.
  - Scans: the totals scan (chain only) runs at 6.7 cycles per vector step, near the 10
    multiplies of one Montgomery and one Shoup product. The K scan takes 0.34 ms cold, 0.25
    warm (b's huge page faulted), 0.25 with its stores into an L1-sized buffer: the transposes
    and stores cost about 0.12 ms, memory nothing measurable.
  - Lost or no gain (whole process, interleaved, 15 rounds, 3 slowest cases): a lane restart
    branch inside the unrolled steps kept x and g on the stack (929 `vmovdqa` per block of the K
    scan, 234 after moving the restart block apart): 14.29 vs 14.31 ms, ratio 0.9997; 16 steps
    per block (a full line per lane) 0.9965; software prefetch of the next block's lines: no
    change in the phases. Kept the 8-step version without branches in the unrolled steps
    (0.9931, noise).
  - Kept (`main.cpp`): 14.69 ms vs floor 5.12 (21 rounds, busier VM; 14.25-14.33 in quieter
    runs).
  - Checks: 28/28 official tests (`judge.py test`, `lc-amd` and `lc-intel`); `stress.py` 400
    (`lc-amd`) and 500 rounds (`lc-intel`): N <= 2000 against `brute.cpp`, larger at 8 points
    by Horner's rule; r of order exactly N, r in {0, 1, 2, P - 1}, sizes 0-8, near powers of
    two, 31-65, 512 (2m + 1) +- 2 (no partial lane), up to 2^19; ASan/UBSan on all 28 official
    cases, file and pipe input.
  - Rebased onto #230 (`lib/ntt/product.hpp`, convolution_mod's transform as `ntt::Product`):
    the forward radix-8 pass and the bottom kernels now come from `lib/ntt`; 28/28 official
    tests (`lc-amd`, `lc-intel`), stress 500 rounds (`lc-intel`), ASan/UBSan as above.
  - Merged as #239 (new problem: CI checks only).
  - No gain after #239 (whole process, interleaved, 21 rounds, 3 slowest cases, against #239):
    - No totals scan: the 1/Q scan starts each lane at 1 and yields the lane products; the K
      scan then fixes Y by each lane's 1/Q at its end. Totals -0.13 ms, K scan +0.10 (load,
      Shoup product, store of Y): ratio 0.9993.
    - Second product in a[L/2, 3L/2) instead of a third buffer: `product_kernels::inverse_top`
      leaves E = (u + w) s in a's lower half, so D can go to the upper half and the scans read
      one stream. In process -0.1 ms (D scan 0.51 -> 0.38, c scan 0.42 -> 0.37, one huge page
      fewer; `inverse_top` +0.08 per product against the two separate top groups); whole
      process ratio 1.0034.
  - Submitted the merged `main.cpp` twice (2 of 5 this session):
    [409509](https://judge.yosupo.jp/submission/409509) AC 21 ms, from launch spikes on
    pow_rN_equal_1_01 (21 ms; 13 in 409510) and small_01 (10); `spikes.py`: clean 15, and
    [409510](https://judge.yosupo.jp/submission/409510) AC 15 ms, clean: max_random_00/01 15,
    max_random_02 14, random_01, y0_00 and pow_rN_equal_1_01 13, the rest at most 12.
- Next: the products are 6.9 of the 9.6 ms above the floor and are `ntt::Product`'s; the local
  `Subtrees` copy goes once `lib/ntt` offers its option (backlog line in #95). Storing K's
  bottom-level leaves would skip their recomputation in the second product (guess: 0.2 ms;
  needs a bottom kernel variant). Fusing the radix-8 passes into the scans needs C = L / 64 and
  saves at most their non-fault part (~0.2 ms in all). Five scans of one chain each are the
  minimum for this form (four output sequences, and the lane totals for their starts).

## Sources

- Interpolation at a geometric progression in the Newton basis with q-analogues, two products:
  A. Bostan, É. Schost, "Polynomial evaluation and interpolation on special sets of points",
  J. Complexity 21 (2005) (the idea; formulas derived here, no code read). q-binomial theorem:
  G. Gasper, M. Rahman, "Basic Hypergeometric Series", ch. 1 (standard).
- Transform: our `lib/ntt` (`product.hpp`; its `Subtrees` copied with an option to reuse b).
- Montgomery reduction: P. Montgomery, "Modular multiplication without trial division",
  Math. Comp. 44 (1985).
