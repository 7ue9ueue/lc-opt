# Judge spikes

About 5% of judged cases take a fixed +9 ms more. The program does not cause it; a resubmission
re-rolls it. `tools/spikes.py ID` flags spiked cases and prints the clean score.

## Data

2026-10-09, claude. All 108 AC submissions by Aiyiyi (2026-09-27 to 2026-10-09), 4014 cases in
classes of at least 3 (same name without `_NN`). Residual = time - class median.

- Bimodal: 3523 cases within ±1 ms, 210 at +6.5 to +13 ms (+8: 69, +9: 79, +10: 31), only 20 at
  +4 to +6. Rate 5.2%.
- Size 9.0 ms whatever the case: median 9.0 (sd 0.8) on cases under 3 ms, 9.0 (sd 1.2) on 3-30 ms,
  9.2 (sd 1.6) on 30+ ms. A 0-1 ms case becomes 9-11 ms.
- Rate does not depend on runtime (4.6% under 1 ms, 5.5% at 1-3 and 10-30 ms, 6.1% over 100 ms;
  2.9% and 3.3% in the two small buckets, 3-10 and 30-100 ms, n = 242 and 92),
  memory (4.5-6.1% from 1 to 128+ MiB), position in the run (4.7-6.1% by fifth), or date.
- Independent: spikes per submission follow Poisson (0: 15 vs 18.6 expected, 1: 35 vs 28.7,
  2: 24 vs 26.5, 3: 17 vs 18.0, 4: 11 vs 9.6); adjacent spiked pairs 13 vs 11 expected.

So it is one fixed delay per process launch, not a slowdown during the run.

## Cause (partly confirmed)

The judge (`yosupo06/library-checker-judge`, `executor/execute.go`) times a case as wall time
between the first and last tick of a 1 ms Go ticker at which the container's `cgroup.procs` lists
at least 2 processes (`docker create --init`: docker-init plus the program). The ticker starts
before `docker start`. Any moment during container setup with 2 processes in the cgroup starts
the clock early, by the setup-to-program gap. That fits every property above: a fixed size,
once per launch, probability = (length of that moment) / 1 ms.

Checked on `lc-amd` (Docker 29.1.3, runc 1.3.4, systemd cgroups), 300 launches traced by a
busy-polling reader of `cgroup.procs`: runc's bootstrap (`runc:[0:PARENT]`, `runc:[1:CHILD]`) puts
2-3 processes in the cgroup for 1.11 ms (median), 44.7 ms before the program starts. With the
judge's monitor that adds ~47 ms to every case, so this setup shows the mechanism, not the judge's
numbers.

The judge runs crun 1.15 with cgroupfs on Ubuntu 22.04 (`packer/base/docker-daemon.json`).
Standalone `crun` 1.15 create + start (no Docker) never showed a second process during setup.
Not tested: Docker + crun (needs a daemon config change on a shared VM; denied). For the
mechanism to give 5.2% and +9 ms, the judge's 2-process moment would last ~52 us, 9 ms before
the program starts (guess).

## Consequences

- Score = max over cases, so a spike on any case within 9 ms of the slowest raises the score.
  P(clean run) = 0.948^k for k such cases. convolution_mod_1000000007 has k = 23: P = 0.29, and
  both submissions spiked (409262: 31 ms, 409263: 32 ms; clean 23 for both).
- No code change helps. Count a judged maximum as a spike only if `spikes.py` flags it.
