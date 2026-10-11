#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

MAX = [262143, 131073, 200000]  # the maximum; just above 2^17 (sparse top transform); a middle size

if sys.argv[1] == '--max-cases':
    print(len(MAX))
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
if kind == 'small':
    if seed < 70:
        print(seed + 1)  # every n up to 70: leaves, the schoolbook threshold, first transforms
    else:
        print(rng.randint(1, 3000))
else:
    print(MAX[seed])
