#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

if sys.argv[1] == '--max-cases':
    print(2)
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
if kind == 'small':
    print(rng.randint(1, 3000) if seed % 4 else rng.randint(1, 10))
else:
    print([10**7, 8388609][seed])  # the maximum; just above 2^23 (largest transforms)
