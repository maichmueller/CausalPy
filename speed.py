import numpy as np
from time import perf_counter
import bisect

if __name__ == '__main__':
    rand_lookup = [np.random.randint(0, 100000) for i in range(10000)]
    l = [np.random.randint(0, int(10e7)) for i in range(100000)]
    hash_l = [hash(k) for k in l]
    assert(all([l_val == hash_l_val for l_val, hash_l_val in zip(l, hash_l)]))
    s = set(l)
    d = {d: True for d in l}
    start = perf_counter()
    for i in rand_lookup:
        k = i in l
    duration = perf_counter() - start
    print(f"Time for List: {duration:.5f} s")
    start = perf_counter()
    for i in rand_lookup:
        k = i in s
    duration = perf_counter() - start
    print(f"Time for Set: {duration:.5f} s")
    start = perf_counter()
    for i in rand_lookup:
        k = i in d
    duration = perf_counter() - start
    print(f"Time for Dictionary: {duration:.5f} s")