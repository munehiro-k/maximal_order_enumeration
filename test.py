import time

import numpy as np

import maximal_order_enumeration as enum

rng = np.random.default_rng()


if __name__ == '__main__':
    n_alphabets = 5
    n_observation = 40
    shuffle = np.empty((n_observation, n_alphabets), dtype=int)
    shuffle[:] = np.array([list(range(n_alphabets))], dtype=int)
    rng.permuted(shuffle, axis=1, out=shuffle)
    sample = np.sum(rng.random(size=(n_observation, n_alphabets)) > 0.7,
                    axis=1)
    sample += 1

    tuple_list = []
    for i in range(n_observation):
        tuple_list.append(tuple(shuffle[i, :sample[i]]))

    enumerator = enum.EnumerateMaximal(tuple_list)
    print(f"input({len(enumerator.unique_name_seq_list)} orders):")
    print(enumerator.unique_name_seq_list)

    valid = True
    answer = None
    # for method in ("canny",):
    for method in ("naive", "canny", "clique"):
        start = time.perf_counter()
        maximal_list =\
            enumerator.enumerate_totally_ordered_maximal_subset(method=method)
        elapsed = time.perf_counter() - start
        print(f"{method}: {elapsed:.2f} (sec)")
        print(maximal_list)
        if answer is not None:
            valid = valid and (answer == maximal_list)
        else:
            answer = maximal_list

    print(f"obtained same result: {valid} ({len(maximal_list)} orders)")
