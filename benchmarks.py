from prettytable import PrettyTable
from build_idxs import sizes, df
from rtree import index
import timeit
import random
import numpy as np
from search_algorithms import *

# Propiedades del RTree
p = index.Property()
p.dimension = (df.shape[1] - 1)
p.dat_extension = 'dat'
p.idx_extension = 'idx'

if __name__ == "__main__":
    t = PrettyTable()
    t.field_names = ["N", "Are Equal", "KNN-Rtree(s)", "KNN-Secuencial(s)"]
    pivot = df.iloc[random.randrange(len(df) - 1),1:]
    for size in sizes:
        idx = index.Index(f'indexes/rtree_{size}',properties=p)
        start = timeit.default_timer()
        rtree_data = rtree_knn_search(idx, pivot)
        rtree_time = timeit.default_timer() - start
        start = timeit.default_timer()
        seq_data = knn_search(size, pivot)
        seq_time = timeit.default_timer() - start
        are_equal = np.array_equal(rtree_data, seq_data)
        t.add_row([size,are_equal,round(rtree_time,10),round(seq_time,10)])
    print(t)