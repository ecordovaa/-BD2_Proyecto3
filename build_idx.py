from rtree import index
import pandas as pd
import timeit
import numpy as np
from preprocessing.raw_writter import DIR_PATH

df = pd.read_csv("data/reduced.csv")

# Propiedades del RTree
p = index.Property()
p.dimension = (df.shape[1] - 1)
p.dat_extension = 'dat'
p.idx_extension = 'idx'
idx = index.Index('index/rtree',properties=p)

def build_index():
    start = timeit.default_timer()
    row_id = 0
    for _, row in df.iterrows():
        print(f"[BUILD INDEX] Proccesing: {DIR_PATH}{row[0]}")
        vector = list(row[1:])
        idx.insert(row_id, np.concatenate([vector, vector]))
        row_id += 1
    stop = timeit.default_timer()
    print(f"[BUILD INDEX] Built in {stop - start}s.")

if __name__ == "__main__":
    build_index()