from glob_consts import index, pd, timeit, np, DIR_PATH, df, sizes

# Propiedades del RTree
p = index.Property()
p.dimension = (df.shape[1] - 1)
p.dat_extension = 'dat'
p.idx_extension = 'idx'

# :: Construye un conjunto de índices de rtree, los cuales tienen diferentes tamaños
def build_indexes():
    idxs = [index.Index(f'indexes/rtree_complete',properties=p)]
    [idxs.append(index.Index(f'indexes/rtree_{size}',properties=p)) for size in list(reversed(sizes))]
    size_idx, max_size = 0, sizes[len(sizes) - 1]
    for row_id, row in df.iterrows():
        print(f"[BUILD INDEXES] Proccesing n°{row_id}: {DIR_PATH}{row[0]}.")
        if row_id <= max_size and row_id == sizes[size_idx]:
            size_idx += 1
            idxs.pop()
            print(f"***[BUILD INDEX-{row_id}] Built index.")
        vector = row[1:]
        [idx.insert(row_id, np.concatenate([vector, vector])) for idx in idxs]
    print(f"***[BUILD INDEX-COMPLETE] Built index.")

if __name__ == "__main__":
    start = timeit.default_timer()
    build_indexes()
    stop = timeit.default_timer()
    print(f"[BUILT ALL INDEXES IN {stop-start}s]")