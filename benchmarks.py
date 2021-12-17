from glob_consts import index, np, timeit, encode_file
from build_idxs import sizes
from search_algorithms import rtree_knn_search, knn_search, range_search
from build_idxs import p
from prettytable import PrettyTable

if __name__ == "__main__":
    pivot = encode_file("Aaron_Eckhart_0001.jpg")
    iterations = 4
    
    knn_table = PrettyTable()
    knn_table.field_names = ["N", "Are Equal", "KNN-Rtree(s)", f"KNN-Secuencial(s), {iterations} iterations", "Mean Time (s)"]
    for size in sizes:
        idx = index.Index(f'indexes/rtree_{size}',properties=p)
        start = timeit.default_timer()
        rtree_data = rtree_knn_search(idx, pivot)
        rtree_time = f"{timeit.default_timer() - start:.6f}"
        seq_times = []
        for _ in range(iterations):
            seq_data = []
            start = timeit.default_timer()
            seq_data = knn_search(size, pivot)
            seq_time = round(timeit.default_timer() - start,6)
            seq_times.append("%.6f" % seq_time)
        are_equal = np.array_equal(rtree_data, seq_data)
        seq_times = [float(t) for t in seq_times]
        seq_mean = f"{np.array(seq_times).mean():.6f}"
        knn_table.add_row([size,are_equal,rtree_time,seq_times,seq_mean])
    print(knn_table)

    range_table = PrettyTable()
    range_table.field_names = ["Radio", "Retrieved", f"Time(s), {iterations} iterations", "Mean Time (s)"]
    for q in [0.6, 0.7275, 0.792, 0.85496]:
        idx = index.Index('indexes/rtree_complete',properties=p)
        times = []
        for _ in range(iterations):
            start = timeit.default_timer()
            range_data = range_search(idx, pivot, q)
            iter_time = round(timeit.default_timer() - start,6)
            times.append("%.6f" % iter_time)
        times = [float(time) for time in times]
        range_mean = f"{np.array(times).mean():.7f}"
        range_table.add_row([f"{q:.5f}", len(range_data), times, range_mean])
    print(range_table)