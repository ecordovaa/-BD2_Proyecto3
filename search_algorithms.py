from glob_consts import np, df
import heapq
from heapq import *
from preprocessing.measure_reduction import local_face_distance
heapq.__lt__ = lambda x,y: x[0] < y[0]

def rtree_knn_search(idx, img_enc, k=8):
    return list(idx.nearest(np.concatenate([img_enc, img_enc]), num_results=k))

def knn_search(size, img_enc, k=8):
    h = []
    for row_id, row in df.iterrows():
        if row_id == size:
            break
        dist = local_face_distance(np.array(img_enc), np.array(list(row)[1:]))
        if row_id < k:
            heappush(h, (float(-dist), row_id))
        else:
            heappushpop(h, (float(-dist), row_id))
    knn = [heappop(h)[1] for _ in range(k)]
    knn.reverse()
    return knn

def range_search(idx, img_enc, radio):
    reshaped_enc = np.array(img_enc).reshape(1,-1)
    left_bound = reshaped_enc - radio
    right_bound = reshaped_enc + radio
    bbox = np.concatenate((left_bound, right_bound), axis=1)
    candidates = list(idx.intersection(tuple(bbox[0]), objects=True))
    distances = []
    for c in candidates:
        distances.append((c.id, local_face_distance(list(c.bbox)[:df.shape[1]-1], reshaped_enc)))
    in_range_points = [(id, dist) for (id,dist) in distances if dist <= radio]
    in_range_points.sort(key=lambda x:x[1])
    return [point[0] for point in in_range_points]