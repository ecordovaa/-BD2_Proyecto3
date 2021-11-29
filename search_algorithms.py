import heapq
from heapq import *
import numpy as np
from preprocessing.measure_reduction import local_face_distance
import pandas as pd
heapq.__lt__ = lambda x,y: x[0] < y[0]

df = pd.read_csv("data/reduced.csv")

def rtree_knn_search(idx, img_enc, k=8):
    return list(idx.nearest(np.concatenate([img_enc, img_enc]), num_results=k))

def knn_search(size, img_enc, k=8):
    h = []
    for row_id, row in df.iterrows():
        if row_id == size:
            break
        heappush(h, (float(local_face_distance(img_enc, list(row)[1:])), row_id))
    return [heappop(h)[1] for _ in range(k)]

def range_search(idx, img_enc, radio):
    reshaped_enc = np.array(img_enc).reshape(1,-1)
    left_bound = reshaped_enc - radio
    right_bound = reshaped_enc + radio
    bbox = np.concatenate((left_bound, right_bound), axis=1)
    # Candidatos: Puntos que pertenecen a una region que intereseca el rango (1er filtro).
    candidates = list(idx.intersection(tuple(bbox[0]), objects=True))
    distances = []
    for c in candidates:
        distances.append((c.id, local_face_distance(list(c.bbox)[:df.shape[1]], reshaped_enc)))
    # Puntos en el rango: candidatos que se encuentran en el rango (2do filtro).
    in_range_points = [(id, dist) for (id,dist) in distances if dist <= radio]
    in_range_points.sort(key=lambda x:x[1])
    return [point[0] for point in in_range_points]