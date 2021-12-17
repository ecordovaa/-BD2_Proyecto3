from face_recognition.api import face_distance
import pandas as pd
import face_recognition as FR
import numpy as np
from rtree import index
from sklearn.decomposition import PCA
import random
import timeit

DIR_PATH = "static/LFW_images/"
df = pd.read_csv("data/reduced.csv")

def encode_file(file):
    try:
        face = df.loc[df["person"] == file].values.tolist()
        return face[0][1:]
    except KeyError:
        print(f"[ENCODE FILE] Error: Not found {file} in {DIR_PATH} or image has zero faces. Returning None object.")
        return None