from face_recognition.api import face_distance
import pandas as pd
import face_recognition as FR
import numpy as np
from rtree import index
from sklearn.decomposition import PCA
import os
import random
import timeit

DIR_PATH = "LFW_images/"
df = pd.read_csv("data/reduced.csv")
file_list = sorted(os.listdir(DIR_PATH))
sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800]

def encode_file(file):
    try:
        face = df.loc[df["person"] == file].values.tolist()
        return face[0][1:]
    except KeyError:
        print(f"[ENCODE FILE] Error: Not found {file} in {DIR_PATH}. Returning None object.")
        return None