import random
import face_recognition as FR
from sklearn.decomposition import PCA
from build_idxs import df
import numpy as np
from preprocessing.measure_reduction import COMPARE_FACTOR, local_face_distance
import matplotlib.pyplot as plt

N = 15000

def encode_file(file):
    rep = FR.load_image_file(f"fotos_query/{file}")
    enc = FR.face_encodings(rep)
    try:
        face = enc[0]
        pca = PCA(n_components=df.shape[1])
        pca.fit(face)
        pca.transform(face)
        return face
    except IndexError:
        print("[ENCODE FILE] Error: Not face found on the provided image.")

def distances_analysis():
    dist = []
    for i in range(N):
        enc1 = df.iloc[random.randrange(len(df)), 1:]
        enc2 = df.iloc[random.randrange(len(df)), 1:]
        dist.append(local_face_distance(enc1, enc2))
        print(f"[DISTANCES ANALYSIS] Done {i+1}th comparation.")
    quantiles = []
    npdist = np.array(dist)
    [quantiles.append(round(np.quantile(npdist, q),4)) for q in [0.25,0.5,0.75]]
    print(f"[DISTANCES ANALYSIS] The quantiles are: {quantiles}")
    plt.hist(dist) 
    plt.title(f"Distance histogram ({len(df)} x {len(df)})")
    plt.show()

if __name__ == "__main__":
    distances_analysis()