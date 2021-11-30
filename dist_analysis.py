from glob_consts import random, df, np
from preprocessing.measure_reduction import local_face_distance
import matplotlib.pyplot as plt

N = 15000

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