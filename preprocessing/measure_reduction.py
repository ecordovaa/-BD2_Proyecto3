from glob_consts import pd, file_list, random, np

COMPARE_FACTOR = 5000
tolerance = 0.6

# :: Equivalente a face_recognition.face_distance, solo que esta hará el cálculo sin validar
#    que los vectores sean 128D
def local_face_distance(face1, face2):
    return np.linalg.norm(face1 - face2)

# :: Toma N imagenes con rostros, las compara con los vectores de ambos csv e indica cuando
#    la reducción no reflejo la realidad (falso negativo).
def measurer():
    raw_vectors = pd.read_csv("data/raw.csv")
    reduced_vectors = pd.read_csv("data/reduced.csv")
    df_size = len(raw_vectors) - 1
    err_count = 0

    for i in range (COMPARE_FACTOR):
        face_idxs = [random.randrange(df_size), random.randrange(df_size)]
        print(f"[MEASURER] Comparation {i}: {file_list[face_idxs[0]]} vs. {file_list[face_idxs[1]]}")
        raw_rand_faces = [raw_vectors.iloc[face, 1:] for face in face_idxs]
        raw_distance = local_face_distance(raw_rand_faces[0], raw_rand_faces[1])
        reduced_rand_faces = [reduced_vectors.iloc[face, 1:] for face in face_idxs]
        reduced_distance = local_face_distance(reduced_rand_faces[0], reduced_rand_faces[1])
        if (raw_distance - tolerance) * (reduced_distance - tolerance) < 0:
            print(f"***[MEASURER] FALSE NEGATIVE: According to face_recognition, these faces are similar, but reduced vectors do not say that.")
            err_count += 1
    print(f"***[MEASURER] Done {COMPARE_FACTOR} comparations, {err_count} false negatives ({round(err_count/COMPARE_FACTOR * 100,3)}%).")
    print(f"***[MEASURER] Dataset has 99.38% of accuracy, so reduced dataset has {round(99.38 - 99.38*err_count/COMPARE_FACTOR, 2)}%.")

if __name__ == "__main__":
    measurer()