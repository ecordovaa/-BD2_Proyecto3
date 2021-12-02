import sys
sys.path.append('../BD2_Proyecto3')
from glob_consts import FR, pd, timeit, DIR_PATH
import os

# @Return: Un csv con la representación dada por la líbreria de face_recognition
#          (128D) para cada una de las imagenes en la carpeta referenciada.
def write_raw_vectors():
    start = timeit.default_timer()
    col_vector = ["person"]
    [col_vector.append(str(d+1)) for d in range(128)]
    df = pd.DataFrame(columns=col_vector)
    row_id = 0
    for file in sorted(os.listdir(DIR_PATH)):
        rep_img = FR.load_image_file(f"{DIR_PATH}{file}")
        rep_enc = FR.face_encodings(rep_img)
        if len(rep_enc) > 0:
            df.loc[row_id] = [file] + list(rep_enc[0])
            row_id += 1
        print(f"[RAW WRITTER] Proccesed file: {DIR_PATH}{file}")
    df.to_csv("data/raw.csv", index=False)
    stop = timeit.default_timer()
    print(f"[RAW WRITTER] All encoding vectors were written in {round(stop - start,5)}s")

if __name__ == "__main__":
    write_raw_vectors()