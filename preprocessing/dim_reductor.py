from raw_writter import *
from sklearn.decomposition import PCA

# @Input: La varianza del dataset. PCA determina un número de componentes de forma que
#         esta varianza actua como una cota máxima de error.
# @Output: Un csv con los vectores reducidos de acuerdo a la varianza de input
# :: Por defecto usamos una varianza de 0.93. Comprobaremos experimentalmente que tan 
#    acertado es este coeficiente (analísis de falso negativo). 
def dim_reductor(variance = .93):
    start = timeit.default_timer()
    df = pd.read_csv("data/raw.csv")
    names = list(df.iloc[:, 0])
    df.drop(["person"], axis="columns", inplace=True)

    # 0 < n_components < 1 y svd_solver='full' produce que PCA haga la estimación de 
    # componentes en base a la varianza que representa n_components
    pca = PCA(n_components=variance, svd_solver='full')
    pca.fit(df)
    reduced_vectors = pca.transform(df)
    dim_factor = reduced_vectors.shape[1]
    print(f"[DIM. REDUCTOR] Reduced dimensions to {dim_factor}D")
    print(f"[DIM. REDUCTOR] Significance vector: \n{pca.explained_variance_ratio_}")

    # Se construye el dataframe y se escribe en disco
    col_vector, row_vector = ["person"], []
    [col_vector.append(str(i+1)) for i in range(dim_factor)]
    [row_vector.append([p] + list(v)) for p, v in zip(names, reduced_vectors)]
    reduced_df = pd.DataFrame(row_vector, columns=col_vector)
    reduced_df.to_csv("data/reduced.csv", index=False)
    stop = timeit.default_timer()
    print(f"[DIM. REDUCTOR] All reductions were written correctly in {round(stop - start,5)}s")

if __name__ == "__main__":
    dim_reductor()