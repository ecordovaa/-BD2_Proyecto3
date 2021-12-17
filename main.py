from flask import Flask, render_template, request, redirect
from search_algorithms import *
from glob_consts import encode_file, df, DIR_PATH
from build_idxs import index, p
import pandas as pd

df = pd.read_csv("data/reduced.csv")
app = Flask(__name__)
app.secret_key = ".."


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/knnsearch', methods=['POST'])
def show_k_best():
    image_name = request.form['query']
    k_results = int(request.form['range_n'])
    array_of_words = image_name.lower().split(' ')
    name_file = open("data/names.txt", 'r')
    line = name_file.readline()
    lowered_version = line.lower()
    probable_coincidence = ""
    number_of_coincidences = 0
    max_number_of_coincidences = 0
    while line:
        words_in_line = lowered_version.split("_")
        for query in array_of_words:
            if (query in words_in_line):
                number_of_coincidences = number_of_coincidences + 1
                if number_of_coincidences > max_number_of_coincidences:
                    probable_coincidence = line
                    max_number_of_coincidences = number_of_coincidences
        number_of_coincidences = 0
        line = name_file.readline()
        lowered_version = line.lower()
    name_file.close()
    if not line:
        redirect('/')
    img_encoding = encode_file(probable_coincidence.rstrip())
    if (img_encoding):
        names = []
        results = knn_search(len(df), img_encoding, k_results)
        for code in results:
            names.append(DIR_PATH + df.iloc[code, 0])
        return render_template("k_best.html", lista_k_mejores=names)
    else:
        redirect('/')


@app.route('/rangesearch', methods=['POST'])
def show_range():
    image_name = request.form['query']
    array_of_words = image_name.lower().split(' ')
    k_results = float(request.form['range_n'])
    name_file = open("data/names.txt", 'r')
    line = name_file.readline()
    lowered_version = line.lower()
    probable_coincidence = ""
    number_of_coincidences = 0
    max_number_of_coincidences = 0
    while line:
        words_in_line = lowered_version.split("_")
        for query in array_of_words:
            if (query in words_in_line):
                number_of_coincidences = number_of_coincidences + 1
                if number_of_coincidences > max_number_of_coincidences:
                    probable_coincidence = line
                    max_number_of_coincidences = number_of_coincidences
        number_of_coincidences = 0
        line = name_file.readline()
        lowered_version = line.lower()
    name_file.close()
    if not line:
        redirect('/')
    print(probable_coincidence)
    img_encoding = encode_file(probable_coincidence.rstrip())
    if (img_encoding):
        names = []
        index_after = index.Index('indexes/rtree_complete',properties=p)
        results = range_search(index_after, img_encoding, k_results)
        for code in results:
            names.append(DIR_PATH + df.iloc[code, 0])
        return render_template("k_best.html", lista_k_mejores=names)
    else:
        redirect('/')

if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)