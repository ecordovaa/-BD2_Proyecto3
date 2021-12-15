from flask import Flask, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from search_algorithms import *
import pandas as pd

df = pd.read_csv("data/reduced.csv")
app = Flask(__name__)
app.secret_key = ".."


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/search', methods=['GET'])
def show_k_best(codigo, k):
    return render_template("k_best.html", lista_k_mejores=knn_search(len(df), codigo, k))


if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)
