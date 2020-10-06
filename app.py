# Importing
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', url ='/static/images/line_plot.jpg')

@app.route('/predict-sales')
def pSales():
    return render_template('sale-prediction.html')



if __name__ == "__main__":
    app.run(debug=True)
    