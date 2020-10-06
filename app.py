# Importing
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)
model = pickle.load(open('testModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', url ='/static/images/line_plot.jpg')

@app.route('/predict-sales')
def pSales():
    return render_template('sale-prediction.html')

@app.route('/p-sales',methods=['POST'])
def predicSales():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('sale-prediction.html', prediction_text='$ {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
    