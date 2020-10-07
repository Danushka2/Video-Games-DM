# Importing
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from dataset.cleanDs import cleanDs


app = Flask(__name__)
model = pickle.load(open('testModel.pkl', 'rb'))

cleanData = cleanDs()
df = cleanData.clean_db()

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


@app.route('/p-sales-new')
def predicSalesNew():

    model = pickle.load(open('DecisionTreeRegression.pkl','rb'))

    df[['Year_of_Release','Platform','Genre','Publisher','Developer']].iloc[0]

    new_row = {'Year_of_Release':1985, 'Platform':'NES', 'Genre':'Platform', 'Publisher':'Nintendo', 'Developer':'Open Source'}
    df_Apnd = df.append(new_row, ignore_index=True)

    df_Apnd[['Year_of_Release','Platform','Genre','Publisher','Developer']].iloc[-1]

    ##################  Get features and labels     ######################
    newDF2 = df_Apnd[['Year_of_Release','Platform','Genre','Publisher','Developer']]
    M = newDF2
    N = df_Apnd.iloc[:,5].values

    ###################  Encoding #########################
    objList2 = M.select_dtypes(include = "object").columns
    le = LabelEncoder()

    for feat2 in objList2:
        M[feat2] = le.fit_transform(M[feat2].astype(str))


    l = M.iloc[-1].values

    #print(model.predict(k.reshape(1,-1)))
    output = model.predict(l.reshape(1,-1))[0]
    
    return render_template('sale-prediction.html', rVal = output)





if __name__ == "__main__":
    app.run(debug=True)
    