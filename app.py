# Importing
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from dataset.cleanDs import cleanDs


import plotly
import plotly.graph_objs as go
import json
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('testModel.pkl', 'rb'))

cleanData = cleanDs()
df = cleanData.clean_db()

@app.route('/')
def home():
#    return render_template('index.html', url ='/static/images/line_plot.jpg')

    labels = df['Genre']
    values = df['Global_Sales']
    
    graphs = [
        dict(
            data=[
                dict(
                    x = df['Genre'],
                    y = df['Global_Sales'],
                    type='bar'
                ),
            ],
            layout=dict(
                title='Bar Chart'
            )
        ),

        dict(
            data = [go.Pie(
                    labels=labels, values=values, hole=.3
                )
            ],
            layout=dict(
                title='Pie Chart'
            )
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           graphJSON=graphJSON)


############################# Sales and User Score prediction ######################################
@app.route('/predict-sales')
def pSales():
    return render_template('sale-prediction.html')

@app.route('/p-sales-new', methods=['POST'])
def predicSalesNew():

    saleYear = int(request.form['saleYear'])
    salePlatform = request.form['salePlatform']
    saleGnre = request.form['saleGnre']
    salePublisher = request.form['salePublisher']
    saleDeveloper = request.form['saleDeveloper']
    
    model = pickle.load(open('DecisionTreeRegression.pkl','rb'))

    new_row = {'Year_of_Release':saleYear, 'Platform': salePlatform, 'Genre': saleGnre, 'Publisher': salePublisher, 'Developer': saleDeveloper}
    df_Apnd = df.append(new_row, ignore_index=True)


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
    print(output)
    
    return render_template('sale-prediction.html', rVal = output)

@app.route('/user-score-p', methods=['POST'])
def usPrediction():
    
    cScore = float(request.form['cScore'])
    gSales = float(request.form['gSales'])
    cCount = float(request.form['cCount'])
    

    model = pickle.load(open('MultipleLinearRegression.pkl','rb'))
    # predict using a value

    new_row = {'Critic_Score': cScore, 'Global_Sales': gSales, 'Critic_Count':cCount}
    df_Apnd = df.append(new_row, ignore_index=True)

    ##################  Get features and labels     ######################
    newDF2 = df_Apnd[['Critic_Score','Global_Sales','Critic_Count']]
    M = newDF2
    N = df_Apnd.iloc[:,5].values

    l = M.iloc[-1].values

    pred = model.predict(l.reshape(1,-1))[0]
    pred = round(pred, 2)
    
    return render_template('sale-prediction.html', rVal = gSales, score = pred)

############################# Genre prediction ######################################

@app.route('/genre-classify')
def cGenre():
    return render_template('genre-classify.html')

@app.route('/c-genre', methods=['POST'])
def genreClassify():

    model = pickle.load(open('GenreClassifier.pkl','rb'))

    gameName = request.form['gameName']
    gamePlatform = request.form['gamePlatform']
    gameYear = int(request.form['gameYear'])
    gamePublisher = request.form['gamePublisher']
    gameRating = request.form['gameRating']
    

#    new_row = {'Name': 'Call of Duty: Black Ops II','Platform':'PS3','Year_of_Release':2012, 'Publisher':'Activision', 'Rating':'E'}
    new_row = {'Name': gameName,'Platform': gamePlatform,'Year_of_Release': gameYear, 'Publisher': gamePublisher, 'Rating': gameRating}

    df_Apnd = df.append(new_row, ignore_index=True)

    df_Apnd[['Name','Platform','Year_of_Release','Publisher','Rating']].iloc[-1]

    newDF2 = df_Apnd[['Name','Platform','Year_of_Release','Publisher','Rating']]
    M = newDF2
    N = df_Apnd.iloc[:,5].values

    objList2 = ['Name','Platform','Year_of_Release','Publisher','Rating']
    le = LabelEncoder()

    for feat2 in objList2:
        M[feat2] = le.fit_transform(M[feat2].astype(str))

    l = M.iloc[-1].values

    pred = model.predict(l.reshape(-1, 5))[0]
    
    return render_template('genre-classify.html', genre = pred)


############################# User Score Clustering ######################################

@app.route('/uscore-clustering')
def uscoreCluster():
    return render_template('uscore-cluster.html')

@app.route('/us-cluster', methods=['POST'])
def usCluster():
    
    cScore = float(request.form['cScore'])
    gSales = float(request.form['gSales'])
    cCount = float(request.form['cCount'])
    

    model = pickle.load(open('ClusteringPredict.pkl','rb'))
    # predict using a value

    new_row = {'User_Score': cScore, 'Global_Sales': gSales, 'Critic_Score': cCount}
    df_Apnd = df.append(new_row, ignore_index=True)

    # Get features and labels
    newDF2 = df_Apnd[['User_Score','Global_Sales','Critic_Score']]
    M = newDF2
    N = df_Apnd.iloc[:,3].values

    # Encoding
    objList2 = M.select_dtypes(include = "object").columns
    le = LabelEncoder()

    for feat2 in objList2:
        M[feat2] = le.fit_transform(M[feat2].astype(str))


    l = M.iloc[-1].values

    clstr = model.predict(l.reshape(1,-1))
    if clstr == 0:
        pred = "Cluster A"
    elif clstr == 1:
        pred = "Cluster B"
    elif clstr == 2:
        pred = "Cluster C"
    else:
        pred = "Cluster D"

    
    return render_template('uscore-cluster.html', score = pred)









@app.route('/test')
def indexT():
#    rng = pd.date_range('1/1/2011', periods=7500, freq='H')
#    ts = pd.Series(np.random.randn(len(rng)), index=rng)

    graphs = [
        dict(
            # Create a trace
            data = [go.Scatter3d(
                x = df['Global_Sales'],
                y = df['User_Score'],
                z = df['Critic_Score'],
                mode = 'markers',marker = dict(
                  size = 12,
                  colorscale = 'Viridis'
                  )
                )]
            )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('test.html',
                           ids=ids,
                           graphJSON=graphJSON)











if __name__ == "__main__":
    app.run(debug=True)
    