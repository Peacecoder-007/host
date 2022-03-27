import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import LogisticRegression
from flask import Flask, jsonify
from flask_restful import Resource,Api,reqparse

app = Flask(__name__)
api =Api(app)

api.add_resource()

@app.route('/', methods=['GET'])
def home():
    warnings.filterwarnings('ignore')
    dataset = pd.read_csv('diabetes.csv')
    Data = dataset.drop(['Outcome'], axis=1)
    Outcome = dataset.Outcome.values
    x_train, x_test, y_train, y_test = train_test_split(Data, Outcome, test_size=0.2,
                                                        random_state=1)
    regressor = LogisticRegression()
    regressor.fit(x_train, y_train)
    output = regressor.score(x_test, y_test)
    if output > 0.5:
        result = "yes"
    else:
        result = "no"
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)

