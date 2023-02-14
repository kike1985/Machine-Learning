from fastapi import FastAPI
import uvicorn
import pickle
import dill
import pandas as pd
import numpy as np
from progressbar import ProgressBar

from resources.my_transformed_target_regressor import MyTransformedTargetRegressor
from resources.my_transformed_target_classifier import MyTransformedTargetClassifier

from resources._0101_Regression_Boston.boston_model import BostonModel
from resources._0103_Regression_Vehicles_Price.vehicle_model import VehicleModel
from resources._0201_Classification_Iris.iris_model import IrisModel

BostonOutlierTransformer = dill.load(
    open('resources/_0101_Regression_Boston/outlier_transformer', 'rb'))
BostonImputeTransformer = dill.load(
    open('resources/_0101_Regression_Boston/impute_transformer', 'rb'))
BostonDataTransformer = dill.load(
    open('resources/_0101_Regression_Boston/data_transformer', 'rb'))

VehiclesOutlierTransformer = dill.load(
    open('resources/_0103_Regression_Vehicles_Price/outlier_transformer', 'rb'))
VehiclesImputeTransformer = dill.load(
    open('resources/_0103_Regression_Vehicles_Price/impute_transformer', 'rb'))
VehiclesDataTransformer = dill.load(
    open('resources/_0103_Regression_Vehicles_Price/data_transformer', 'rb'))

IrisOutlierTransformer = dill.load(
    open('resources/_0201_Classification_Iris/outlier_transformer', 'rb'))
IrisImputeTransformer = dill.load(
    open('resources/_0201_Classification_Iris/impute_transformer', 'rb'))
IrisDataTransformer = dill.load(
    open('resources/_0201_Classification_Iris/data_transformer', 'rb'))


def init():
    global app
    app = FastAPI()

    global boston_sklearn_model
    boston_sklearn_model = pickle.load(
        open('resources/_0101_Regression_Boston/sklearn_model.pkl', 'rb'))

    global boston_keras_model
    boston_keras_model = MyTransformedTargetRegressor.load(
        'resources/_0101_Regression_Boston/keras_model')

    global vehicle_sklearn_model
    vehicle_sklearn_model = pickle.load(
        open('resources/_0103_Regression_Vehicles_Price/sklearn_model.pkl', 'rb'))

    global vehicle_keras_model
    vehicle_keras_model = MyTransformedTargetRegressor.load(
        'resources/_0103_Regression_Vehicles_Price/keras_model')

    global iris_sklearn_model
    iris_sklearn_model = pickle.load(
        open('resources/_0201_Classification_Iris/sklearn_model.pkl', 'rb'))

    global iris_keras_model
    iris_keras_model = MyTransformedTargetClassifier.load(
        'resources/_0201_Classification_Iris/keras_model')


init()


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API (/docs)'}


@app.post('/api/boston')
def boston(data: BostonModel):
    df = pd.DataFrame(data=data.dict(), index=[0])

    sklearn_pred = boston_sklearn_model.predict(X=df)
    keras_pred = boston_keras_model.predict(X=df)

    return {'sklearn_pred': sklearn_pred.item(), 'keras_pred': keras_pred.item()}


@app.post('/api/vehicle')
def vehicle(data: VehicleModel):
    df = pd.DataFrame(data=data.dict(), index=[0])

    sklearn_pred = vehicle_sklearn_model.predict(X=df)
    keras_pred = vehicle_keras_model.predict(X=df)

    return {'sklearn_pred': sklearn_pred.item(), 'keras_pred': keras_pred.item()}


@app.post('/api/iris')
def iris(data: IrisModel):
    df = pd.DataFrame(data=data.dict(), index=[0])

    sklearn_pred = iris_sklearn_model.predict(X=df)
    keras_pred = iris_keras_model.predict(X=df)

    return {'sklearn_pred': sklearn_pred.item(), 'keras_pred': keras_pred.item()}


# -------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
