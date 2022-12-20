from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

from models.my_transformed_target_regressor import MyTransformedTargetRegressor
from models.my_transformed_target_classifier import MyTransformedTargetClassifier
from models.boston_model import BostonModel
from models.vehicle_model import VehicleModel
from models.iris_model import IrisModel


def init():
    global app
    app = FastAPI()

    global boston_sklearn_model
    boston_sklearn_model = pickle.load(
        open('FastAPI_REST/estimators/0101_sklearn_model.pkl', 'rb'))

    global boston_keras_model
    boston_keras_model = MyTransformedTargetRegressor.load(
        'FastAPI_REST/estimators/0101_keras_model')

    global vehicle_sklearn_model
    vehicle_sklearn_model = pickle.load(
        open('FastAPI_REST/estimators/0103_sklearn_model.pkl', 'rb'))

    global vehicle_keras_model
    vehicle_keras_model = MyTransformedTargetRegressor.load(
        'FastAPI_REST/estimators/0103_keras_model')

    global iris_sklearn_model
    iris_sklearn_model = pickle.load(
        open('FastAPI_REST/estimators/0201_sklearn_model.pkl', 'rb'))

    global iris_keras_model
    iris_keras_model = MyTransformedTargetClassifier.load(
        'FastAPI_REST/estimators/0201_keras_model')


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
