from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd

from models.boston_model import BostonModel
from models.my_transformed_target_regressor import MyTransformedTargetRegressor


def init():
    global app
    app = FastAPI()

    global boston_sklearn_model
    boston_sklearn_model = pickle.load(
        open('FastAPI_REST/estimators/0101_sklearn_model.pkl', 'rb'))

    global boston_keras_model
    boston_keras_model = MyTransformedTargetRegressor.load(
        'FastAPI_REST/estimators/0101_keras_model')


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


# -------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
