from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd
from keras.models import load_model
from models.boston_model import BostonModel


def init():
    global app
    app = FastAPI()

    global boston_estimator
    boston_estimator = pickle.load(
        open('FastAPI_REST/estimators/0101_estimator.pkl', 'rb'))

    global boston_preprocessor_pickle
    boston_preprocessor_pickle = pickle.load(
        open(f'FastAPI_REST/estimators/0101_model/preprocessor.pkl', 'rb'))

    global boston_keras_model_h5
    boston_keras_model_h5 = load_model(
        f'FastAPI_REST/estimators/0101_model/keras.h5')

    global boston_target_transformer_pickle
    boston_target_transformer_pickle = pickle.load(
        open(f'FastAPI_REST/estimators/0101_model/target_transformer.pkl', 'rb'))


init()


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API (/docs)'}


@app.post('/api/ml_predict_boston_median_value')
def ml_predict_boston_median_value(data: BostonModel):
    df = pd.DataFrame(data=data.dict(), index=[0])
    pred = boston_estimator.predict(X=df)

    return {'prediction': pred[0]}


@app.post('/api/dl_predict_boston_median_value')
def dl_predict_boston_median_value(data: BostonModel):
    df = pd.DataFrame(data=data.dict(), index=[0])

    pred = boston_preprocessor_pickle.transform(df)
    pred = boston_keras_model_h5.predict(pred)
    pred = boston_target_transformer_pickle.inverse_transform(pred)

    return {'prediction': pred[0][0].item()}


# -------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
