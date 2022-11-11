from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from sklearn.pipeline import Pipeline
from keras.models import load_model


class MyTransformedTargetRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, regressor, transformer):
        self.regressor = regressor
        self.transformer = transformer

    def predict(self, X, **predict_params):
        y = self.regressor.predict(X, **predict_params)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        y = self.transformer.inverse_transform(y)

        return y.squeeze()

    @staticmethod
    def load(path):
        preprocessor = pickle.load(open(f'{path}/preprocessor.pkl', 'rb'))

        model = load_model(f'{path}/model.h5')

        transformer = pickle.load(
            open(f'{path}/transformer.pkl', 'rb'))

        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

        return MyTransformedTargetRegressor(regressor=pipe, transformer=transformer)
