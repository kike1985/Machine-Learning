from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from keras.models import load_model


class MyKerasRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, model, target_transformer):
        self.preprocessor = preprocessor
        self.model = model
        self.target_transformer = target_transformer

    def predict(self, X, **predict_params):
        X_val = self.preprocessor.transform(X)
        y_val = self.model.predict(X_val, **predict_params)

        y_val = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        y_val = self.target_transformer.inverse_transform(y_val)

        return y_val.squeeze()

    @staticmethod
    def load(path):
        preprocessor = pickle.load(open(f'{path}/preprocessor.pkl', 'rb'))

        model = load_model(f'{path}/keras.h5')

        target_transformer = pickle.load(
            open(f'{path}/target_transformer.pkl', 'rb'))

        return MyKerasRegressor(preprocessor, model, target_transformer)
