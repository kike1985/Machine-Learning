from sklearn.base import BaseEstimator, TransformerMixin


class MySklearnRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, model, target_transformer):
        self.preprocessor = preprocessor
        self.model = model
        self.target_transformer = target_transformer

    def predict(self, X, **predict_params):
        X = self.preprocessor.transform(X)
        y = self.model.predict(X, **predict_params)

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        y = self.target_transformer.inverse_transform(y)

        return y.squeeze()
