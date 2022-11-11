from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from sklearn.pipeline import Pipeline
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


class MyTransformedTargetClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, classifier, transformer):
        self.classifier = classifier
        self.transformer = transformer

    def predict(self, X, **predict_params):
        y = self.classifier.predict(X, **predict_params)

        y = self.__reshape_target(y)
        y = self.transformer.inverse_transform(y)

        return y.squeeze()

    def __reshape_target(self, y):
        if isinstance(self.transformer, LabelEncoder) == False:
            y = y.reshape(-1, 1) if y.ndim == 1 else y

        return y

    def transform_target(self, y):
        y = y.values
        y = self.__reshape_target(y)
        return self.transformer.transform(y)

    @staticmethod
    def load(path):
        preprocessor = pickle.load(open(f'{path}/preprocessor.pkl', 'rb'))

        model = load_model(f'{path}/model.h5')

        transformer = pickle.load(
            open(f'{path}/transformer.pkl', 'rb'))

        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

        return MyTransformedTargetClassifier(classifier=pipe, transformer=transformer)
