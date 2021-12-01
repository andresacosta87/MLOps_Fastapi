import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import joblib
from pathlib import Path

class Model:
    def __init__(self, model_path: str = None):
        self.model = None
        self._model_path = model_path 
        self.load()

    def train(self, X: np.ndarray, y: np.ndarray):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

        self._model = PolynomialFeatures(2)
        self._model.fit(x_train, y_train)
        return self 

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self):
        if self._model is not None:
            joblib.dump(self._model, self._model_path)
        else:
            raise TypeError("Train the model")

    def load(self):
        try:
            self._model = joblib.load(self._model_path)

        except:
            self._model = None
        return self


model_path =Path(__file__).parent / "model.joblib"
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('/Users/andres/Documents/MLOPS_course/MLOps_Fastapi/housing.csv', header=None, delimiter=r"\s+", names=column_names)
n_features =data[['LSTAT', 'RM']]
n_features = np.array(n_features).shape[1]
model = Model(model_path) #invoca la clase
print(n_features)

def get_model():
    return model_path

if __name__ == '__main__':
    X = data[['LSTAT', 'RM']]
    y = data['MEDV']
    model.train(X,y)
    model.save()
