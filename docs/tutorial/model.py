# -*- coding: utf-8 -*-

""" A model for predicting Boston house prices (part of the ML2P tutorial).
"""

import jsonpickle

import pandas as pd
from ml2p.core import Model, ModelPredictor, ModelTrainer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class BostonTrainer(ModelTrainer):
    def train(self):
        """ Train the model. """
        training_channel = self.env.dataset_folder()
        training_csv = str(training_channel / "house-prices.csv")
        df = pd.read_csv(training_csv)
        y = df["target"]
        X = df.drop(columns="target")
        features = sorted(X.columns)
        X = X[features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        model = LinearRegression().fit(X_train, y_train)
        with (self.env.model_folder() / "boston-model.json").open("w") as f:
            f.write(jsonpickle.encode({"model": model, "features": features}))


class BostonPredictor(ModelPredictor):
    def setup(self):
        """ Load the model.
        """
        with (self.env.model_folder() / "boston-model.json").open("r") as f:
            data = jsonpickle.decode(f.read())
            self.model = data["model"]
            self.features = data["features"]

    def result(self, data):
        """ Perform a prediction on the given data and return the result.

            :param dict data:
                The data to perform the prediction on.

            :returns dict:
                The result of the prediction.
        """
        X = pd.DataFrame([data])
        X = X[self.features]
        price = self.model.predict(X)[0]
        return {"predicted_price": price}


class BostonModel(Model):

    TRAINER = BostonTrainer
    PREDICTOR = BostonPredictor
