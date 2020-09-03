# -*- coding: utf-8 -*-

""" A model for predicting Boston house prices (part of the ML2P tutorial).
"""
import json
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
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = LinearRegression().fit(X_train, y_train)
        encoded_model = jsonpickle.encode(model)
        with (self.env.model_folder() / "boston-model.json").open("w") as f:
            f.write(encoded_model)


class BostonPredictor(ModelPredictor):
    def setup(self):
        """ Load the model. """

        with (self.env.model_folder() / "boston-model.json").open("r") as f:
            self.model = jsonpickle.decode(f.read())

    def result(self, data):
        """ Perform a prediction on the given data and return the result.

            :param dict data:
                The data to perform the prediction on.

            :returns dict:
                The result of the prediction.
        """
        result = self.model.predict(data)
        return result


class BostonModel(Model):

    TRAINER = BostonTrainer
    PREDICTOR = BostonPredictor
