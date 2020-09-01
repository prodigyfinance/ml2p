# -*- coding: utf-8 -*-

""" A model for predicting Boston house prices (part of the ML2P tutorial).
"""

from ml2p.core import Model, ModelPredictor, ModelTrainer


class BostonTrainer(ModelTrainer):
    def train(self):
        """ Train the model. """


class BostonPredictor(ModelPredictor):
    def setup(self):
        """ Load the model. """

    def result(self, data):
        """ Perform a prediction on the given data and return the result.

            :param dict data:
                The data to perform the prediction on.

            :returns dict:
                The result of the prediction.
        """


class BostonModel(Model):

    TRAINER = BostonTrainer
    PREDICTOR = BostonPredictor
