# -*- coding: utf-8 -*-

""" ML2PModel descriptor.
"""

from sagemaker.estimator import Estimator


class ML2PModel:
    """ ML2PModel descriptor.

        A model aims to provide a declarative description of the data,
        code and parameters needed to train a model and make predictions
        via Amazon SageMaker.

        The intention is to provide enough information to both:

        * call the SageMaker API, and
        * run the model inside a docker container.
    """

    def sagemake_estimator(self):
        """ Create a Python SageMaker SDK Estimator object for this model.

            This is run when setting up SageMaker tasks or services via the
            SageMaker API.

            :returns sagemaker.estimator.Estimator:
                A Python SageMaker Estimator model for this object.
        """
        return Estimator()

    def train(self, sgctxt):
        """ Perform a training run.

            This is run inside the SageMaker Docker container.

            :parameter SageMakerTrainContext sgtxt:
                A convenient way to access the SageMaker docker context in
                which this training run is happening (e.g. access data,
                parameters, etc).
        """
        raise NotImplementedError(
            "ML2PModel sub-classes should implement .train(...)")

    def predict(self, sgctxt):
        """ Perform a prediction.

            This is run inside the SageMaker Docker container.

            :parameter SageMakerPredictContext sgtxt:
                A convenient way to access the SageMaker docker context in
                which this prediction is happening (e.g. parameters, etc).
        """
        raise NotImplementedError(
            "ML2PModel sub-classes should implement .predict(...)")
