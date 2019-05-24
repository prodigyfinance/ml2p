# -*- coding: utf-8 -*-

""" ML2P core utilities.
"""

import datetime
import importlib
import json
import os
import pathlib


class SageMakerEnv:
    """ An interface to the SageMaker docker environment. """

    def __init__(self, ml_folder):
        self._ml_folder = pathlib.Path(ml_folder)
        self.model_version = os.environ.get("ML2P_MODEL_VERSION", "Unknown")

    def hyperparameters(self):
        with (
            self._ml_folder / "input" / "config" / "hyperparameters.json"
        ).open() as f:
            return json.load(f)

    def resourceconfig(self):
        with (self._ml_folder / "input" / "config" / "resourceconfig.json").open() as f:
            return json.load(f)

    def dataset_folder(self, dataset):
        return self._ml_folder / "input" / "data" / dataset

    def model_folder(self):
        return self._ml_folder / "model"

    def write_failure(self, text):
        with open(self._ml_folder / "output" / "failure", "w") as f:
            f.write(text)


def import_string(name):
    """ Import a class given its absolute name.

        :param str name:
            The name of the model, e.g. mypackage.submodule.ModelTrainerClass.
    """
    modname, _, classname = name.rpartition(".")
    mod = importlib.import_module(modname)
    return getattr(mod, classname)


class ModelTrainer:
    """ An interface that allows ml2p-docker to train models within SageMaker.
    """

    def __init__(self, env):
        self.env = env

    def train(self):
        """ Train the model.

            This method should:

            * Read training data (using self.env to determine where to read data from).
            * Train the model.
            * Write the model out (using self.env to determine where to write the model
              to).
            * Write out any validation or model analysis alongside the model.
        """
        raise NotImplementedError("Sub-classes should implement .train()")


class ModelPredictor:
    """ An interface that allows ml2p-docker to make predictions from a model within
        SageMaker.
    """

    def __init__(self, env):
        self.env = env

    def setup(self):
        """ Called once before any calls to .predict(...) are made.

            This method should:

            * Load the model (using self.env to determine where to read the model from).
            * Allocate any other resources needed in order to make predictions.
        """
        pass

    def teardown(self):
        """ Called once after all calls to .predict(...) have ended.

            This method should:

            * Cleanup any resources acquired in .setup().
        """
        pass

    def invoke(self, data):
        """ Invokes the model and returns the full result.

            :param dict data:
                The input data the model is being invoked with.
            :rtype: dict
            :returns:
                The result as a dictionary.

            By default this method results a dictionary containing:

              * metadata: The result of calling .metadata(data).
              * result: The result of calling .result(data).
        """
        return {"metadata": self.metadata(data), "result": self.result(data)}

    def metadata(self, data):
        """ Return metadata for a prediction that is about to be made.

            :param dict data:
                The input data the prediction is going to be made from.
            :rtype: dict
            :returns:
                The metadata as a dictionary.

            By default this method returns a dictionary containing:

              * model_version: The ML2P_MODEL_VERSION (str).
              * timestamp: The UTC POSIX timestamp in seconds (float).
        """
        return {
            "model_version": self.env.model_version,
            "timestamp": datetime.datetime.utcnow().timestamp(),
        }

    def result(self, data):
        """ Make a prediction given the input data.

            :param dict data:
                The input data to make a prediction from.
            :rtype: dict
            :returns:
                The prediction result as a dictionary.
        """
        raise NotImplementedError("Sub-classes should implement .result(...)")


class Model:
    """ A holder for a trainer and predictor.

        Sub-classes should:

        * Set the attribute TRAINER to a ModelTrainer sub-class.
        * Set the attribute PREDICTOR to a ModelPredictor sub-class.
    """

    TRAINER = None
    PREDICTOR = None

    def trainer(self, env):
        if self.TRAINER is None:
            raise ValueError(".TRAINER should be an instance of ModelTrainer")
        return self.TRAINER(env)

    def predictor(self, env):
        if self.PREDICTOR is None:
            raise ValueError(".PREDICTOR should be an instance of ModelPredictor")
        return self.PREDICTOR(env)
