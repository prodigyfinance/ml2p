# -*- coding: utf-8 -*-

""" ML2P core utilities.
"""

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


class ML2PDockerInterface:
    """ An interface that allows ml2p-docker to train and deploy models within
        SageMaker.
    """

    def __init__(self, env):
        self.env = env

    def train(self):
        raise NotImplementedError("Sub-classes should implement .train()")

    def setup_predict(self):
        """ Called once before any calls to .predict(...) are made.

            Any setup required before predictions can be made (e.g. loading the model)
            should happen in this method.
        """
        pass

    def teardown_predict(self):
        """ Called once after all calls to .predict(...) have ended.

            Any cleanup of resources acquired in .setup_predict() should be performed
            here.
        """
        pass

    def predict(self, data):
        raise NotImplementedError("Sub-classes should implement .predict()")
        # TODO: Document that this should just return the prediction result
        #       Final result is a combination of the prediction and model
        #       metadata.
        #       Would be good if model metadata was standardised, but maybe
        #       we don't know enough yet to know what that should be?
        #       - e.g. threshold
        #       There is probably also metadata that ML2P should add:
        #       - e.g. current timestamp
        #       - e.g. ml2p_model_version
        #       - e.g. other stuff from SageMaker?
