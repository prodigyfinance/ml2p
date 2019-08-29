# -*- coding: utf-8 -*-

""" ML2P core utilities.
"""

import datetime
import importlib
import json
import os
import pathlib
import re
import urllib.parse


class NamingError(Exception):
    """ Raised when a training job, model, or endpoint name does not follow
        convention.
    """

    pass


class S3URL:
    """ A friendly interface to an S3 URL. """

    def __init__(self, s3folder):
        self._s3url = urllib.parse.urlparse(s3folder)
        self._s3root = self._s3url.path.strip("/")

    def bucket(self):
        return self._s3url.netloc

    def path(self, suffix):
        path = self._s3root + "/" + suffix.lstrip("/")
        return path.lstrip("/")  # handles empty s3root

    def url(self, suffix=""):
        return "s3://{}/{}".format(self._s3url.netloc, self.path(suffix))


class SageMakerEnv:
    """ An interface to the SageMaker docker environment.

        Attributes:

        * `project` - The ML2P project name (type: str).

        * `s3` - The URL of the project S3 bucket (type: ml2p.core.S3URL).

        * `model_version` - The full job name of the deployed model, or None
          during training (type: str).

        Note: `project` and `s3` are not currently set in the training environemnt.
    """

    def __init__(self, ml_folder):
        self._ml_folder = pathlib.Path(ml_folder)
        # The attributes below are None except during model training.
        self.model_version = os.environ.get("ML2P_MODEL_VERSION", None)
        self.project = os.environ.get("ML2P_PROJECT", None)
        self.s3 = None
        if "ML2P_S3_URL" in os.environ:
            self.s3 = S3URL(os.environ["ML2P_S3_URL"])

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


def validate_name(name, resource):
    """ Validate that the name of the SageMaker resource complies with
        convention.

        :param str name:
            The name of the SageMaker resource to validate.
        :param str resource:
            The type of SageMaker resource to validate. One of "dataset",
            "training-job", "model", "endpoint".
    """
    re_dict = {
        "training-job": r"^[a-zA-Z0-9\-]*-[0-9].[0-9].[0-9](\-dev)?$",
        "model": r"^[a-zA-Z0-9\-]*-[0-9].[0-9].[0-9](\-dev)?$",
        "endpoint": r"^[a-zA-Z0-9\-]*-[0-9].[0-9].[0-9]"
        r"(\-dev)?(\-(live|analysis|test))?$",
    }
    message_dict = {
        "training-job": "Training job names should be in the"
        " format <model-name>-X.Y.Z-[dev]",
        "model": "Model names should be in the format <model-name>-X.Y.Z-[dev]",
        "endpoint": "Endpoint names should be in the"
        " format <model-name>-X.Y.Z-[dev]-[live|analysis|test]",
    }
    if re.match(re_dict[resource], name) is None:
        raise NamingError(message_dict[resource])
