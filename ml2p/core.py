# -*- coding: utf-8 -*-

""" ML2P core utilities.
"""

import datetime
import enum
import importlib
import json
import os
import pathlib
import re
import urllib.parse
import warnings

from . import errors, hyperparameters


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


class SageMakerEnvType(enum.Enum):
    """ The type of SageMakerEnvironment.
    """

    TRAIN = "train"
    SERVE = "serve"


class SageMakerEnv:
    """ An interface to the SageMaker docker environment.

        Attributes that are expected to be available in both training and serving
        environments:

        * `env_type` - Whether this is a training or serving environment
          (type: ml2p.core.SageMakerEnvType).

        * `project` - The ML2P project name (type: str).

        * `model_cls` - The fulled dotted Python name of the ml2p.core.Model class to
          be used for training and prediction (type: str). This may be None if the
          docker image itself specifies the name with `ml2p-docker --model ...`.

        * `s3` - The URL of the project S3 bucket (type: ml2p.core.S3URL).

        Attributes that are only expected to be available while training (and that will
        be None when serving the model):

        * `training_job_name` - The full job name of the training job (type: str).

        Attributes that are only expected to be available while serving the model (and
        that will be None when serving the model):

        * `model_version` - The full job name of the deployed model, or None
          during training (type: str).

        In the training environment settings are loaded from hyperparameters stored by
        ML2P when the training job is created.

        In the serving environment settings are loaded from environment variables stored
        by ML2P when the model is created.
    """

    TRAIN = SageMakerEnvType.TRAIN
    SERVE = SageMakerEnvType.SERVE

    def __init__(self, ml_folder):
        self._ml_folder = pathlib.Path(ml_folder)
        if "TRAINING_JOB_NAME" in os.environ:
            # this is a training job instance
            self.env_type = self.TRAIN
            environ = self.hyperparameters().get("ML2P_ENV", {})
            self.training_job_name = os.environ.get("TRAINING_JOB_NAME", None)
            self.model_version = None
        else:
            # this is a serving instance
            self.env_type = self.SERVE
            environ = os.environ
            self.training_job_name = None
            self.model_version = environ.get("ML2P_MODEL_VERSION", None)
        self.project = environ.get("ML2P_PROJECT", None)
        self.model_cls = environ.get("ML2P_MODEL_CLS", None)
        self.s3 = None
        if "ML2P_S3_URL" in environ:
            self.s3 = S3URL(environ["ML2P_S3_URL"])

    def hyperparameters(self):
        hp_path = self._ml_folder / "input" / "config" / "hyperparameters.json"
        if not hp_path.exists():
            return {}
        with hp_path.open() as f:
            return hyperparameters.decode(json.load(f))

    def resourceconfig(self):
        rc_path = self._ml_folder / "input" / "config" / "resourceconfig.json"
        if not rc_path.exists():
            return {}
        with rc_path.open() as f:
            return json.load(f)

    def dataset_folder(self, dataset=None):
        if dataset is None:
            dataset = "training"
        else:
            warnings.warn(
                "Passing a dataset name to dataset_folder method(...) is deprecated."
                " If you wish to access the ML2P training dataset, do not pass any"
                " parameters. If you wish to access data for a specific channel, please"
                " use data_channel_folder(...) instead, which matches the terminology"
                " used by AWS SageMaker more accurately.",
                DeprecationWarning,
            )
        return self._ml_folder / "input" / "data" / dataset

    def data_channel_folder(self, channel):
        return self._ml_folder / "input" / "data" / channel

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
        "dataset": r"^[a-zA-Z0-9\-]*-[0-9]{8}$",
        "training-job": r"^[a-zA-Z0-9\-]+-[0-9]+-[0-9]+-[0-9]+(\-dev)?$",
        "model": r"^[a-zA-Z0-9\-]+-[0-9]+-[0-9]+-[0-9]+(\-dev)?$",
        "endpoint": r"^[a-zA-Z0-9\-]+-[0-9]+-[0-9]+-[0-9]+"
        r"(\-dev)?(\-(live|analysis|test))?$",
    }
    message_dict = {
        "dataset": "Dataset names should be in the format <model-name>-YYYYMMDD",
        "training-job": "Training job names should be in the"
        " format <model-name>-X-Y-Z-[dev]",
        "model": "Model names should be in the format <model-name>-X-Y-Z-[dev]",
        "endpoint": "Endpoint names should be in the"
        " format <model-name>-X-Y-Z-[dev]-[live|analysis|test]",
    }
    if re.match(re_dict[resource], name) is None:
        raise errors.NamingError(message_dict[resource])
