# -*- coding: utf-8 -*-

""" ML2P core utilities.
"""

import datetime
import enum
import importlib
import json
import os
import pathlib
import shutil
import tarfile
import urllib.parse
import uuid
import warnings

import boto3
import yaml

from . import __version__ as ml2p_version
from . import hyperparameters
from .errors import LocalEnvError


class ModellingProject:
    """ Object for holding CLI context. """

    def __init__(self, cfg):
        with open(cfg) as f:
            self.cfg = yaml.safe_load(f)
        self.project = self.cfg["project"]
        self.s3 = S3URL(self.cfg["s3folder"])
        self.train = ModellingSubCfg(self.cfg, "train")
        self.deploy = ModellingSubCfg(self.cfg, "deploy")
        self.notebook = ModellingSubCfg(self.cfg, "notebook")
        self.models = ModellingSubCfg(self.cfg, "models", defaults="models")

    def full_job_name(self, job_name):
        return "{}-{}".format(self.project, job_name)

    def tags(self):
        return [{"Key": "ml2p-project", "Value": self.cfg["project"]}]


class ModellingSubCfg:
    """ Holder for training or deployment config. """

    def __init__(self, cfg, section, defaults="defaults"):
        self._cfg = cfg
        self._defaults = cfg.get(defaults, {})
        self._section = cfg.get(section, {})

    def __getattr__(self, name):
        if name in self._section:
            return self._section[name]
        return self._defaults[name]

    def __getitem__(self, name):
        if name in self._section:
            return self._section[name]
        return self._defaults[name]

    def __setitem__(self, name, value):
        self._section[name] = value

    def keys(self):
        keys = set(self._section.keys())
        keys.update(self._defaults.keys())
        return sorted(keys)

    def get(self, name, default=None):
        if name in self._section:
            return self._section[name]
        return self._defaults.get(name, default)


class S3URL:
    """ A friendly interface to an S3 URL. """

    def __init__(self, s3folder):
        self._s3url = urllib.parse.urlparse(s3folder)
        self._s3root = self._s3url.path.strip("/")

    def bucket(self):
        """ Return the bucket of the S3 URL.

            :rtype: str
            :returns:
                The bucket of the S3 URL.
        """
        return self._s3url.netloc

    def path(self, suffix):
        """ Return the base path of the S3 URL followed by a '/' and the
            given suffix.

            :param str suffix:
                The suffix to append.

            :rtype: str
            :returns:
                The path with the suffix appended.
        """
        path = self._s3root + "/" + suffix.lstrip("/")
        return path.lstrip("/")  # handles empty s3root

    def url(self, suffix=""):
        """ Return S3 URL followed by a '/' and the given suffix.

            :param str suffix:
                The suffix to append. Default: "".

            :rtype: str
            :returns:
                The URL with the suffix appended.
        """
        return "s3://{}/{}".format(self._s3url.netloc, self.path(suffix))


class SageMakerEnvType(enum.Enum):
    """ The type of SageMakerEnvironment.
    """

    TRAIN = "train"
    SERVE = "serve"
    LOCAL = "local"


class SageMakerEnv:
    """ An interface to the SageMaker docker environment.

        Attributes that are expected to be available in both training and serving
        environments:

        * `env_type` - Whether this is a training, serving or local environment
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

        * `record_invokes` - Whether to store a record of each invocation of the
          endpoint in S3 (type: bool).

        In the training environment settings are loaded from hyperparameters stored by
        ML2P when the training job is created.

        In the serving environment settings are loaded from environment variables stored
        by ML2P when the model is created.
    """

    TRAIN = SageMakerEnvType.TRAIN
    SERVE = SageMakerEnvType.SERVE
    LOCAL = SageMakerEnvType.LOCAL

    def __init__(self, ml_folder, environ=None):
        self._ml_folder = pathlib.Path(ml_folder)
        if environ is None:
            if "TRAINING_JOB_NAME" in os.environ:
                # this is a training job instance
                environ = self._train_environ()
            else:
                # this is a serving instance
                environ = self._serve_environ()
        self.env_type = environ["env_type"]
        self.training_job_name = environ["training_job_name"]
        self.model_version = environ["model_version"]
        self.record_invokes = environ["record_invokes"]
        self.project = environ["project"]
        self.model_cls = environ["model_cls"]
        self.s3 = None
        if environ["s3_url"]:
            self.s3 = S3URL(environ["s3_url"])

    def _train_environ(self):
        environ = self.hyperparameters().get("ML2P_ENV", {})
        return {
            "env_type": self.TRAIN,
            "training_job_name": os.environ.get("TRAINING_JOB_NAME", None),
            "model_version": None,
            "record_invokes": None,
            "project": environ.get("ML2P_PROJECT", None),
            "model_cls": environ.get("ML2P_MODEL_CLS", None),
            "s3_url": environ.get("ML2P_S3_URL", None),
        }

    def _serve_environ(self):
        environ = os.environ
        return {
            "env_type": self.SERVE,
            "training_job_name": None,
            "model_version": environ.get("ML2P_MODEL_VERSION", None),
            "record_invokes": environ.get("ML2P_RECORD_INVOKES", "false") == "true",
            "project": environ.get("ML2P_PROJECT", None),
            "model_cls": environ.get("ML2P_MODEL_CLS", None),
            "s3_url": environ.get("ML2P_S3_URL", None),
        }

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


class LocalEnv(SageMakerEnv):
    """ An interface to a local dummy of the SageMaker environment.

        :param str ml_folder:
            The directory the environments files are stored in. An
            error is raised if this directory does not exist. Files
            and folders are created within this directory as needed.
        :param str cfg:
            The path to an ml2p.yml configuration file.
        :param boto3.session.Session session:
            A boto3 session object. Maybe be None if downloading files from
            S3 is not required.

        Attributes that are expected to be available in the local environment:

        * `env_type` - Whether this is a training, serving or local environment
          (type: ml2p.core.SageMakerEnvType).

        * `project` - The ML2P project name (type: str).

        * `s3` - The URL of the project S3 bucket (type: ml2p.core.S3URL).

        * `model_version` - The fixed value "local" (type: str).

        In the local environment settings are loaded directly from the ML2P
        configuration file.
    """

    def __init__(self, ml_folder, cfg, session=None):
        self._session = session
        self._prj = ModellingProject(cfg)
        super().__init__(ml_folder, environ=self._local_environ())
        if not self._ml_folder.is_dir():
            raise LocalEnvError(f"Local environment folder {ml_folder} does not exist.")
        self.model_folder().mkdir(exist_ok=True)

    def _local_environ(self):
        return {
            "env_type": self.LOCAL,
            "training_job_name": None,
            "model_version": "local",
            "record_invokes": False,
            "project": self._prj.project,
            "model_cls": None,
            "s3_url": self._prj.s3.url(),
        }

    def clean_model_folder(self):
        """ Remove and recreate the model folder.

            This is useful to run before training a model if one wants to ensure
            that the model folder is empty beforehand.
        """
        model_folder = self.model_folder()
        shutil.rmtree(model_folder)
        model_folder.mkdir()

    def download_dataset(self, dataset):
        """ Download the given dataset from S3 into the local environment.

            :param str dataset:
                The name of the dataset in S3 to download.
        """
        if self._session is None:
            raise LocalEnvError("Downloading datasets requires a boto session.")
        client = self._session.resource("s3")
        bucket = client.Bucket(self.s3.bucket())

        local_dataset = self.dataset_folder()
        local_dataset.mkdir(parents=True, exist_ok=True)
        s3_dataset = self.s3.path("datasets") + "/" + dataset
        len_prefix = len(s3_dataset)

        for s3_object in bucket.objects.filter(Prefix=s3_dataset):
            if s3_object.key.endswith("/"):
                # keys that end in a / are probably folders, so skip downloading them
                continue
            local_object = local_dataset / (s3_object.key[len_prefix:].lstrip("/"))
            local_object.parent.mkdir(parents=True, exist_ok=True)
            with local_object.open("wb") as f:
                bucket.download_fileobj(s3_object.key, f)

    def download_model(self, training_job):
        """ Download the given trained model from S3 and unpack it into the local environment.

            :param str training_job:
                The name of the training job whose model should be downloaded.
        """
        if self._session is None:
            raise LocalEnvError("Downloading models requires a boto session.")
        client = self._session.resource("s3")
        bucket = client.Bucket(self.s3.bucket())

        local_model_tgz = self.model_folder() / "model.tar.gz"
        local_model_tgz.parent.mkdir(parents=True, exist_ok=True)
        s3_model_tgz = (
            self.s3.path("/models")
            + "/"
            + self._prj.full_job_name(training_job)
            + "/output/model.tar.gz"
        )

        with local_model_tgz.open("wb") as f:
            bucket.download_fileobj(s3_model_tgz, f)

        tf = tarfile.open(local_model_tgz)
        tf.extractall(self.model_folder())


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
        self.s3_client = boto3.client("s3")

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

              * metadata: The result of calling .metadata().
              * result: The result of calling .result(data).
        """
        prediction = {"metadata": self.metadata(), "result": self.result(data)}
        if self.env.record_invokes:
            self.record_invoke(data, prediction)
        return prediction

    def metadata(self):
        """ Return metadata for a prediction that is about to be made.

            :rtype: dict
            :returns:
                The metadata as a dictionary.

            By default this method returns a dictionary containing:

              * model_version: The ML2P_MODEL_VERSION (str).
              * timestamp: The UTC POSIX timestamp in seconds (float).
        """
        return {
            "model_version": self.env.model_version,
            "ml2p_version": ml2p_version,
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

    def batch_invoke(self, data):
        """ Invokes the model on a batch of input data and returns the full result for
            each instance.

            :param dict data:
                The batch of input data the model is being invoked with.
            :rtype: list
            :returns:
                The result as a list of dictionaries.

            By default this method results a list of dictionaries containing:

              * metadata: The result of calling .metadata().
              * result: The result of calling .batch_result(data).
        """
        metadata = self.metadata()
        results = self.batch_result(data)
        predictions = [{"metadata": metadata, "result": result} for result in results]
        if self.env.record_invokes:
            for datum, prediction in zip(data, predictions):
                self.record_invoke(datum, prediction)
        return {"predictions": predictions}

    def batch_result(self, data):
        """ Make a batch prediction given a batch of input data.

            :param dict data:
                The batch of input data to make a prediction from.
            :rtype: list
            :returns:
                The list of predictions made for instance of the input data.

            This method can be overrided for sub-classes in order to improve
            performance of batch predictions.
        """
        return [self.result(datum) for datum in data]

    def record_invoke_id(self, datum, prediction):
        """ Return an id for an invocation record.

            :param dict datum:
                The dictionary of input values passed when invoking the endpoint.

            :param dict result:
                The prediction returned for datum by this predictor.

            :returns dict:
                Returns an *ordered* dictionary of key-value pairs that make up
                the unique identifier for the invocation request.

            By default this method returns a dictionary containing the following:

                * "ts": an ISO8601 formatted UTC timestamp.
                * "uuid": a UUID4 unique identifier.

            Sub-classes may override this method to return their own identifiers,
            but including these default identifiers is recommended.

            The name of the record in S3 is determined by combining the key value pairs
            with a dash ("-") and then separating each pair with a double dash ("--").
        """
        return {"ts": datetime.datetime.utcnow().isoformat(), "uuid": str(uuid.uuid4())}

    def record_invoke(self, datum, prediction):
        """ Store an invocation of the endpoint in the ML2P project S3 bucket.

            :param dict datum:
                The dictionary of input values passed when invoking the endpoint.

            :param dict result:
                The prediction returned for datum by this predictor.
        """
        invoke_id = self.record_invoke_id(datum, prediction)
        record_filename = (
            "--".join(["{}-{}".format(k, v) for k, v in invoke_id.items()]) + ".json"
        )
        record = {"input": datum, "result": prediction}
        record_bytes = json.dumps(record).encode("utf-8")
        s3_key = self.env.s3.path(
            "/predictions/{}/{}".format(self.env.model_version, record_filename)
        )
        self.s3_client.put_object(
            Bucket=self.env.s3.bucket(), Key=s3_key, Body=record_bytes
        )


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
