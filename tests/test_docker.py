# -*- coding: utf-8 -*-

""" Tests for ml2p.docker. """

import atexit
import collections
import re

import pytest
from click.testing import CliRunner
from flask_api.exceptions import APIException

import ml2p.docker
from ml2p import __version__ as ml2p_version
from ml2p.core import Model, ModelPredictor, ModelTrainer
from ml2p.docker import ml2p_docker
from ml2p.errors import ClientError, ServerError


def assert_cli_result(result, output, exit_code=0, exception=None):
    """ Assert that a CliRunner invocation returned the expected results. """
    assert result.exit_code == exit_code
    assert result.output.splitlines()[: len(output)] == output
    if exception is not None:
        assert type(result.exception) is type(exception)
        assert str(result.exception) == str(exception)
    else:
        assert result.exception is None


def model_cls_path(model):
    """ Return the dotted path name of a class. """
    return "{}.{}".format(model.__module__, model.__qualname__)


def check_train_or_serve(
    cmd, output, args=None, sagemaker=None, model=None, exit_code=0, exception=None
):
    """ Invoke the train or serve command of ml2p_docker and check the result. """
    runner = CliRunner()
    cmd_args = []
    if sagemaker:
        cmd_args += ["--ml-folder", str(sagemaker.ml_folder)]
    if model:
        cmd_args += ["--model", model_cls_path(model)]
    cmd_args += [cmd]
    if args:
        cmd_args += args
    result = runner.invoke(ml2p_docker, cmd_args)
    assert_cli_result(result, output, exit_code=exit_code, exception=exception)
    return result


def assert_traceback(tb, expected):
    """ Assert that a traceback matches a given pattern. """
    pattern = re.escape(expected)
    pattern = pattern.replace(r"\.\.\.", '[^"]*')
    pattern = pattern.replace(r"XX", "[0-9]+")
    assert re.match(pattern, tb), "Traceback:\n{}\ndoes not match expected:\n{}".format(
        tb, expected
    )


class HappyModelTrainer(ModelTrainer):
    def train(self):
        output = self.env.model_folder() / "output.txt"
        with output.open("w") as f:
            f.write("Success!")


class UnhappyModelTrainer(ModelTrainer):
    def train(self):
        raise ValueError("Much unhappiness")


class HappyModelPredictor(ModelPredictor):

    setup_called = False

    def setup(self):
        self.setup_called = True

    def result(self, data):
        if "client_error" in data:
            raise ClientError("client", data["client_error"])
        if "server_error" in data:
            raise ServerError("server", data["server_error"])
        if "flask_api_error" in data:
            raise APIException(data["flask_api_error"])
        if "generic_error" in data:
            raise Exception(data["generic_error"])
        return {"probability": 0.5, "input": data["input"]}


class HappyModel(Model):
    TRAINER = HappyModelTrainer
    PREDICTOR = HappyModelPredictor


class UnhappyModel(Model):
    TRAINER = UnhappyModelTrainer
    PREDICTOR = HappyModelPredictor


class DummyApp:
    """ A dummy flask app. """

    def __init__(self, runs):
        self._runs = runs

    def run(self, *args, **kw):
        self._runs.append((args, kw))


class TestML2PDocker:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(ml2p_docker, ["--help"])
        assert_cli_result(
            result,
            [
                "Usage: ml2p-docker [OPTIONS] COMMAND [ARGS]...",
                "",
                "  ML2P Sagemaker Docker container helper CLI.",
            ],
        )

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(ml2p_docker, ["--version"])
        assert result.exit_code == 0
        assert result.output.splitlines() == [
            "ml2p-docker, version {}".format(ml2p_version)
        ]


class TestML2PDockerTrain:
    def check_train(self, *args, **kw):
        return check_train_or_serve("train", *args, **kw)

    def test_help(self):
        self.check_train(
            ["Usage: ml2p-docker train [OPTIONS]", "", "  Train the model."],
            args=["--help"],
        )

    def test_training_success(self, sagemaker):
        model_folder = sagemaker.ml_folder.join("model").mkdir()
        sagemaker.train()
        self.check_train(
            ["Starting training job test-train-1.2.3.", "Done."],
            sagemaker=sagemaker,
            model=HappyModel,
        )
        assert model_folder.join("output.txt").read() == "Success!"

    def test_train_success_model_passed_via_hyperparameters(self, sagemaker):
        model_folder = sagemaker.ml_folder.join("model").mkdir()
        sagemaker.train(ML2P_MODEL_CLS=model_cls_path(HappyModel))
        self.check_train(
            ["Starting training job test-train-1.2.3.", "Done."],
            sagemaker=sagemaker,
            model=None,
        )
        assert model_folder.join("output.txt").read() == "Success!"

    def test_training_failure_no_model(self, sagemaker):
        sagemaker.train()
        self.check_train(
            [
                "Usage: ml2p-docker train [OPTIONS]",
                "",
                "Error: The global parameter --model must either be given when calling"
                " the train command or --model-type must be given when creating the"
                " training job.",
            ],
            exit_code=2,
            exception=SystemExit(2),
            sagemaker=sagemaker,
            model=None,
        )

    def test_training_exception(self, sagemaker):
        output_folder = sagemaker.ml_folder.join("output").mkdir()
        sagemaker.train()
        self.check_train(
            ["Starting training job test-train-1.2.3."],
            exit_code=1,
            exception=ValueError("Much unhappiness"),
            sagemaker=sagemaker,
            model=UnhappyModel,
        )
        assert_traceback(
            output_folder.join("failure").read(),
            "\n".join(
                [
                    r"Traceback (most recent call last):",
                    r'  File ".../ml2p/docker.py", line XX, in train',
                    r"    trainer.train()",
                    r'  File ".../tests/test_docker.py", line XX, in train',
                    r'    raise ValueError("Much unhappiness")',
                    r"ValueError: Much unhappiness",
                ]
            ),
        )


docker_serve_type = collections.namedtuple(
    "docker_serve_type", ["app", "teardowns", "runs"]
)


@pytest.fixture
def docker_serve(monkeypatch):
    teardowns = []
    runs = []
    app = DummyApp(runs)
    monkeypatch.setattr(atexit, "register", teardowns.append)
    monkeypatch.setattr(ml2p.docker, "app", app)
    return docker_serve_type(app=app, teardowns=teardowns, runs=runs)


class TestML2PDockerServe:
    def check_serve(self, *args, **kw):
        return check_train_or_serve("serve", *args, **kw)

    def test_help(self):
        self.check_serve(
            [
                "Usage: ml2p-docker serve [OPTIONS]",
                "",
                "  Serve the model and make predictions.",
            ],
            args=["--help"],
        )

    def test_serve(self, docker_serve, sagemaker):
        env = sagemaker.serve()
        self.check_serve(
            ["Starting server for model version test-model-1.2.3.", "Done."],
            sagemaker=sagemaker,
            model=HappyModel,
        )
        assert isinstance(docker_serve.app.predictor, HappyModelPredictor)
        assert docker_serve.app.predictor.env.model_folder() == env.model_folder()
        assert docker_serve.app.predictor.setup_called
        assert docker_serve.teardowns == [docker_serve.app.predictor.teardown]
        assert docker_serve.runs == [
            ((), {"host": "0.0.0.0", "port": 8080, "debug": False})
        ]

    def test_serve_model_passed_via_hyperparameters(self, docker_serve, sagemaker):
        sagemaker.serve(ML2P_MODEL_CLS=model_cls_path(HappyModel))
        self.check_serve(
            ["Starting server for model version test-model-1.2.3.", "Done."],
            sagemaker=sagemaker,
            model=None,
        )
        assert isinstance(docker_serve.app.predictor, HappyModelPredictor)

    def test_serve_failure_no_model(self, docker_serve, sagemaker):
        sagemaker.serve()
        self.check_serve(
            [
                "Usage: ml2p-docker serve [OPTIONS]",
                "",
                "Error: The global parameter --model must either be given when calling"
                " the serve command or --model-type must be given when creating the"
                " model.",
            ],
            exit_code=2,
            exception=SystemExit(2),
            sagemaker=sagemaker,
            model=None,
        )


@pytest.fixture
def api_client(sagemaker):
    app = ml2p.docker.app
    app.config["TESTING"] = True
    app.predictor = HappyModelPredictor(sagemaker.serve())
    app.predictor.setup()
    client = app.test_client()

    yield client

    app.predictor.teardown()
    del app.predictor
    del app.config["TESTING"]


class TestAPI:
    def test_ping(self, api_client):
        response = api_client.get("/ping")
        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert response.get_json() == {
            "model_version": "test-model-1.2.3",
            "ml2p_version": str(ml2p_version),
        }

    def test_invocations(self, api_client, fake_utcnow):
        response = api_client.post("/invocations", json={"input": 12345})
        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert response.get_json() == {
            "metadata": {
                "model_version": "test-model-1.2.3",
                "ml2p_version": str(ml2p_version),
                "timestamp": fake_utcnow.timestamp(),
            },
            "result": {"probability": 0.5, "input": 12345},
        }

    def test_batch_invocations(self, api_client, fake_utcnow):
        response = api_client.post(
            "/invocations", json={"instances": [{"input": 12345}, {"input": 12346}]}
        )
        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert response.get_json() == {
            "predictions": [
                {
                    "metadata": {
                        "model_version": "test-model-1.2.3",
                        "ml2p_version": str(ml2p_version),
                        "timestamp": fake_utcnow.timestamp(),
                    },
                    "result": {"probability": 0.5, "input": 12345},
                },
                {
                    "metadata": {
                        "model_version": "test-model-1.2.3",
                        "ml2p_version": str(ml2p_version),
                        "timestamp": fake_utcnow.timestamp(),
                    },
                    "result": {"probability": 0.5, "input": 12346},
                },
            ]
        }

    def test_invocation_with_generic_error(self, api_client, fake_utcnow):
        with pytest.raises(Exception) as err:
            api_client.post("/invocations", json={"generic_error": "test error"})
        assert str(err.value) == "test error"
        assert err.value.__class__ is Exception

    def test_invocation_with_flask_api_error(self, api_client, fake_utcnow):
        response = api_client.post(
            "/invocations", json={"flask_api_error": "message eep"}
        )
        assert response.status_code == 500
        assert response.content_type == "application/json"
        assert response.get_json() == {"message": "message eep"}

    def test_invocation_with_client_error(self, api_client, fake_utcnow):
        response = api_client.post("/invocations", json={"client_error": "bad param"})
        assert response.status_code == 400
        assert response.content_type == "application/json"
        assert response.get_json() == {"message": "client", "details": ["bad param"]}

    def test_invocation_with_server_error(self, api_client, fake_utcnow):
        response = api_client.post("/invocations", json={"server_error": "eep"})
        assert response.status_code == 500
        assert response.content_type == "application/json"
        assert response.get_json() == {"message": "server", "details": ["eep"]}

    def test_execution_parameters(self, api_client, fake_utcnow):
        response = api_client.get("/execution-parameters")
        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert response.get_json() == {
            "MaxConcurrentTransforms": 1,
            "BatchStrategy": "MULTI_RECORD",
            "MaxPayloadInMB": 6,
        }
