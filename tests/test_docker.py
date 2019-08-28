# -*- coding: utf-8 -*-

""" Tests for ml2p.docker. """

import atexit
import re

import pytest
from click.testing import CliRunner

import ml2p.docker
from ml2p.core import Model, ModelPredictor, ModelTrainer
from ml2p.docker import ml2p_docker


def assert_cli_result(result, output, exit_code=0, exception=None):
    """ Assert that a CliRunner invocation returned the expected results. """
    assert result.exit_code == exit_code
    assert result.output.splitlines()[: len(output)] == output
    if exception is not None:
        assert type(result.exception) is type(exception)
        assert str(result.exception) == str(exception)
    else:
        assert result.exception is None


def check_train_or_serve(
    cmd, output, args=None, sagemaker=None, model=None, exit_code=0, exception=None
):
    """ Invoke the train or serve command of ml2p_docker and check the result. """
    runner = CliRunner()
    cmd_args = []
    if sagemaker:
        cmd_args += ["--ml-folder", str(sagemaker.ml_folder)]
    if model:
        cmd_args += ["--model", "{}.{}".format(model.__module__, model.__qualname__)]
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

    def test_run(self, monkeypatch, sagemaker):
        teardowns = []
        runs = []
        app = DummyApp(runs)
        monkeypatch.setattr(atexit, "register", teardowns.append)
        monkeypatch.setattr(ml2p.docker, "app", app)
        env = sagemaker.serve()
        self.check_serve(
            ["Starting server for model version test-model-1.2.3.", "Done."],
            sagemaker=sagemaker,
            model=HappyModel,
        )
        assert isinstance(app.predictor, HappyModelPredictor)
        assert app.predictor.env.model_folder() == env.model_folder()
        assert app.predictor.setup_called
        assert teardowns == [app.predictor.teardown]
        assert runs == [((), {"host": "0.0.0.0", "port": 8080, "debug": False})]


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
        assert response.get_json() == {"model_version": "test-model-1.2.3"}

    def test_invocations(self, api_client, fake_utcnow):
        response = api_client.post("/invocations", json={"input": 12345})
        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert response.get_json() == {
            "metadata": {
                "model_version": "test-model-1.2.3",
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
                        "timestamp": fake_utcnow.timestamp(),
                    },
                    "result": {"probability": 0.5, "input": 12345},
                },
                {
                    "metadata": {
                        "model_version": "test-model-1.2.3",
                        "timestamp": fake_utcnow.timestamp(),
                    },
                    "result": {"probability": 0.5, "input": 12346},
                },
            ]
        }

    def test_execution_parameters(self, api_client, fake_utcnow):
        response = api_client.get("/execution-parameters")
        assert response.status_code == 200
        assert response.content_type == "application/json"
        assert response.get_json() == {
            "MaxConcurrentTransforms": 1,
            "BatchStrategy": "MULTI_RECORD",
            "MaxPayloadInMB": 6,
        }
