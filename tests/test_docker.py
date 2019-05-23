# -*- coding: utf-8 -*-

""" Tests for ml2p.docker. """

import re

from click.testing import CliRunner

from ml2p.docker import ml2p_docker
from ml2p.core import ModelTrainer


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
    cmd, extra_args, output, sagemaker=None, model=None, exit_code=0, exception=None
):
    """ Invoke the train or serve command of ml2p_docker and check the result. """
    runner = CliRunner()
    args = [cmd]
    if sagemaker:
        args = ["--ml-folder", str(sagemaker.ml_folder)] + args
    if model:
        args += ["{}.{}".format(model.__module__, model.__qualname__)]
    args += extra_args
    result = runner.invoke(ml2p_docker, args)
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
            ["--help"],
            [
                "Usage: ml2p-docker train [OPTIONS] MODEL_TRAINER",
                "",
                "  Train the model.",
            ],
        )

    def test_training_success(self, sagemaker):
        model_folder = sagemaker.ml_folder.join("model").mkdir()
        self.check_train(
            [],
            ["Training model version test-model-1.2.3.", "Done."],
            sagemaker=sagemaker,
            model=HappyModelTrainer,
        )
        assert model_folder.join("output.txt").read() == "Success!"

    def test_training_exception(self, sagemaker):
        output_folder = sagemaker.ml_folder.join("output").mkdir()
        self.check_train(
            [],
            ["Training model version test-model-1.2.3."],
            exit_code=1,
            exception=ValueError("Much unhappiness"),
            sagemaker=sagemaker,
            model=UnhappyModelTrainer,
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
            ["--help"],
            [
                "Usage: ml2p-docker serve [OPTIONS] MODEL_PREDICTOR",
                "",
                "  Serve the model and make predictions.",
            ],
        )
