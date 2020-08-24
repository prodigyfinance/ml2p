# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

from click.testing import CliRunner
import click
import pytest

from ml2p import __version__ as ml2p_version
from ml2p.cli import ml2p, validate_model_type
from ml2p.core import ModellingProject


class CtxFixture:
    def __init__(self, tmp_path, **cfg):
        cfg_file = tmp_path / "ml2p.yml"
        cfg_file.write_text(json.dumps(cfg))
        self.obj = ModellingProject(str(cfg_file))


@pytest.fixture
def ctx_single_model(tmp_path):
    return CtxFixture(
        tmp_path,
        project="my-models",
        s3folder="s3://my-bucket/my-models/",
        models={"model-1": "my_models.ml2p.Model1"},
    )


@pytest.fixture
def ctx_no_model(tmp_path):
    return CtxFixture(
        tmp_path, project="my-models", s3folder="s3://my-bucket/my-models/", models={},
    )


@pytest.fixture
def ctx_multiple_models(tmp_path):
    return CtxFixture(
        tmp_path,
        project="my-models",
        s3folder="s3://my-bucket/my-models/",
        models={"model-1": "my_models.ml2p.Model1", "model-2": "my_models.ml2p.Model2"},
    )


class TestValidateModelType:
    def test_value_in_model_types(self, ctx_single_model):
        assert (
            validate_model_type(ctx_single_model, "model_type", "model-1") == "model-1"
        )

    def test_value_not_in_model_types(self, ctx_single_model):
        with pytest.raises(click.BadParameter) as err:
            validate_model_type(ctx_single_model, "model_type", "model-2")
        assert str(err.value) == "Unknown model type."

    def test_value_is_none_no_model_types(self, ctx_no_model):
        assert validate_model_type(ctx_no_model, "model_type", None) is None

    def test_value_is_none_single_model_type(self, ctx_single_model):
        assert validate_model_type(ctx_single_model, "model_type", None) == "model-1"

    def test_value_is_none_multiple_model_types(self, ctx_multiple_models):
        with pytest.raises(click.BadParameter) as err:
            validate_model_type(ctx_multiple_models, "model_type", None)
        assert str(err.value) == (
            "Model type may only be omitted if zero or one models are listed"
            " in the ML2P config YAML file."
        )


class TestML2P:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(ml2p, ["--help"])
        assert result.exit_code == 0
        assert result.output.splitlines()[:5] == [
            "Usage: ml2p [OPTIONS] COMMAND [ARGS]...",
            "",
            "  Minimal Lovable Machine Learning Pipeline.",
            "",
            "  A friendlier interface to AWS SageMaker.",
        ]

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(ml2p, ["--version"])
        assert result.exit_code == 0
        assert result.output.splitlines() == ["ml2p, version {}".format(ml2p_version)]
