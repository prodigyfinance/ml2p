# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

from click.testing import CliRunner
import click
import pytest

from ml2p import __version__ as ml2p_version
from ml2p import cli
from ml2p.core import ModellingProject


class DummyCtx:
    pass


class ConfigMaker:
    def __init__(self, moto_session, tmp_path, **base_cfg):
        self._moto_session = moto_session
        self._tmp_path = tmp_path
        self._base_cfg = base_cfg

    def _apply_base_cfg(self, **kw):
        d = {}
        d.update(self._base_cfg)
        d.update(kw)
        return d

    def cfg(self, config_name="ml2p.yml", **kw):
        cfg_file = self._tmp_path / config_name
        cfg_file.write_text(json.dumps(self._apply_base_cfg(**kw)))
        return str(cfg_file)

    def ctx(self, **kw):
        ctx = DummyCtx()
        ctx.obj = ModellingProject(self.cfg(**kw))
        return ctx

    def s3(self):
        return self._moto_session.client("s3")


@pytest.fixture
def cfg_maker(moto_session, tmp_path):
    return ConfigMaker(
        moto_session,
        tmp_path,
        project="my-models",
        s3folder="s3://my-bucket/my-models/",
    )


class TestValidateModelType:
    def test_value_in_model_types(self, cfg_maker):
        ctx = cfg_maker.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert cli.validate_model_type(ctx, "model_type", "model-1") == "model-1"

    def test_value_not_in_model_types(self, cfg_maker):
        ctx = cfg_maker.ctx(models={"model-1": "my_models.ml2p.Model1"})
        with pytest.raises(click.BadParameter) as err:
            cli.validate_model_type(ctx, "model_type", "model-2")
        assert str(err.value) == "Unknown model type."

    def test_value_is_none_no_model_types(self, cfg_maker):
        ctx = cfg_maker.ctx(models={})
        assert cli.validate_model_type(ctx, "model_type", None) is None

    def test_value_is_none_single_model_type(self, cfg_maker):
        ctx = cfg_maker.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert cli.validate_model_type(ctx, "model_type", None) == "model-1"

    def test_value_is_none_multiple_model_types(self, cfg_maker):
        ctx = cfg_maker.ctx(
            models={
                "model-1": "my_models.ml2p.Model1",
                "model-2": "my_models.ml2p.Model2",
            }
        )
        with pytest.raises(click.BadParameter) as err:
            cli.validate_model_type(ctx, "model_type", None)
        assert str(err.value) == (
            "Model type may only be omitted if zero or one models are listed"
            " in the ML2P config YAML file."
        )


class TestModellingProjectWithSagemakerClient:
    def test_create(self, moto_session, cfg_maker):
        prj = cli.ModellingProjectWithSagemakerClient(cfg_maker.cfg())
        assert type(prj.client).__name__ == "SageMaker"


class TestML2P:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli.ml2p, ["--help"])
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
        result = runner.invoke(cli.ml2p, ["--version"])
        assert result.exit_code == 0
        assert result.output.splitlines() == ["ml2p, version {}".format(ml2p_version)]


class TestDataset:
    def test_help(self, cfg_maker):
        runner = CliRunner()
        result = runner.invoke(
            cli.ml2p, ["--cfg", cfg_maker.cfg(), "dataset", "--help"]
        )
        assert result.exit_code == 0
        assert result.output.splitlines()[:3] == [
            "Usage: ml2p dataset [OPTIONS] COMMAND [ARGS]...",
            "",
            "  Create and manage datasets.",
        ]

    def test_create(self, cfg_maker):
        s3 = cfg_maker.s3()
        s3.create_bucket(Bucket="my-bucket")
        runner = CliRunner()
        result = runner.invoke(
            cli.ml2p, ["--cfg", cfg_maker.cfg(), "dataset", "create", "ds-20201012"]
        )
        assert result.exit_code == 0
        assert result.output.splitlines() == []
        keys = [item["Key"] for item in s3.list_objects(Bucket="my-bucket")["Contents"]]
        assert keys == ["my-models/datasets/ds-20201012/README.rst"]
        readme = (
            s3.get_object(
                Bucket="my-bucket", Key="my-models/datasets/ds-20201012/README.rst"
            )["Body"]
            .read()
            .decode("utf-8")
        )
        assert readme == "Dataset ds-20201012 for project my-models."
