# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

import click
import pytest
from click.testing import CliRunner

from ml2p import __version__ as ml2p_version
from ml2p import cli
from ml2p.core import ModellingProject


class DummyCtx:
    pass


class CLIHelper:
    def __init__(self, moto_session, tmp_path, bucket, base_cfg):
        self._moto_session = moto_session
        self._tmp_path = tmp_path
        self._bucket = bucket
        self._base_cfg = base_cfg
        self.s3 = self._moto_session.client("s3")

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

    def s3_create_bucket(self):
        self.s3.create_bucket(Bucket=self._bucket)

    def s3_list_objects(self):
        list_objects = self.s3.list_objects(Bucket=self._bucket)
        if "Contents" not in list_objects:
            return None
        return [item["Key"] for item in list_objects["Contents"]]

    def s3_get_object(self, key):
        return self.s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def s3_put_object(self, key, data):
        return self.s3.put_object(Bucket=self._bucket, Key=key, Body=data)

    def invoke(
        self,
        args,
        output=None,
        output_startswith=None,
        output_jsonl=None,
        exit_code=0,
        cfg=None,
    ):
        if cfg is None:
            cfg = {}
        runner = CliRunner()
        result = runner.invoke(
            cli.ml2p, ["--cfg", self.cfg(**cfg)] + args, catch_exceptions=False,
        )
        assert result.exit_code == exit_code
        if output is not None:
            assert result.output.splitlines() == output
        if output_startswith is not None:
            assert (
                result.output.splitlines()[: len(output_startswith)]
                == output_startswith
            )
        if output_jsonl is not None:
            assert result.output == "\n".join(
                [json.dumps(data, indent=2) for data in output_jsonl] + [""]
            )


@pytest.fixture
def cli_helper(moto_session, tmp_path):
    return CLIHelper(
        moto_session,
        tmp_path,
        bucket="my-bucket",
        base_cfg={"project": "my-models", "s3folder": "s3://my-bucket/my-models/"},
    )


class TestValidateModelType:
    def test_value_in_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert cli.validate_model_type(ctx, "model_type", "model-1") == "model-1"

    def test_value_not_in_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        with pytest.raises(click.BadParameter) as err:
            cli.validate_model_type(ctx, "model_type", "model-2")
        assert str(err.value) == "Unknown model type."

    def test_value_is_none_no_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={})
        assert cli.validate_model_type(ctx, "model_type", None) is None

    def test_value_is_none_single_model_type(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert cli.validate_model_type(ctx, "model_type", None) == "model-1"

    def test_value_is_none_multiple_model_types(self, cli_helper):
        ctx = cli_helper.ctx(
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
    def test_create(self, cli_helper):
        prj = cli.ModellingProjectWithSagemakerClient(cli_helper.cfg())
        assert type(prj.client).__name__ == "MockSageMakerClient"


class TestML2P:
    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["--help"],
            output_startswith=[
                "Usage: ml2p [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Minimal Lovable Machine Learning Pipeline.",
                "",
                "  A friendlier interface to AWS SageMaker.",
            ],
        )

    def test_version(self, cli_helper):
        cli_helper.invoke(
            ["--version"], output=["ml2p, version {}".format(ml2p_version)]
        )


class TestInit:
    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["init", "--help"],
            output_startswith=[
                "Usage: ml2p init [OPTIONS]",
                "",
                "  Initialize the project S3 bucket.",
            ],
        )

    def test_init(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["init"], output=[])
        assert cli_helper.s3_list_objects() == [
            "my-models/datasets/README.rst",
            "my-models/models/README.rst",
        ]
        assert (
            cli_helper.s3_get_object("my-models/datasets/README.rst").decode("utf-8")
            == "Datasets for my-models."
        )
        assert (
            cli_helper.s3_get_object("my-models/models/README.rst").decode("utf-8")
            == "Models for my-models."
        )


class TestDataset:
    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["dataset", "--help"],
            output_startswith=[
                "Usage: ml2p dataset [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and manage datasets.",
            ],
        )

    def test_create(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["dataset", "create", "ds-20201012"], output=[])
        assert cli_helper.s3_list_objects() == [
            "my-models/datasets/ds-20201012/README.rst"
        ]
        assert (
            cli_helper.s3_get_object(
                "my-models/datasets/ds-20201012/README.rst"
            ).decode("utf-8")
            == "Dataset ds-20201012 for project my-models."
        )

    def test_list(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["dataset", "create", "ds-20201012"])
        cli_helper.invoke(["dataset", "create", "ds-20201013"])
        cli_helper.invoke(
            ["dataset", "list"], output_jsonl=["ds-20201012", "ds-20201013"]
        )

    def test_delete(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["dataset", "create", "ds-20201012"], [])
        cli_helper.invoke(["dataset", "delete", "ds-20201012"], [])
        assert cli_helper.s3_list_objects() is None

    def test_ls(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/a.txt", b"aa")
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/b.txt", b"bbb")
        cli_helper.invoke(
            ["dataset", "ls", "ds-20201012"],
            output_jsonl=[
                {"filename": "a.txt", "size": 2},
                {"filename": "b.txt", "size": 3},
            ],
        )

    def test_up(self, cli_helper, data_fixtures):
        cli_helper.s3_create_bucket()
        training_set = str(data_fixtures / "training_set.csv")
        cli_helper.invoke(["dataset", "up", "ds-20201012", training_set], [])
        assert cli_helper.s3_list_objects() == [
            "my-models/datasets/ds-20201012/training_set.csv"
        ]
        assert (
            cli_helper.s3_get_object(
                "my-models/datasets/ds-20201012/training_set.csv"
            ).decode("utf-8")
            == 'X1,X2,X3,y\n"A",1,10.5,1\n"B",2,7.4,0\n'
        )

    def test_dn(self, cli_helper, data_fixtures, tmpdir):
        cli_helper.s3_create_bucket()
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/a.txt", b"aa")
        with tmpdir.as_cwd():
            cli_helper.invoke(["dataset", "dn", "ds-20201012", "a.txt"])
        assert (tmpdir / "a.txt").read() == "aa"

    def test_rm(self, cli_helper, data_fixtures):
        cli_helper.s3_create_bucket()
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/a.txt", b"aa")
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/b.txt", b"bbb")
        cli_helper.invoke(["dataset", "rm", "ds-20201012", "a.txt"])
        assert cli_helper.s3_list_objects() == ["my-models/datasets/ds-20201012/b.txt"]
