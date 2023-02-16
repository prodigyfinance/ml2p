# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

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
        self.sagefaker = moto_session.client("sagemaker")

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
            cli.ml2p,
            ["--cfg", self.cfg(**cfg)] + args,
            catch_exceptions=False,
        )
        if result.exception is not None:
            raise result.exception.with_traceback(result.exc_info[2])
        assert result.exit_code == exit_code
        if output is not None:
            assert result.output.splitlines() == output
            return None
        if output_startswith is not None:
            assert (
                result.output.splitlines()[: len(output_startswith)]
                == output_startswith
            )
            return None
        if output_jsonl is not None:
            assert result.output == "\n".join(
                [json.dumps(data, indent=2) for data in output_jsonl] + [""]
            )
            return None
        return result.output


@pytest.fixture
def cli_helper(moto_session, tmp_path):
    return CLIHelper(
        moto_session,
        tmp_path,
        bucket="my-bucket",
        base_cfg={"project": "my-models", "s3folder": "s3://my-bucket/my-models/"},
    )


class TestModellingProjectWithSagemakerClient:
    def test_create(self, cli_helper):
        prj = cli.ModellingProjectWithSagemakerClient(cli_helper.cfg())
        assert type(prj.client).__name__ == "SageMaker"


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
