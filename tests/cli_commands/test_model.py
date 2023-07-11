# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

from pkg_resources import resource_filename
from pytest import fixture


@fixture
def fake_multimodel_cfg():
    return resource_filename("tests.fixture_files", "multimodel-cfg.yml")


class TestModel:
    def cfg(self):
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            }
        }
        return cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["model", "--help"],
            output_startswith=[
                "Usage: ml2p model [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and inspect models.",
            ],
        )

    def test_list_empty(self, cli_helper):
        cli_helper.invoke(["model", "list"], output_jsonl=[])

    def test_create_and_list(self, cli_helper, fake_utcnow):
        cfg = self.cfg()
        model_output = json.loads(
            cli_helper.invoke(["model", "create", "mdl-0-1-12"], cfg=cfg)
        )
        assert model_output == {
            "ModelArn": (
                "arn:aws:sagemaker:us-east-1:123456789012:model/my-models-mdl-0-1-12"
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "server": "amazon.com",
                    "date": "Thu, 31 Jan 2019 12:00:02 GMT",
                },
                "RetryAttempts": 0,
            },
        }
        list_output = json.loads(cli_helper.invoke(["model", "list"]))
        assert list_output["ModelName"] == "my-models-mdl-0-1-12"
        assert list_output["ModelArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:model/my-models-mdl-0-1-12"
        )

    def test_create_and_describe(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "mdl-0-1-12"], cfg=cfg)
        describe_output = json.loads(
            cli_helper.invoke(["model", "describe", "mdl-0-1-12"])
        )
        assert describe_output["ModelName"] == "my-models-mdl-0-1-12"
        assert describe_output["PrimaryContainer"]["Image"] == (
            "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2"
        )
        assert (
            describe_output["ExecutionRoleArn"] == "arn:aws:iam::12345:role/role-name"
        )

    def test_create_and_delete(self, cli_helper, fake_utcnow):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "mdl-0-1-12"], cfg=cfg)
        delete_output = json.loads(
            cli_helper.invoke(["model", "delete", "mdl-0-1-12"], cfg=cfg)
        )
        assert delete_output == {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "server": "amazon.com",
                    "date": "Thu, 31 Jan 2019 12:00:02 GMT",
                },
                "RetryAttempts": 0,
            }
        }

    def test_create_mutlimodel_and_list(
        self, cli_helper, fake_multimodel_cfg, fake_utcnow
    ):
        cfg = self.cfg()
        model_output = json.loads(
            cli_helper.invoke(
                ["model", "create-multi", "test-multimodel-0-0-1", fake_multimodel_cfg],
                cfg=cfg,
            )
        )
        assert model_output == {
            "ModelArn": (
                "arn:aws:sagemaker:us-east-1:123456789012:model/my-models-test-"
                "multimodel-0-0-1"
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "server": "amazon.com",
                    "date": "Thu, 31 Jan 2019 12:00:02 GMT",
                },
                "RetryAttempts": 0,
            },
        }

    def test_create_mutlimodel_and_list_second_model(
        self, cli_helper, fake_multimodel_cfg, fake_utcnow
    ):
        cfg = self.cfg()
        model_output = json.loads(
            cli_helper.invoke(
                [
                    "model",
                    "create-multi",
                    "test-multimodeltwo-0-0-1",
                    fake_multimodel_cfg,
                ],
                cfg=cfg,
            )
        )
        assert model_output == {
            "ModelArn": (
                "arn:aws:sagemaker:us-east-1:123456789012:model/my-models-test-"
                "multimodeltwo-0-0-1"
            ),
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "server": "amazon.com",
                    "date": "Thu, 31 Jan 2019 12:00:02 GMT",
                },
                "RetryAttempts": 0,
            },
        }

    def test_multimodel_create_and_describe(self, cli_helper, fake_multimodel_cfg):
        cfg = self.cfg()
        cli_helper.invoke(
            ["model", "create-multi", "test-multimodel-0-0-1", fake_multimodel_cfg],
            cfg=cfg,
        )
        describe_output = json.loads(
            cli_helper.invoke(["model", "describe", "test-multimodel-0-0-1"])
        )
        assert describe_output["ModelName"] == "my-models-test-multimodel-0-0-1"
        assert describe_output["Containers"] == [
            {
                "ContainerHostname": "model-0-0-1",
                "Image": "123.ecr.com/repo:0.0.1",
                "ModelDataUrl": "s3://my-bucket/my-models/models/my-models-test-repo-"
                "model-0-0-1/output/model.tar.gz",
                "Environment": {
                    "ML2P_MODEL_VERSION": "my-models-model-0-0-1",
                    "ML2P_PROJECT": "my-models",
                    "ML2P_S3_URL": "s3://my-bucket/my-models/",
                    "ML2P_MODEL_CLS": "test_repo.ml2p.MultiModelML2P",
                },
            },
            {
                "ContainerHostname": "model-0-0-2",
                "Image": "123.ecr.com/repo:0.0.1",
                "ModelDataUrl": "s3://my-bucket/my-models/models/my-models-test-repo-"
                "model-0-0-2/output/model.tar.gz",
                "Environment": {
                    "ML2P_MODEL_VERSION": "my-models-model-0-0-2",
                    "ML2P_PROJECT": "my-models",
                    "ML2P_S3_URL": "s3://my-bucket/my-models/",
                    "ML2P_MODEL_CLS": "test_repo.ml2p.extra_dir.MultiModelML2P",
                },
            },
        ]
        assert (
            describe_output["ExecutionRoleArn"] == "arn:aws:iam::12345:role/role-name"
        )

    def test_multimodel_create_and_delete(
        self, cli_helper, fake_multimodel_cfg, fake_utcnow
    ):
        cfg = self.cfg()
        cli_helper.invoke(
            ["model", "create-multi", "test-multimodel-0-0-1", fake_multimodel_cfg],
            cfg=cfg,
        )
        delete_output = json.loads(
            cli_helper.invoke(["model", "delete", "test-multimodel-0-0-1"], cfg=cfg)
        )
        assert delete_output == {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "server": "amazon.com",
                    "date": "Thu, 31 Jan 2019 12:00:02 GMT",
                },
                "RetryAttempts": 0,
            }
        }
