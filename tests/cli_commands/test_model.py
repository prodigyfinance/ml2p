# -*- coding: utf-8 -*-

"""Tests for ml2p.cli."""

import json


class TestModel:
    def cfg(self):
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            }
        }
        return cfg

    def cfg_multimodels(self):
        cfg = {
            "models": {
                "model-type-1": {
                    "model-0-0-1": {
                        "training_job": "model-0-0-1",
                        "image_tag": "0.0.1",
                        "cls": "my.pkg.module.model",
                    },
                    "model-0-0-2": {
                        "training_job": "0-2-0",
                        "image_tag": "0.0.1-updated",
                        "cls": "my.pkg.module.model",
                    },
                },
                "model-type-2": {
                    "model-defaults": {"cls": "my.pkg.module.modeltwo"},
                    "model-0-0-1": {
                        "training_job": "test-repo-model-0-0-1",
                        "image_tag": "0.0.1",
                    },
                    "model-0-0-2": {
                        "training_job": "test-repo-model-0-0-2",
                        "image_tag": "0.0.2",
                        "cls": "my.pkg.module.submodule.modeltwo",
                    },
                },
            },
            "defaults": {
                "image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
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
        assert (
            model_output["ModelArn"]
            == "arn:aws:sagemaker:us-east-1:123456789012:model/my-models-mdl-0-1-12"
        )
        assert model_output["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert model_output["ResponseMetadata"]["RetryAttempts"] == 0
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
        assert delete_output["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert delete_output["ResponseMetadata"]["RetryAttempts"] == 0

    def test_create_mutlimodel_and_list(self, cli_helper, fake_utcnow):
        cfg = self.cfg_multimodels()
        model_output = json.loads(
            cli_helper.invoke(
                [
                    "model",
                    "create",
                    "model-0-0-1",
                    "-m",
                    "model-type-1",
                ],
                cfg=cfg,
            )
        )
        assert model_output["ModelArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:model/" "my-models-model-0-0-1"
        )
        assert model_output["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert model_output["ResponseMetadata"]["RetryAttempts"] == 0

    def test_create_mutlimodel_and_list_second_model(self, cli_helper, fake_utcnow):
        cfg = self.cfg_multimodels()
        model_output = json.loads(
            cli_helper.invoke(
                [
                    "model",
                    "create",
                    "model-type-two-0-0-1",
                    "-m",
                    "model-type-2",
                ],
                cfg=cfg,
            )
        )
        assert model_output["ModelArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:model/"
            "my-models-model-type-two-0-0-1"
        )
        assert model_output["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert model_output["ResponseMetadata"]["RetryAttempts"] == 0

    def test_multimodel_create_and_describe(self, cli_helper):
        cfg = self.cfg_multimodels()
        cli_helper.invoke(
            [
                "model",
                "create",
                "multi-model-0-0-1",
                "-m",
                "model-type-1",
            ],
            cfg=cfg,
        )
        describe_output = json.loads(
            cli_helper.invoke(["model", "describe", "multi-model-0-0-1"])
        )
        assert describe_output["ModelName"] == "my-models-multi-model-0-0-1"
        assert describe_output["Containers"] == [
            {
                "ContainerHostname": "model-0-0-1",
                "Image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.1",
                "ModelDataUrl": (
                    "s3://my-bucket/my-models/models/my-models-model-0-0-1/"
                    "output/model.tar.gz"
                ),
                "Environment": {
                    "ML2P_MODEL_VERSION": "my-models-model-0-0-1",
                    "ML2P_PROJECT": "my-models",
                    "ML2P_S3_URL": "s3://my-bucket/my-models/",
                    "ML2P_MODEL_CLS": "my.pkg.module.model",
                },
            },
            {
                "ContainerHostname": "model-0-0-2",
                "Image": (
                    "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:"
                    "0.0.1-updated"
                ),
                "ModelDataUrl": (
                    "s3://my-bucket/my-models/models/my-models-0-2-0/"
                    "output/model.tar.gz"
                ),
                "Environment": {
                    "ML2P_MODEL_VERSION": "my-models-model-0-0-2",
                    "ML2P_PROJECT": "my-models",
                    "ML2P_S3_URL": "s3://my-bucket/my-models/",
                    "ML2P_MODEL_CLS": "my.pkg.module.model",
                },
            },
        ]
        assert (
            describe_output["ExecutionRoleArn"] == "arn:aws:iam::12345:role/role-name"
        )

    def test_multimodel_create_and_delete(self, cli_helper, fake_utcnow):
        cfg = self.cfg_multimodels()
        cli_helper.invoke(
            [
                "model",
                "create",
                "multi-model-0-0-1",
                "-m",
                "model-type-1",
            ],
            cfg=cfg,
        )
        delete_output = json.loads(
            cli_helper.invoke(["model", "delete", "multi-model-0-0-1"], cfg=cfg)
        )
        assert delete_output["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert delete_output["ResponseMetadata"]["RetryAttempts"] == 0
