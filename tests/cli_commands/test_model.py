# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

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

    def test_create_and_list(self, cli_helper):
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
                "HTTPHeaders": {"server": "amazon.com"},
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

    def test_create_and_delete(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "mdl-0-1-12"], cfg=cfg)
        delete_output = json.loads(
            cli_helper.invoke(["model", "delete", "mdl-0-1-12"], cfg=cfg)
        )
        assert delete_output == {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {"server": "amazon.com"},
                "RetryAttempts": 0,
            }
        }
