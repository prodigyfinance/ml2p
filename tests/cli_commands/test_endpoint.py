# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

import pytest


class TestEndpoint:
    def cfg(self):
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
            "deploy": {"instance_type": "ml.t2.medium"},
        }
        return cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["endpoint", "--help"],
            output_startswith=[
                "Usage: ml2p endpoint [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and inspect endpoints.",
            ],
        )

    def test_list_empty(self, cli_helper):
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["endpoint", "list"], output_jsonl=[])
        assert str(err.value) == "The list_endpoints action has not been implemented"

    def test_create_and_list(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "endpoint-0-1-12"], cfg=cfg)
        create_output = json.loads(
            cli_helper.invoke(["endpoint", "create", "endpoint-0-1-12"], cfg=cfg)
        )
        assert create_output["EndpointArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "endpoint/my-models-endpoint-0-1-12"
        )
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["endpoint", "list"], output_jsonl=[])
        assert str(err.value) == "The list_endpoints action has not been implemented"

    def test_create_and_describe(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "endpoint-0-1-12"], cfg=cfg)
        cli_helper.invoke(["endpoint", "create", "endpoint-0-1-12"], cfg=cfg)
        describe_output = json.loads(
            cli_helper.invoke(["endpoint", "describe", "endpoint-0-1-12"])
        )
        assert describe_output["EndpointName"] == "my-models-endpoint-0-1-12"
        assert describe_output["EndpointArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "endpoint/my-models-endpoint-0-1-12"
        )
        assert describe_output["EndpointConfigName"] == (
            "my-models-endpoint-0-1-12-config"
        )
        assert describe_output["EndpointStatus"] == "InService"

    def test_create_and_delete(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "endpoint-0-1-12"], cfg=cfg)
        cli_helper.invoke(["endpoint", "create", "endpoint-0-1-12"], cfg=cfg)
        delete_output = cli_helper.invoke(
            ["endpoint", "delete", "endpoint-0-1-12"], cfg=cfg
        )
        delete_output = json.loads("[" + delete_output.replace("}\n{", "},{") + "]")
        assert delete_output[0]["ResponseMetadata"]["HTTPStatusCode"] == 200
        assert delete_output[1]["ResponseMetadata"]["HTTPStatusCode"] == 200

    @pytest.mark.skip(
        reason=("Currently we cannot mock the 'sagemaker-runtime' client with moto")
    )
    def test_create_and_invoke(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "endpoint-0-1-12"], cfg=cfg)
        cli_helper.invoke(["endpoint", "create", "endpoint-0-1-12"], cfg=cfg)
        cli_helper.invoke(
            ["endpoint", "invoke", "endpoint-0-1-12", json.dumps({"j": "son"})],
            output_jsonl=[{"Body": {"inputs": {"j": "son"}}}],
            cfg=cfg,
        )

    def test_create_and_wait(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["model", "create", "endpoint-0-1-12"], cfg=cfg)
        cli_helper.invoke(["endpoint", "create", "endpoint-0-1-12"], cfg=cfg)
        cli_helper.invoke(["endpoint", "wait", "endpoint-0-1-12"], output=[], cfg=cfg)
