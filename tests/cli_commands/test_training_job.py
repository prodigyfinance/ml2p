# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json


class TestTrainingJob:
    def cfg(self):
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
            "train": {"instance_type": "ml.m5.large"},
        }
        return cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["training-job", "--help"],
            output_startswith=[
                "Usage: ml2p training-job [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and inspect training jobs.",
            ],
        )

    def test_list_empty(self, cli_helper):
        cli_helper.invoke(
            ["training-job", "list"],
            output_jsonl=[],
        )

    def test_create_and_list(self, cli_helper):
        cfg = self.cfg()
        create_output = json.loads(
            cli_helper.invoke(
                ["training-job", "create", "tj-0-1-11", "ds-20201012"], cfg=cfg
            )
        )
        assert create_output["TrainingJobArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:training-job/my-models-tj-0-1-11"
        )
        list_output = json.loads(cli_helper.invoke(["training-job", "list"]))
        assert list_output["TrainingJobName"] == "my-models-tj-0-1-11"
        assert list_output["TrainingJobStatus"] == "Completed"

    def test_create_and_describe(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(
            ["training-job", "create", "tj-0-1-11", "ds-20201012"], cfg=cfg
        )
        describe_output = json.loads(
            cli_helper.invoke(["training-job", "describe", "tj-0-1-11"])
        )
        assert describe_output["TrainingJobName"] == "my-models-tj-0-1-11"
        assert describe_output["TrainingJobArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:training-job/my-models-tj-0-1-11"
        )
        assert describe_output["RoleArn"] == "arn:aws:iam::12345:role/role-name"

    def test_create_and_wait(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(
            ["training-job", "create", "tj-0-1-11", "ds-20201012"], cfg=cfg
        )
        cli_helper.invoke(["training-job", "wait", "tj-0-1-11"], output=[])
