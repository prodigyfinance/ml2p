# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

import pytest


class TestNotebook:
    def cfg(self):
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
            "notebook": {"instance_type": "ml.t2.medium", "volume_size": 8},
        }
        return cfg

    def cfg_repo_url(self):
        cfg = self.cfg()
        cfg["notebook"].update(
            **{
                "repo_url": "https://example.com/repo-1234",
                "repo_branch": "master",
                "repo_secret_arn": "arn:secret:1234",
            }
        )
        return cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["notebook", "--help"],
            output_startswith=[
                "Usage: ml2p notebook [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and manage notebooks.",
            ],
        )

    def test_list_empty(self, cli_helper):
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["notebook", "list"], output_jsonl=[])
        assert (
            str(err.value)
            == "The list_notebook_instances action has not been implemented"
        )

    def test_create_and_list(self, cli_helper):
        cfg = self.cfg()
        create_output = json.loads(
            cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        )
        assert create_output["NotebookInstanceArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "notebook-instance/my-models-notebook-test"
        )
        assert create_output["ResponseMetadata"]["HTTPStatusCode"] == 200
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["notebook", "list"], output_jsonl=[])
        assert (
            str(err.value)
            == "The list_notebook_instances action has not been implemented"
        )

    def test_create_and_list_with_repo_url(self, cli_helper):
        cfg = self.cfg_repo_url()
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        assert (
            str(err.value)
            == "The create_code_repository action has not been implemented"
        )
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["notebook", "list"], output_jsonl=[])
        assert (
            str(err.value)
            == "The list_notebook_instances action has not been implemented"
        )

    def test_create_and_describe(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        describe_output = json.loads(
            cli_helper.invoke(["notebook", "describe", "notebook-test"])
        )
        assert describe_output["NotebookInstanceArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "notebook-instance/my-models-notebook-test"
        )
        assert describe_output["NotebookInstanceName"] == "my-models-notebook-test"
        assert describe_output["NotebookInstanceStatus"] == "InService"
        assert describe_output["InstanceType"] == "ml.t2.medium"
        assert describe_output["RoleArn"] == "arn:aws:iam::12345:role/role-name"

    def test_create_and_delete(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        delete_output = json.loads(
            cli_helper.invoke(["notebook", "delete", "notebook-test"], cfg=cfg)
        )
        assert delete_output["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_create_and_delete_while_in_service(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        delete_output = json.loads(
            cli_helper.invoke(["notebook", "delete", "notebook-test"], cfg=cfg)
        )
        assert delete_output["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_create_and_delete_with_repo(self, cli_helper):
        cfg = self.cfg_repo_url()
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
            cli_helper.invoke(["notebook", "delete", "notebook-test"], cfg=cfg)
        assert (
            str(err.value)
            == "The create_code_repository action has not been implemented"
        )

    def test_presigned_url(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["notebook", "presigned-url", "notebook-test"], cfg=cfg)
        assert str(err.value) == (
            "The create_presigned_notebook_instance_url "
            "action has not been implemented"
        )

    def test_stop(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        cli_helper.invoke(
            ["notebook", "stop", "notebook-test"],
            output_jsonl=[],
            cfg=cfg,
        )
        describe_output = json.loads(
            cli_helper.invoke(["notebook", "describe", "notebook-test"])
        )
        assert describe_output["NotebookInstanceArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "notebook-instance/my-models-notebook-test"
        )
        assert describe_output["NotebookInstanceName"] == "my-models-notebook-test"
        assert describe_output["NotebookInstanceStatus"] == "Stopped"

    def test_start(self, cli_helper):
        cfg = self.cfg()
        cli_helper.invoke(["notebook", "create", "notebook-test"], cfg=cfg)
        cli_helper.invoke(
            ["notebook", "stop", "notebook-test"],
            output_jsonl=[],
            cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "start", "notebook-test"],
            output_jsonl=[],
            cfg=cfg,
        )
        describe_output = json.loads(
            cli_helper.invoke(["notebook", "describe", "notebook-test"])
        )
        assert describe_output["NotebookInstanceArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "notebook-instance/my-models-notebook-test"
        )
        assert describe_output["NotebookInstanceName"] == "my-models-notebook-test"
        assert describe_output["NotebookInstanceStatus"] == "InService"
