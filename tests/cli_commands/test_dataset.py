# -*- coding: utf-8 -*-

""" Tests for ml2p.cli_commands.dataset. """

import json


class TestDataset:
    def cfg(self):
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.us-east-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
            "dataset": {"instance_type": "ml.t2.medium", "volume_size": 8},
        }
        return cfg

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

    def test_generate(self, cli_helper):
        cfg = self.cfg()
        cli_helper.s3_create_bucket()
        generate_output = json.loads(
            cli_helper.invoke(["dataset", "generate", "ds-20201012"], cfg=cfg)
        )
        assert generate_output["ProcessingJobArn"] == (
            "arn:aws:sagemaker:us-east-1:123456789012"
            ":processing-job/my-models-ds-20201012"
        )
        assert generate_output["ResponseMetadata"]["HTTPStatusCode"] == 200
