# -*- coding: utf-8 -*-

""" Tests for ml2p.cli_commands.init. """


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
