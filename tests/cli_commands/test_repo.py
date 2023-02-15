# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import pytest


class TestRepo:
    def repo(self):
        repo = {
            "CodeRepositoryName": "my-models-repo-1234",
            "GitConfig": {
                "RepositoryUrl": "https://example.com/repo-1234",
                "Branch": "master",
                "SecretArn": "arn:secret:repo-1234",
            },
        }
        return repo

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["repo", "--help"],
            output_startswith=[
                "Usage: ml2p repo [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Describe and list code repositories.",
            ],
        )

    def test_list_empty(self, cli_helper):
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["repo", "list"], output_jsonl=[])
        assert (
            str(err.value)
            == "The list_code_repositories action has not been implemented"
        )

    def test_list(self, cli_helper):
        repo = self.repo()
        with pytest.raises(NotImplementedError) as err:
            cli_helper.sagefaker.create_code_repository(**repo)
        assert (
            str(err.value)
            == "The create_code_repository action has not been implemented"
        )
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["repo", "list"], output_jsonl=[repo])
        assert (
            str(err.value)
            == "The list_code_repositories action has not been implemented"
        )

    def test_describe(self, cli_helper):
        repo = self.repo()
        with pytest.raises(NotImplementedError) as err:
            cli_helper.sagefaker.create_code_repository(**repo)
        assert (
            str(err.value)
            == "The create_code_repository action has not been implemented"
        )
        with pytest.raises(NotImplementedError) as err:
            cli_helper.invoke(["repo", "describe", "repo-1234"], output_jsonl=[repo])
        assert (
            str(err.value)
            == "The describe_code_repository action has not been implemented"
        )
