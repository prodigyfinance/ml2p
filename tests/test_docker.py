# -*- coding: utf-8 -*-

""" Tests for ml2p.docker. """

from click.testing import CliRunner

from ml2p.docker import ml2p_docker


class TestML2PDocker:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(ml2p_docker, ["--help"])
        assert result.exit_code == 0
        assert result.output.splitlines()[:3] == [
            "Usage: ml2p-docker [OPTIONS] COMMAND [ARGS]...",
            "",
            "  ML2P Sagemaker Docker container helper CLI.",
        ]
