# -*- coding: utf-8 -*-

""" Tests for ml2p.docker. """

from click.testing import CliRunner

from ml2p.docker import ml2p_docker


class TestML2PDocker:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(ml2p_docker, ["--help"])
        assert result.exit_code == 0
