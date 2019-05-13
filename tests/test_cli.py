# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

from click.testing import CliRunner

from ml2p.cli import ml2p


class TestML2P:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(ml2p, ["--help"])
        assert result.exit_code == 0
        assert result.output.splitlines()[:5] == [
            "Usage: ml2p [OPTIONS] COMMAND [ARGS]...",
            "",
            "  Minimal Lovable Machine Learning Pipeline.",
            "",
            "  A friendlier interface to AWS SageMaker.",
        ]
