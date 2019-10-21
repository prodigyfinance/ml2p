# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

from click.testing import CliRunner

from ml2p import __version__ as ml2p_version
from ml2p.cli import ml2p, ModellingSubCfg


def mk_subcfg(defaults="defaults"):
    return ModellingSubCfg(
        {"sub": {"a": 1, "b": "boo"}, "defaults": {"c": 3}}, "sub", defaults=defaults
    )


class TestModellingSubCfg:
    def test_getattr(self):
        subcfg = mk_subcfg()
        assert subcfg.a == 1
        assert subcfg.c == 3

    def test_getitem(self):
        subcfg = mk_subcfg()
        assert subcfg["a"] == 1
        assert subcfg["c"] == 3

    def test_setitem(self):
        subcfg = mk_subcfg()
        subcfg["d"] = 5
        assert subcfg.d == 5
        assert subcfg["d"] == 5

    def test_keys(self):
        subcfg = mk_subcfg()
        assert subcfg.keys() == ["a", "b", "c"]

    def test_get(self):
        subcfg = mk_subcfg()
        assert subcfg.get("a") == 1
        assert subcfg.get("d") is None
        assert subcfg.get("d", 3) == 3


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

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(ml2p, ["--version"])
        assert result.exit_code == 0
        assert result.output.splitlines() == ["ml2p, version {}".format(ml2p_version)]
