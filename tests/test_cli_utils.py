# -*- coding: utf-8 -*-

""" Tests for ml2p.cli_utils. """

import pytest

import datetime

from ml2p import cli_utils


class TestCliUtils:
    def test_date_to_string_serializer(self):
        value = datetime.datetime(1, 1, 1)
        assert cli_utils.date_to_string_serializer(value) == "0001-01-01 00:00:00"
        with pytest.raises(TypeError) as exc_info:
            cli_utils.date_to_string_serializer("")
        assert str(exc_info.value) == ""

    def test_click_echo_json(self, capsys):
        response = {"NotebookInstanceName": "notebook-1"}
        cli_utils.click_echo_json(response)
        assert (
            capsys.readouterr().out == '{\n  "NotebookInstanceName": "notebook-1"\n}\n'
        )

    def test_endpoint_url_for_arn(self):
        return
