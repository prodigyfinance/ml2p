# -*- coding: utf-8 -*-
"""Tests for ml2p.errors using Werkzeug HTTPException."""

from werkzeug.exceptions import HTTPException

from ml2p.errors import APIError, ClientError, ServerError


class TestAPIError:
    def test_defaults(self):
        err = APIError(message="Test error")
        assert isinstance(err, HTTPException)
        assert isinstance(err, APIError)
        assert err.message == "Test error"
        assert err.details == []
        assert err.status_code == 500

    def test_details_str(self):
        err = APIError(message="Test error", details="zoom")
        assert err.details == ["zoom"]

    def test_details_list_of_str(self):
        err = APIError(message="Test error", details=["a", "b"])
        assert err.details == ["a", "b"]

    def test_status_code_override(self):
        err = APIError(message="Test error", status_code=501)
        assert err.status_code == 501


class TestClientError:
    def test_defaults(self):
        err = ClientError(message="Test client error")
        assert isinstance(err, HTTPException)
        assert isinstance(err, APIError)
        assert err.message == "Test client error"
        assert err.status_code == 400


class TestServerError:
    def test_defaults(self):
        err = ServerError(message="Test server error")
        assert isinstance(err, HTTPException)
        assert isinstance(err, APIError)
        assert err.message == "Test server error"
        assert err.status_code == 500
