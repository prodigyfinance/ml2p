# -*- coding: utf-8 -*-

"""ML2P exceptions without relying on Flask-API."""

from werkzeug.exceptions import HTTPException


class NamingError(Exception):
    """Raised when a training job, model, or endpoint name does not follow
    convention.
    """


class ConfigError(Exception):
    """Raised when the configuration in the ml2p project file is invalid."""


class LocalEnvError(Exception):
    """Raised when an error specific to working with a local environment occurs."""


class APIError(HTTPException):
    """Raised when an error occurs in the ML2P prediction API.

    :param str message:
        An error message.
    :param details:
        None, str or list of str
    :param status_code:
        The HTTP status code associated with the exception.
        Defaults to 500.
    """

    code = 500

    def __init__(self, message, details=None, status_code=None):
        # Initialize HTTPException with description
        super().__init__(description=message)
        if details is None:
            details = []
        elif isinstance(details, str):
            details = [details]
        self.details = details
        if status_code is not None:
            self.code = status_code
        self.message = message

    @property
    def status_code(self):
        return self.code

    def get_body(self, environ=None):
        # Response body as JSON-like dict
        return {"message": self.message, "details": self.details}

    def get_headers(self, environ=None):
        # Ensure JSON content type
        return [("Content-Type", "application/json")]


class ServerError(APIError):
    """Raised when the HTTP server fails while trying to process a request."""

    code = 500


class ClientError(APIError):
    """Raised when an HTTP client invokes a prediction endpoint with
    invalid inputs.
    """

    code = 400
