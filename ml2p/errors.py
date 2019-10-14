# -*- coding: utf-8 -*-

""" ML2P exceptions.
"""

from flask_api.exceptions import APIException


class NamingError(Exception):
    """ Raised when a training job, model, or endpoint name does not follow
        convention.
    """


class APIError(APIException):
    """ Raised when an error occurs in the ML2P prediction API.

        :param str message:
            An error message.
        :type errors:
            None, str or list of str
        :param errors:
            Details of the errors that occurred.
    """

    status_code = 500

    def __init__(self, message, errors=None):
        if errors is None:
            errors = message
        if isinstance(errors, str):
            errors = [errors]
        self.message = message
        self.errors = errors


class ServerError(APIError):
    """ Raised when the HTTP server fails while trying to process a request.
    """

    status_code = 500


class ClientError(APIError):
    """ Raised when an HTTP client invokes a prediction endpoint with
        invalid inputs.
    """

    status_code = 400
