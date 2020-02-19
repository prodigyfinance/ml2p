# -*- coding: utf-8 -*-

""" ML2P exceptions.
"""

from flask_api.exceptions import APIException


class NamingError(Exception):
    """ Raised when a training job, model, or endpoint name does not follow
        convention.
    """


class LocalEnvError(Exception):
    """ Raised when an error specific to working with a local environment occurs.
    """


class APIError(APIException):
    """ Raised when an error occurs in the ML2P prediction API.

        :param str message:
            An error message.
        :type details:
            None, str or list of str
        :param details:
            Details of the errors that occurred.
    """

    status_code = 500

    def __init__(self, message, details=None):
        if details is None:
            details = []
        elif isinstance(details, str):
            details = [details]
        self.message = message
        self.details = details


class ServerError(APIError):
    """ Raised when the HTTP server fails while trying to process a request.
    """

    status_code = 500


class ClientError(APIError):
    """ Raised when an HTTP client invokes a prediction endpoint with
        invalid inputs.
    """

    status_code = 400
