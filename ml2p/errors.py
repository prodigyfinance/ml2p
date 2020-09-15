# -*- coding: utf-8 -*-

""" ML2P exceptions.
"""

from flask_api.exceptions import APIException


class NamingError(Exception):
    """ Raised when a training job, model, or endpoint name does not follow
        convention.
    """


class ConfigError(Exception):
    """ Raised when the configuration in the ml2p project file is invalid.
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
        :type status_code:
            None or int
        :param status_code:
            The HTTP status code associated with the exception.
            Defaults to the `status_code` attribute of the
            exception class, which is 500 for this base exception class.
    """

    status_code = 500

    def __init__(self, message, details=None, status_code=None):
        if details is None:
            details = []
        elif isinstance(details, str):
            details = [details]
        if status_code is None:
            status_code = self.__class__.status_code
        self.message = message
        self.details = details
        self.status_code = status_code


class ServerError(APIError):
    """ Raised when the HTTP server fails while trying to process a request.
    """

    status_code = 500


class ClientError(APIError):
    """ Raised when an HTTP client invokes a prediction endpoint with
        invalid inputs.
    """

    status_code = 400
