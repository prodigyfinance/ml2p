# -*- coding: utf-8 -*-

""" ML2P exceptions.
"""


class ML2PError(Exception):
    """ The top-level ML2P exception. """


class ML2PDataChannelMissing(ML2PError):
    """ Raised when a data channel is not configured. """


class ML2PDataChannelIncorrectedMode(ML2PError):
    """ Raised when a data channel is not configured with the required mode.
    """
