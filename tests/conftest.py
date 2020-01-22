# -*- coding: utf-8 -*-

""" Configuration for pytest. """

from .fixtures import (  # noqa: imported so that pytest can find the fixtures
    fake_utcnow,
    fake_uuid4,
    moto_session,
    sagemaker,
)
