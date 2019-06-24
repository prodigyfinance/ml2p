# -*- coding: utf-8 -*-

""" Pytest fixtures for tests. """

import collections
import datetime

import pytest

from ml2p.core import SageMakerEnv


@pytest.fixture
def fake_utcnow(monkeypatch):
    utcnow = datetime.datetime(2019, 1, 31, 12, 0, 2, tzinfo=datetime.timezone.utc)

    class fake_datetime(datetime.datetime):
        @classmethod
        def utcnow(cls):
            return utcnow

    monkeypatch.setattr(datetime, "datetime", fake_datetime)
    return utcnow


sagemaker_type = collections.namedtuple("sagemaker_env_type", ["env", "ml_folder"])


@pytest.fixture
def sagemaker(tmpdir, monkeypatch):
    monkeypatch.setenv("ML2P_MODEL_VERSION", "test-model-1.2.3")
    monkeypatch.setenv("ML2P_PROJECT", "test-project")
    monkeypatch.setenv("ML2P_S3_URL", "s3://foo/bar")
    env = SageMakerEnv(str(tmpdir))
    return sagemaker_type(env=env, ml_folder=tmpdir)
