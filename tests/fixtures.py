# -*- coding: utf-8 -*-

""" Pytest fixtures for tests. """

import collections

import pytest

from ml2p.core import SageMakerEnv

sagemaker_type = collections.namedtuple("sagemaker_env_type", ["env", "ml_folder"])


@pytest.fixture
def sagemaker(tmpdir, monkeypatch):
    monkeypatch.setenv("ML2P_MODEL_VERSION", "test-model-1.2.3")
    env = SageMakerEnv(str(tmpdir))
    return sagemaker_type(env=env, ml_folder=tmpdir)
