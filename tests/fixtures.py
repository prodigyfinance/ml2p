# -*- coding: utf-8 -*-

""" Pytest fixtures for tests. """

import datetime
import json
import os

import boto3
import moto
import pytest

from ml2p import hyperparameters
from ml2p.core import SageMakerEnv

MOTO_TEST_REGION = "us-east-1"


@pytest.fixture
def fake_utcnow(monkeypatch):
    utcnow = datetime.datetime(2019, 1, 31, 12, 0, 2, tzinfo=datetime.timezone.utc)

    class fake_datetime(datetime.datetime):
        @classmethod
        def utcnow(cls):
            return utcnow

    monkeypatch.setattr(datetime, "datetime", fake_datetime)
    return utcnow


class SageMakerFixture:
    def __init__(self, ml_folder, monkeypatch):
        self.ml_folder = ml_folder
        self.monkeypatch = monkeypatch

    def generic(self):
        return SageMakerEnv(str(self.ml_folder))

    def train(self, **kw):
        self.monkeypatch.setenv("TRAINING_JOB_NAME", "test-train-1.2.3")
        params = {"ML2P_PROJECT": "test-project", "ML2P_S3_URL": "s3://foo/bar"}
        params.update(kw)
        for k, v in kw.items():
            if v is None:
                del params[k]
        self.ml_folder.mkdir("input").mkdir("config").join(
            "hyperparameters.json"
        ).write(json.dumps(hyperparameters.encode({"ML2P_ENV": params})))
        return SageMakerEnv(str(self.ml_folder))

    def serve(self, **kw):
        self.monkeypatch.delenv("TRAINING_JOB_NAME", raising=False)
        envvars = {
            "ML2P_MODEL_VERSION": "test-model-1.2.3",
            "ML2P_PROJECT": "test-project",
            "ML2P_S3_URL": "s3://foo/bar",
        }
        envvars.update(kw)
        for k, v in envvars.items():
            if v is None:
                self.monkeypatch.delenv(k, raising=False)
            else:
                self.monkeypatch.setenv(k, v)
        return SageMakerEnv(str(self.ml_folder))


@pytest.fixture
def sagemaker(tmpdir, monkeypatch):
    return SageMakerFixture(ml_folder=tmpdir, monkeypatch=monkeypatch)


@pytest.fixture
def moto_session(monkeypatch):
    with moto.mock_s3(), moto.mock_ssm():
        for k in list(os.environ):
            if k.startswith("AWS_"):
                monkeypatch.delitem(os.environ, k)
        # The environment variables duplicate what happens when an AWS Lambda
        # is executed. See
        # https://docs.aws.amazon.com/lambda/latest/dg/current-supported-versions.html
        monkeypatch.setitem(os.environ, "AWS_ACCESS_KEY_ID", "dummy-access-key")
        monkeypatch.setitem(
            os.environ, "AWS_SECRET_ACCESS_KEY", "dummy-access-key-secret"
        )
        monkeypatch.setitem(os.environ, "AWS_REGION", MOTO_TEST_REGION)
        yield boto3.Session(region_name=MOTO_TEST_REGION)
