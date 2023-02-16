# -*- coding: utf-8 -*-

""" Pytest fixtures for tests. """

import datetime
import json
import os
import pathlib
import uuid

import boto3
import moto
import pytest
import yaml

from ml2p.core import LocalEnv, SageMakerEnv

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


@pytest.fixture
def fake_uuid4(monkeypatch):
    uuid4_constant = uuid.UUID(
        bytes=b" \xf8\x03\xc4\xf1UF\x9e\xba\x96\x14\xca\xa3\n\xf9\xe1"
    )

    def fake_uuid4():
        return uuid4_constant

    monkeypatch.setattr(uuid, "uuid4", fake_uuid4)
    return uuid4_constant


class SageMakerFixture:
    def __init__(self, ml_folder, monkeypatch, moto_session):
        self.ml_folder = ml_folder
        self.monkeypatch = monkeypatch
        self.moto_session = moto_session
        self.s3 = moto_session.client("s3")

    def s3_create_bucket(self, bucket):
        self.s3.create_bucket(Bucket=bucket)

    def s3_get_object(self, bucket, s3_key):
        response = self.s3.get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(response["Body"].read())
        return data

    def s3_put_object(self, bucket, s3_key, data):
        body = json.dumps(data)
        self.s3.put_object(Bucket=bucket, Key=s3_key, Body=body)

    def s3_put_bytes(self, bucket, s3_key, data):
        self.s3.put_object(Bucket=bucket, Key=s3_key, Body=data)

    def generic(self):
        return SageMakerEnv(str(self.ml_folder))

    def train(self, **kw):
        envvars = {
            "ML2P_TRAINING_JOB": "test-train-1.2.3",
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

    def serve(self, **kw):
        self.s3_create_bucket("foo")
        self.monkeypatch.delenv("ML2P_TRAINING_JOB", raising=False)
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

    def dataset(self, **kw):
        self.s3_create_bucket("foo")
        envvars = {
            "ML2P_DATASET": "test-dataset-20220112",
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

    def local(self, **kw):
        self.s3_create_bucket("foo")
        cfg_file = self.ml_folder / "ml2p.yaml"
        with cfg_file.open("w") as f:
            cfg = {"project": "test-project", "s3folder": "s3://foo/bar"}
            yaml.safe_dump(cfg, f)
        ml_folder = kw.pop("ml_folder", self.ml_folder)
        session = kw.pop("session", self.moto_session)
        return LocalEnv(str(ml_folder), str(cfg_file), session)


@pytest.fixture
def sagemaker(tmpdir, monkeypatch, moto_session):
    return SageMakerFixture(
        ml_folder=tmpdir, monkeypatch=monkeypatch, moto_session=moto_session
    )


@pytest.fixture
def moto_session(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AWS_"):
            monkeypatch.delitem(os.environ, k)
    # The environment variables duplicate what happens when an AWS Lambda
    # is executed. See
    # https://docs.aws.amazon.com/lambda/latest/dg/current-supported-versions.html
    monkeypatch.setitem(os.environ, "AWS_ACCESS_KEY_ID", "dummy-access-key")
    monkeypatch.setitem(os.environ, "AWS_SECRET_ACCESS_KEY", "dummy-access-key-secret")
    monkeypatch.setitem(os.environ, "AWS_SECURITY_TOKEN", "dummy-security-token")
    monkeypatch.setitem(os.environ, "AWS_SESSION_TOKEN", "dummy-session-token")
    monkeypatch.setitem(os.environ, "AWS_DEFAULT_REGION", MOTO_TEST_REGION)
    with moto.mock_s3(), moto.mock_ssm(), moto.mock_sagemaker():
        yield boto3.Session(region_name=MOTO_TEST_REGION)


@pytest.fixture()
def data_fixtures():
    """Load a data fixture."""
    return pathlib.Path(__file__).parent / "data"
