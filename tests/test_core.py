# -*- coding: utf-8 -*-

""" Tests for ml2p.core. """

import collections
import pathlib

import pytest

from ml2p.core import ModelTrainer, ModelPredictor, SageMakerEnv, import_string

sagemaker_type = collections.namedtuple("sagemaker_env_type", ["env", "ml_folder"])


@pytest.fixture
def sagemaker(tmpdir, monkeypatch):
    monkeypatch.setenv("ML2P_MODEL_VERSION", "test-model-1.2.3")
    env = SageMakerEnv(str(tmpdir))
    return sagemaker_type(env=env, ml_folder=tmpdir)


class TestSageMakerEnv:
    def test_create_env_without_model_version(self, tmpdir, monkeypatch):
        monkeypatch.delenv("ML2P_MODEL_VERSION", raising=False)
        env = SageMakerEnv(str(tmpdir))
        assert env.model_version == "Unknown"

    def test_create_env_with_model_version(self, sagemaker):
        assert sagemaker.env.model_version == "test-model-1.2.3"

    def test_hyperparameters(self, sagemaker):
        sagemaker.ml_folder.mkdir("input").mkdir("config").join(
            "hyperparameters.json"
        ).write('{"param": "value"}')
        assert sagemaker.env.hyperparameters() == {"param": "value"}

    def test_resourceconfig(self, sagemaker):
        sagemaker.ml_folder.mkdir("input").mkdir("config").join(
            "resourceconfig.json"
        ).write('{"config": "value"}')
        assert sagemaker.env.resourceconfig() == {"config": "value"}

    def test_dataset_folder(self, sagemaker):
        assert sagemaker.env.dataset_folder("foo") == pathlib.Path(
            str(sagemaker.ml_folder.join("input/data/foo"))
        )

    def test_model_folder(self, sagemaker):
        assert sagemaker.env.model_folder() == pathlib.Path(
            str(sagemaker.ml_folder.join("model"))
        )

    def test_write_failure(self, sagemaker):
        failure_path = sagemaker.ml_folder.mkdir("output").join("failure")
        text = "\n".join(["BadModel", "no biscuit"])
        sagemaker.env.write_failure(text)
        assert failure_path.read() == text


class TestImportString:
    def test_import_string(self):
        cls = import_string("tests.test_core.TestImportString")
        assert cls is TestImportString


class TestModelTrainer:
    def test_create(self, sagemaker):
        trainer = ModelTrainer(sagemaker.env)
        assert trainer.env is sagemaker.env


class TestModelPredictor:
    def test_create(self, sagemaker):
        predictor = ModelPredictor(sagemaker.env)
        assert predictor.env is sagemaker.env
