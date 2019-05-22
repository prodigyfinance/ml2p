# -*- coding: utf-8 -*-

""" Tests for ml2p.core. """

import pathlib

import pytest

from ml2p.core import SageMakerEnv


@pytest.fixture
def ml_folder(tmpdir):
    return tmpdir


class TestSageMakerEnv:
    def test_create_env_without_model_version(self, ml_folder):
        env = SageMakerEnv(str(ml_folder))
        assert env.model_version == "Unknown"

    def test_create_env_with_model_version(self, ml_folder, monkeypatch):
        monkeypatch.setenv("ML2P_MODEL_VERSION", "mymodel-1.2.3")
        env = SageMakerEnv(str(ml_folder))
        assert env.model_version == "mymodel-1.2.3"

    def test_hyperparameters(self, ml_folder):
        ml_folder.mkdir("input").mkdir("config").join("hyperparameters.json").write(
            '{"param": "value"}'
        )
        env = SageMakerEnv(str(ml_folder))
        assert env.hyperparameters() == {"param": "value"}

    def test_resourceconfig(self, ml_folder):
        ml_folder.mkdir("input").mkdir("config").join("resourceconfig.json").write(
            '{"config": "value"}'
        )
        env = SageMakerEnv(str(ml_folder))
        assert env.resourceconfig() == {"config": "value"}

    def test_dataset_folder(self, ml_folder):
        env = SageMakerEnv(str(ml_folder))
        assert env.dataset_folder("foo") == pathlib.Path(
            str(ml_folder.join("input/data/foo"))
        )

    def test_model_folder(self, ml_folder):
        env = SageMakerEnv(str(ml_folder))
        assert env.model_folder() == pathlib.Path(str(ml_folder.join("model")))

    def test_write_failure(self, ml_folder):
        failure_path = ml_folder.mkdir("output").join("failure")
        text = "\n".join(["BadModel", "no biscuit"])
        env = SageMakerEnv(str(ml_folder))
        env.write_failure(text)
        assert failure_path.read() == text
