# -*- coding: utf-8 -*-

""" Tests for ml2p.core. """

import pathlib

import pytest

from ml2p.core import (
    Model,
    ModelPredictor,
    ModelTrainer,
    S3URL,
    SageMakerEnv,
    import_string,
)


class TestS3URL:
    def test_bucket(self):
        assert S3URL("s3://bucket/foo").bucket() == "bucket"

    def test_path(self):
        assert S3URL("s3://bucket/foo/").path("bar.txt") == "foo/bar.txt"

    def test_path_with_empty_roo(self):
        assert S3URL("s3://bucket").path("bar.txt") == "bar.txt"

    def test_url(self):
        assert S3URL("s3://bucket/foo/").url("bar.txt") == "s3://bucket/foo/bar.txt"


class TestSageMakerEnv:
    def test_create_env_without_model_version(self, tmpdir, monkeypatch):
        monkeypatch.delenv("ML2P_MODEL_VERSION", raising=False)
        env = SageMakerEnv(str(tmpdir))
        assert env.model_version is None

    def test_create_env_with_model_version(self, sagemaker):
        assert sagemaker.env.model_version == "test-model-1.2.3"

    def test_create_env_without_project_name(self, tmpdir, monkeypatch):
        monkeypatch.delenv("ML2P_PROJECT", raising=False)
        env = SageMakerEnv(str(tmpdir))
        assert env.project is None

    def test_create_env_with_project_name(self, sagemaker):
        assert sagemaker.env.project == "test-project"

    def test_create_env_without_s3_url(self, tmpdir, monkeypatch):
        monkeypatch.delenv("ML2P_S3_URL", raising=False)
        env = SageMakerEnv(str(tmpdir))
        assert env.s3 is None

    def test_create_env_with_s3_url(self, sagemaker):
        assert sagemaker.env.s3.url("baz.txt") == "s3://foo/bar/baz.txt"

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

    def test_train(self, sagemaker):
        trainer = ModelTrainer(sagemaker.env)
        with pytest.raises(NotImplementedError) as exc_info:
            trainer.train()
        assert str(exc_info.value) == "Sub-classes should implement .train()"


class TestModelPredictor:
    def test_create(self, sagemaker):
        predictor = ModelPredictor(sagemaker.env)
        assert predictor.env is sagemaker.env

    def test_setup(self, sagemaker):
        predictor = ModelPredictor(sagemaker.env)
        predictor.setup()

    def test_teardown(self, sagemaker):
        predictor = ModelPredictor(sagemaker.env)
        predictor.teardown()

    def test_invoke_with_result_not_implemented(self, sagemaker):
        predictor = ModelPredictor(sagemaker.env)
        with pytest.raises(NotImplementedError) as exc_info:
            predictor.invoke({})
        assert str(exc_info.value) == "Sub-classes should implement .result(...)"

    def test_invoke_with_result_implemented(self, sagemaker, fake_utcnow):
        class MyPredictor(ModelPredictor):
            def result(self, data):
                return {"probability": 0.5, "input": data["input"]}

        predictor = MyPredictor(sagemaker.env)
        assert predictor.invoke({"input": 1}) == {
            "metadata": {
                "model_version": "test-model-1.2.3",
                "timestamp": 1548936002.0,
            },
            "result": {"probability": 0.5, "input": 1},
        }

    def test_metadata(self, sagemaker, fake_utcnow):
        predictor = ModelPredictor(sagemaker.env)
        assert predictor.metadata({}) == {
            "model_version": "test-model-1.2.3",
            "timestamp": 1548936002.0,
        }

    def test_result(self, sagemaker):
        predictor = ModelPredictor(sagemaker.env)
        with pytest.raises(NotImplementedError) as exc_info:
            predictor.result({})
        assert str(exc_info.value) == "Sub-classes should implement .result(...)"


class TestModel:
    def test_trainer_not_set(self, sagemaker):
        with pytest.raises(ValueError) as exc_info:
            Model().trainer(sagemaker.env)
        assert str(exc_info.value) == ".TRAINER should be an instance of ModelTrainer"

    def test_trainer_set(self, sagemaker):
        class MyModel(Model):
            TRAINER = ModelTrainer

        trainer = MyModel().trainer(sagemaker.env)
        assert trainer.__class__ is ModelTrainer
        assert trainer.env is sagemaker.env

    def test_predictor_not_set(self, sagemaker):
        with pytest.raises(ValueError) as exc_info:
            Model().predictor(sagemaker.env)
        assert (
            str(exc_info.value) == ".PREDICTOR should be an instance of ModelPredictor"
        )

    def test_predictor_set(self, sagemaker):
        class MyModel(Model):
            PREDICTOR = ModelPredictor

        predictor = MyModel().predictor(sagemaker.env)
        assert predictor.__class__ is ModelPredictor
        assert predictor.env is sagemaker.env
