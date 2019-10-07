# -*- coding: utf-8 -*-

""" Tests for ml2p.core. """

import pathlib

import pytest

from ml2p.core import (
    S3URL,
    Model,
    ModelPredictor,
    ModelTrainer,
    NamingError,
    import_string,
    validate_name,
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

    def test_url_with_no_suffix(self):
        assert S3URL("s3://bucket/foo/").url() == "s3://bucket/foo/"


class TestSageMakerEnvTrain:
    def test_basic_env(self, sagemaker):
        env = sagemaker.train()
        assert env.env_type == env.TRAIN
        assert env.training_job_name == "test-train-1.2.3"
        assert env.model_version is None
        assert env.project == "test-project"

    def test_create_env_without_project_name(self, sagemaker):
        env = sagemaker.train(ML2P_PROJECT=None)
        assert env.project is None

    def test_create_env_without_s3_url(self, sagemaker):
        env = sagemaker.train(ML2P_S3_URL=None)
        assert env.s3 is None

    def test_create_env_with_s3_url(self, sagemaker):
        assert sagemaker.train().s3.url("baz.txt") == "s3://foo/bar/baz.txt"

    def test_create_env_without_model_cls(self, sagemaker):
        env = sagemaker.train(ML2P_MODEL_CLS=None)
        assert env.model_cls is None

    def test_create_env_with_model_cls(self, sagemaker):
        env = sagemaker.train(ML2P_MODEL_CLS="my.pkg.model")
        assert env.model_cls == "my.pkg.model"


class TestSageMakerEnvServe:
    def test_basic_env(self, sagemaker):
        env = sagemaker.serve()
        assert env.env_type == env.SERVE
        assert env.model_version == "test-model-1.2.3"
        assert env.training_job_name is None
        assert env.project == "test-project"

    def test_create_env_without_model_version(self, sagemaker):
        env = sagemaker.serve(ML2P_MODEL_VERSION=None)
        assert env.model_version is None

    def test_create_env_without_project_name(self, sagemaker):
        env = sagemaker.serve(ML2P_PROJECT=None)
        assert env.project is None

    def test_create_env_without_s3_url(self, sagemaker):
        env = sagemaker.serve(ML2P_S3_URL=None)
        assert env.s3 is None

    def test_create_env_with_s3_url(self, sagemaker):
        assert sagemaker.serve().s3.url("baz.txt") == "s3://foo/bar/baz.txt"

    def test_create_env_without_model_cls(self, sagemaker):
        env = sagemaker.serve(ML2P_MODEL_CLS=None)
        assert env.model_cls is None

    def test_create_env_with_model_cls(self, sagemaker):
        env = sagemaker.serve(ML2P_MODEL_CLS="my.pkg.model")
        assert env.model_cls == "my.pkg.model"


class TestSageMakerEnvGeneric:
    def test_hyperparameters(self, sagemaker):
        sagemaker.ml_folder.mkdir("input").mkdir("config").join(
            "hyperparameters.json"
        ).write('{"param": "\\"value\\""}')
        assert sagemaker.generic().hyperparameters() == {"param": "value"}

    def test_nested_hyperparameters(self, sagemaker):
        sagemaker.ml_folder.mkdir("input").mkdir("config").join(
            "hyperparameters.json"
        ).write('{"a.b": "1", "a.c": "2"}')
        assert sagemaker.generic().hyperparameters() == {"a": {"b": 1, "c": 2}}

    def test_missing_hyperparameters_file(self, sagemaker):
        assert sagemaker.generic().hyperparameters() == {}

    def test_resourceconfig(self, sagemaker):
        sagemaker.ml_folder.mkdir("input").mkdir("config").join(
            "resourceconfig.json"
        ).write('{"config": "value"}')
        assert sagemaker.generic().resourceconfig() == {"config": "value"}

    def test_missing_resourceconfig_file(self, sagemaker):
        assert sagemaker.generic().resourceconfig() == {}

    def test_dataset_folder(self, sagemaker):
        with pytest.deprecated_call():
            result = sagemaker.generic().dataset_folder("foo")
        assert result == pathlib.Path(str(sagemaker.ml_folder.join("input/data/foo")))

    def test_data_channel_folder(self, sagemaker):
        assert sagemaker.generic().data_channel_folder("foo") == pathlib.Path(
            str(sagemaker.ml_folder.join("input/data/foo"))
        )

    def test_model_folder(self, sagemaker):
        assert sagemaker.generic().model_folder() == pathlib.Path(
            str(sagemaker.ml_folder.join("model"))
        )

    def test_write_failure(self, sagemaker):
        failure_path = sagemaker.ml_folder.mkdir("output").join("failure")
        text = "\n".join(["BadModel", "no biscuit"])
        sagemaker.generic().write_failure(text)
        assert failure_path.read() == text


class TestImportString:
    def test_import_string(self):
        cls = import_string("tests.test_core.TestImportString")
        assert cls is TestImportString


class TestModelTrainer:
    def test_create(self, sagemaker):
        env = sagemaker.train()
        trainer = ModelTrainer(env)
        assert trainer.env is env

    def test_train(self, sagemaker):
        trainer = ModelTrainer(sagemaker.train())
        with pytest.raises(NotImplementedError) as exc_info:
            trainer.train()
        assert str(exc_info.value) == "Sub-classes should implement .train()"


class TestModelPredictor:
    def test_create(self, sagemaker):
        env = sagemaker.serve()
        predictor = ModelPredictor(env)
        assert predictor.env is env

    def test_setup(self, sagemaker):
        predictor = ModelPredictor(sagemaker.serve())
        predictor.setup()

    def test_teardown(self, sagemaker):
        predictor = ModelPredictor(sagemaker.serve())
        predictor.teardown()

    def test_invoke_with_result_not_implemented(self, sagemaker):
        predictor = ModelPredictor(sagemaker.serve())
        with pytest.raises(NotImplementedError) as exc_info:
            predictor.invoke({})
        assert str(exc_info.value) == "Sub-classes should implement .result(...)"

    def test_invoke_with_result_implemented(self, sagemaker, fake_utcnow):
        class MyPredictor(ModelPredictor):
            def result(self, data):
                return {"probability": 0.5, "input": data["input"]}

        predictor = MyPredictor(sagemaker.serve())
        assert predictor.invoke({"input": 1}) == {
            "metadata": {
                "model_version": "test-model-1.2.3",
                "timestamp": 1548936002.0,
            },
            "result": {"probability": 0.5, "input": 1},
        }

    def test_metadata(self, sagemaker, fake_utcnow):
        predictor = ModelPredictor(sagemaker.serve())
        assert predictor.metadata({}) == {
            "model_version": "test-model-1.2.3",
            "timestamp": 1548936002.0,
        }

    def test_result(self, sagemaker):
        predictor = ModelPredictor(sagemaker.serve())
        with pytest.raises(NotImplementedError) as exc_info:
            predictor.result({})
        assert str(exc_info.value) == "Sub-classes should implement .result(...)"


class TestModel:
    def test_trainer_not_set(self, sagemaker):
        with pytest.raises(ValueError) as exc_info:
            Model().trainer(sagemaker.train())
        assert str(exc_info.value) == ".TRAINER should be an instance of ModelTrainer"

    def test_trainer_set(self, sagemaker):
        class MyModel(Model):
            TRAINER = ModelTrainer

        env = sagemaker.train()
        trainer = MyModel().trainer(env)
        assert trainer.__class__ is ModelTrainer
        assert trainer.env is env

    def test_predictor_not_set(self, sagemaker):
        with pytest.raises(ValueError) as exc_info:
            Model().predictor(sagemaker.serve())
        assert (
            str(exc_info.value) == ".PREDICTOR should be an instance of ModelPredictor"
        )

    def test_predictor_set(self, sagemaker):
        class MyModel(Model):
            PREDICTOR = ModelPredictor

        env = sagemaker.serve()
        predictor = MyModel().predictor(env)
        assert predictor.__class__ is ModelPredictor
        assert predictor.env is env


class TestNamingValidation:
    def test_naming_validation_noncompliance(self):
        with pytest.raises(NamingError) as exc_info:
            validate_name("a wrong name", "training-job")
        assert (
            str(exc_info.value) == "Training job names should be in the "
            "format <model-name>-X-Y-Z-[dev]"
        )
        with pytest.raises(NamingError) as exc_info:
            validate_name("a wrong name", "model")
        assert (
            str(exc_info.value) == "Model names should be in the"
            " format <model-name>-X-Y-Z-[dev]"
        )
        with pytest.raises(NamingError) as exc_info:
            validate_name("a wrong name", "endpoint")
        assert (
            str(exc_info.value) == "Endpoint names should be in the"
            " format <model-name>-X-Y-Z-[dev]-[live|analysis|test]"
        )

    def test_naming_validation_compliance(self):
        validate_name("test-model-0-0-0-dev", "training-job")
        validate_name("test-model-0-0-0", "training-job")
        validate_name("test-model-10-11-12", "training-job")
        validate_name("test-model-0-0-0-dev", "model")
        validate_name("test-model-0-0-0", "model")
        validate_name("test-model-10-11-12", "model")
        validate_name("test-model-0-0-0-dev", "endpoint")
        validate_name("test-model-0-0-0-dev-live", "endpoint")
        validate_name("test-model-0-0-0-dev-analysis", "endpoint")
        validate_name("test-model-0-0-0-dev-test", "endpoint")
        validate_name("test-model-0-0-0", "endpoint")
        validate_name("test-model-10-11-12", "endpoint")
        validate_name("test-model-0-0-0-live", "endpoint")
        validate_name("test-model-0-0-0-analysis", "endpoint")
        validate_name("test-model-0-0-0-test", "endpoint")
