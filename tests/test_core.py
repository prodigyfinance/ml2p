# -*- coding: utf-8 -*-

""" Tests for ml2p.core. """

import io
import pathlib
import tarfile

import pytest

from ml2p import __version__ as ml2p_version
from ml2p.core import (
    S3URL,
    Model,
    ModellingSubCfg,
    ModelPredictor,
    ModelTrainer,
    import_string,
)
from ml2p.errors import LocalEnvError


def mk_subcfg(defaults="defaults"):
    return ModellingSubCfg(
        {"sub": {"a": 1, "b": "boo"}, "defaults": {"c": 3}}, "sub", defaults=defaults
    )


class TestModellingSubCfg:
    def test_getattr(self):
        subcfg = mk_subcfg()
        assert subcfg.a == 1
        assert subcfg.c == 3

    def test_getitem(self):
        subcfg = mk_subcfg()
        assert subcfg["a"] == 1
        assert subcfg["c"] == 3

    def test_setitem(self):
        subcfg = mk_subcfg()
        subcfg["d"] = 5
        assert subcfg.d == 5
        assert subcfg["d"] == 5

    def test_keys(self):
        subcfg = mk_subcfg()
        assert subcfg.keys() == ["a", "b", "c"]

    def test_get(self):
        subcfg = mk_subcfg()
        assert subcfg.get("a") == 1
        assert subcfg.get("d") is None
        assert subcfg.get("d", 3) == 3


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
        assert env.record_invokes is None
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
        assert env.record_invokes is False
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

    def test_create_env_with_record_invokes(self, sagemaker):
        env = sagemaker.serve(ML2P_RECORD_INVOKES="true")
        assert env.record_invokes is True


class TestSageMakerEnvLocal:
    def test_basic_env(self, sagemaker):
        env = sagemaker.local()
        assert env.env_type == env.LOCAL
        assert env.model_version == "local"
        assert env.record_invokes is False
        assert env.training_job_name is None
        assert env.project == "test-project"
        assert env.model_cls is None
        assert env.s3.url() == "s3://foo/bar/"
        assert env.model_folder().is_dir() is True

    def test_fails_when_local_folder_does_not_exist(self, sagemaker):
        ml_folder = sagemaker.ml_folder / "does-not-exist"
        with pytest.raises(LocalEnvError) as e:
            sagemaker.local(ml_folder=ml_folder)
        assert str(e.value) == f"Local environment folder {ml_folder} does not exist."

    def test_no_session(self, sagemaker):
        env = sagemaker.local(session=None)
        assert env.env_type == env.LOCAL
        assert env.model_version == "local"
        assert env.record_invokes is False
        assert env.training_job_name is None
        assert env.project == "test-project"
        assert env.model_cls is None
        assert env.s3.url() == "s3://foo/bar/"
        with pytest.raises(LocalEnvError) as e:
            env.download_dataset("my-dataset")
        assert str(e.value) == "Downloading datasets requires a boto session."
        with pytest.raises(LocalEnvError) as e:
            env.download_model("my-training-job")
        assert str(e.value) == "Downloading models requires a boto session."

    def test_clean_model_folder(self, sagemaker):
        env = sagemaker.local()
        model_folder = sagemaker.ml_folder / "model"
        (model_folder / "old.txt").write("some-data")
        assert sorted(p.basename for p in model_folder.listdir()) == ["old.txt"]
        env.clean_model_folder()
        assert sorted(p.basename for p in model_folder.listdir()) == []

    def test_download_dataset(self, sagemaker):
        env = sagemaker.local()
        sagemaker.s3_put_object(
            "foo", "bar/datasets/bubbles-2012/params.json", {"a": 1}
        )
        # create a dummy folder in S3 to check folder skipping
        sagemaker.s3_put_bytes("foo", "bar/datasets/bubbles-2012/subdir/", b"")
        sagemaker.s3_put_object(
            "foo", "bar/datasets/bubbles-2012/subdir/data.json", {"b": 2}
        )
        env.download_dataset("bubbles-2012")
        training_folder = sagemaker.ml_folder / "input" / "data" / "training"
        training_root = sorted(p.basename for p in training_folder.listdir())
        assert training_root == ["params.json", "subdir"]
        training_subdir = sorted(
            p.basename for p in (training_folder / "subdir").listdir()
        )
        assert training_subdir == ["data.json"]
        assert (training_folder / "params.json").read() == '{"a": 1}'
        assert (training_folder / "subdir" / "data.json").read() == '{"b": 2}'

    def make_tarfile(self, tmpdir, name, files):
        raw = io.BytesIO()
        with tarfile.open(name=name, fileobj=raw, mode="w:gz") as tar:
            for arcname, data in files.items():
                f = tmpdir.join(arcname)
                f.write(data)
                tarinfo = tar.gettarinfo(f, arcname=arcname)
                with f.open("rb") as fh:
                    tar.addfile(tarinfo, fileobj=fh)
        return raw.getvalue()

    def test_download_model(self, sagemaker, tmpdir):
        env = sagemaker.local()
        model_tar_gz_data = self.make_tarfile(
            tmpdir, "model.tar.gz", {"pipeline.json": b'{"a": 1}'}
        )
        sagemaker.s3_put_bytes(
            "foo",
            "bar/models/test-project-train-2012/output/model.tar.gz",
            model_tar_gz_data,
        )
        env.download_model("train-2012")
        model_folder = sagemaker.ml_folder / "model"
        model_root = sorted(p.basename for p in model_folder.listdir())
        assert model_root == ["model.tar.gz", "pipeline.json"]
        assert (model_folder / "pipeline.json").read() == '{"a": 1}'


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

    def test_dataset_folder_default(self, sagemaker):
        assert sagemaker.generic().dataset_folder() == pathlib.Path(
            str(sagemaker.ml_folder.join("input/data/training"))
        )

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


class DummyPredictor(ModelPredictor):
    def result(self, data):
        return {"probability": 0.5, "input": data["input"]}


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
        predictor = DummyPredictor(sagemaker.serve())
        assert predictor.invoke({"input": 1}) == {
            "metadata": {
                "model_version": "test-model-1.2.3",
                "ml2p_version": str(ml2p_version),
                "timestamp": 1548936002.0,
            },
            "result": {"probability": 0.5, "input": 1},
        }

    def test_invoke_with_recording(self, sagemaker, fake_utcnow, fake_uuid4):
        predictor = DummyPredictor(sagemaker.serve(ML2P_RECORD_INVOKES="true"))
        data = {"input": 1}
        prediction = predictor.invoke(data)
        record = sagemaker.s3_get_object(
            "foo",
            "bar/predictions/test-model-1.2.3/"
            "ts-2019-01-31T12:00:02+00:00--"
            "uuid-20f803c4-f155-469e-ba96-14caa30af9e1.json",
        )
        assert record == {"input": data, "result": prediction}

    def test_invoke_batch_with_result_not_implemented(self, sagemaker):
        predictor = ModelPredictor(sagemaker.serve())
        with pytest.raises(NotImplementedError) as exc_info:
            predictor.batch_invoke([{}])
        assert str(exc_info.value) == "Sub-classes should implement .result(...)"

    def test_invoke_batch_with_result_implemented(self, sagemaker, fake_utcnow):
        predictor = DummyPredictor(sagemaker.serve())
        assert predictor.batch_invoke([{"input": 1}]) == {
            "predictions": [
                {
                    "metadata": {
                        "model_version": "test-model-1.2.3",
                        "ml2p_version": str(ml2p_version),
                        "timestamp": 1548936002.0,
                    },
                    "result": {"probability": 0.5, "input": 1},
                }
            ]
        }

    def test_invoke_batch_with_recording(self, sagemaker, fake_utcnow, fake_uuid4):
        predictor = DummyPredictor(sagemaker.serve(ML2P_RECORD_INVOKES="true"))
        data = {"input": 1}
        prediction = predictor.batch_invoke([data])
        record = sagemaker.s3_get_object(
            "foo",
            "bar/predictions/test-model-1.2.3/"
            "ts-2019-01-31T12:00:02+00:00--"
            "uuid-20f803c4-f155-469e-ba96-14caa30af9e1.json",
        )
        assert record == {"input": data, "result": prediction["predictions"][0]}

    def test_metadata(self, sagemaker, fake_utcnow):
        predictor = ModelPredictor(sagemaker.serve())
        assert predictor.metadata() == {
            "model_version": "test-model-1.2.3",
            "ml2p_version": str(ml2p_version),
            "timestamp": 1548936002.0,
        }

    def test_result(self, sagemaker):
        predictor = ModelPredictor(sagemaker.serve())
        with pytest.raises(NotImplementedError) as exc_info:
            predictor.result({})
        assert str(exc_info.value) == "Sub-classes should implement .result(...)"

    def test_record_invoke(self, sagemaker, fake_utcnow, fake_uuid4):
        predictor = ModelPredictor(sagemaker.serve())
        datum = {"feature_a": 1, "feature_b": "b"}
        prediction = {
            "metadata": predictor.metadata(),
            "result": {"probability": 0.5, "input": 1},
        }
        predictor.record_invoke(datum, prediction)
        record = sagemaker.s3_get_object(
            "foo",
            "bar/predictions/test-model-1.2.3/"
            "ts-2019-01-31T12:00:02+00:00--"
            "uuid-20f803c4-f155-469e-ba96-14caa30af9e1.json",
        )
        assert record == {"input": datum, "result": prediction}

    def test_record_invoke_id(self, sagemaker, fake_utcnow, fake_uuid4):
        predictor = ModelPredictor(sagemaker.serve())
        assert predictor.record_invoke_id({"a": "inputs"}, {"b": "outputs"}) == {
            "ts": "2019-01-31T12:00:02+00:00",
            "uuid": "20f803c4-f155-469e-ba96-14caa30af9e1",
        }


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
