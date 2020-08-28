# -*- coding: utf-8 -*-

""" Tests for ml2p.cli. """

import json

import click
import pytest
from click.testing import CliRunner

from ml2p import __version__ as ml2p_version
from ml2p import cli
from ml2p.core import ModellingProject


class DummyCtx:
    pass


class CLIHelper:
    def __init__(self, moto_session, moto_sagemaker, tmp_path, bucket, base_cfg):
        self._moto_session = moto_session
        self._tmp_path = tmp_path
        self._bucket = bucket
        self._base_cfg = base_cfg
        self.s3 = self._moto_session.client("s3")
        self.sagefaker = moto_sagemaker.mocked_client("sagemaker")

    def _apply_base_cfg(self, **kw):
        d = {}
        d.update(self._base_cfg)
        d.update(kw)
        return d

    def cfg(self, config_name="ml2p.yml", **kw):
        cfg_file = self._tmp_path / config_name
        cfg_file.write_text(json.dumps(self._apply_base_cfg(**kw)))
        return str(cfg_file)

    def ctx(self, **kw):
        ctx = DummyCtx()
        ctx.obj = ModellingProject(self.cfg(**kw))
        return ctx

    def s3_create_bucket(self):
        self.s3.create_bucket(Bucket=self._bucket)

    def s3_list_objects(self):
        list_objects = self.s3.list_objects(Bucket=self._bucket)
        if "Contents" not in list_objects:
            return None
        return [item["Key"] for item in list_objects["Contents"]]

    def s3_get_object(self, key):
        return self.s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def s3_put_object(self, key, data):
        return self.s3.put_object(Bucket=self._bucket, Key=key, Body=data)

    def invoke(
        self,
        args,
        output=None,
        output_startswith=None,
        output_jsonl=None,
        exit_code=0,
        cfg=None,
    ):
        if cfg is None:
            cfg = {}
        runner = CliRunner()
        result = runner.invoke(
            cli.ml2p, ["--cfg", self.cfg(**cfg)] + args, catch_exceptions=False,
        )
        if result.exception is not None:
            raise result.exception.with_traceback(result.exc_info[2])
        assert result.exit_code == exit_code
        if output is not None:
            assert result.output.splitlines() == output
        if output_startswith is not None:
            assert (
                result.output.splitlines()[: len(output_startswith)]
                == output_startswith
            )
        if output_jsonl is not None:
            assert result.output == "\n".join(
                [json.dumps(data, indent=2) for data in output_jsonl] + [""]
            )


@pytest.fixture
def cli_helper(moto_session, moto_sagemaker, tmp_path):
    return CLIHelper(
        moto_session,
        moto_sagemaker,
        tmp_path,
        bucket="my-bucket",
        base_cfg={"project": "my-models", "s3folder": "s3://my-bucket/my-models/"},
    )


class TestValidateModelType:
    def test_value_in_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert cli.validate_model_type(ctx, "model_type", "model-1") == "model-1"

    def test_value_not_in_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        with pytest.raises(click.BadParameter) as err:
            cli.validate_model_type(ctx, "model_type", "model-2")
        assert str(err.value) == "Unknown model type."

    def test_value_is_none_no_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={})
        assert cli.validate_model_type(ctx, "model_type", None) is None

    def test_value_is_none_single_model_type(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert cli.validate_model_type(ctx, "model_type", None) == "model-1"

    def test_value_is_none_multiple_model_types(self, cli_helper):
        ctx = cli_helper.ctx(
            models={
                "model-1": "my_models.ml2p.Model1",
                "model-2": "my_models.ml2p.Model2",
            }
        )
        with pytest.raises(click.BadParameter) as err:
            cli.validate_model_type(ctx, "model_type", None)
        assert str(err.value) == (
            "Model type may only be omitted if zero or one models are listed"
            " in the ML2P config YAML file."
        )


class TestModellingProjectWithSagemakerClient:
    def test_create(self, cli_helper):
        prj = cli.ModellingProjectWithSagemakerClient(cli_helper.cfg())
        assert type(prj.client).__name__ == "SageFakerClient"


class TestML2P:
    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["--help"],
            output_startswith=[
                "Usage: ml2p [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Minimal Lovable Machine Learning Pipeline.",
                "",
                "  A friendlier interface to AWS SageMaker.",
            ],
        )

    def test_version(self, cli_helper):
        cli_helper.invoke(
            ["--version"], output=["ml2p, version {}".format(ml2p_version)]
        )


class TestInit:
    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["init", "--help"],
            output_startswith=[
                "Usage: ml2p init [OPTIONS]",
                "",
                "  Initialize the project S3 bucket.",
            ],
        )

    def test_init(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["init"], output=[])
        assert cli_helper.s3_list_objects() == [
            "my-models/datasets/README.rst",
            "my-models/models/README.rst",
        ]
        assert (
            cli_helper.s3_get_object("my-models/datasets/README.rst").decode("utf-8")
            == "Datasets for my-models."
        )
        assert (
            cli_helper.s3_get_object("my-models/models/README.rst").decode("utf-8")
            == "Models for my-models."
        )


class TestDataset:
    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["dataset", "--help"],
            output_startswith=[
                "Usage: ml2p dataset [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and manage datasets.",
            ],
        )

    def test_create(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["dataset", "create", "ds-20201012"], output=[])
        assert cli_helper.s3_list_objects() == [
            "my-models/datasets/ds-20201012/README.rst"
        ]
        assert (
            cli_helper.s3_get_object(
                "my-models/datasets/ds-20201012/README.rst"
            ).decode("utf-8")
            == "Dataset ds-20201012 for project my-models."
        )

    def test_list(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["dataset", "create", "ds-20201012"])
        cli_helper.invoke(["dataset", "create", "ds-20201013"])
        cli_helper.invoke(
            ["dataset", "list"], output_jsonl=["ds-20201012", "ds-20201013"]
        )

    def test_delete(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.invoke(["dataset", "create", "ds-20201012"], [])
        cli_helper.invoke(["dataset", "delete", "ds-20201012"], [])
        assert cli_helper.s3_list_objects() is None

    def test_ls(self, cli_helper):
        cli_helper.s3_create_bucket()
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/a.txt", b"aa")
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/b.txt", b"bbb")
        cli_helper.invoke(
            ["dataset", "ls", "ds-20201012"],
            output_jsonl=[
                {"filename": "a.txt", "size": 2},
                {"filename": "b.txt", "size": 3},
            ],
        )

    def test_up(self, cli_helper, data_fixtures):
        cli_helper.s3_create_bucket()
        training_set = str(data_fixtures / "training_set.csv")
        cli_helper.invoke(["dataset", "up", "ds-20201012", training_set], [])
        assert cli_helper.s3_list_objects() == [
            "my-models/datasets/ds-20201012/training_set.csv"
        ]
        assert (
            cli_helper.s3_get_object(
                "my-models/datasets/ds-20201012/training_set.csv"
            ).decode("utf-8")
            == 'X1,X2,X3,y\n"A",1,10.5,1\n"B",2,7.4,0\n'
        )

    def test_dn(self, cli_helper, data_fixtures, tmpdir):
        cli_helper.s3_create_bucket()
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/a.txt", b"aa")
        with tmpdir.as_cwd():
            cli_helper.invoke(["dataset", "dn", "ds-20201012", "a.txt"])
        assert (tmpdir / "a.txt").read() == "aa"

    def test_rm(self, cli_helper, data_fixtures):
        cli_helper.s3_create_bucket()
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/a.txt", b"aa")
        cli_helper.s3_put_object("my-models/datasets/ds-20201012/b.txt", b"bbb")
        cli_helper.invoke(["dataset", "rm", "ds-20201012", "a.txt"])
        assert cli_helper.s3_list_objects() == ["my-models/datasets/ds-20201012/b.txt"]


class TestTrainingJob:
    def example_1(self):
        training_job = {
            "TrainingJobName": "my-models-tj-0-1-11",
            "AlgorithmSpecification": {
                "TrainingImage": (
                    "12345.dkr.ecr.eu-west-1.amazonaws.com/docker-image:0.0.2"
                ),
                "TrainingInputMode": "File",
            },
            "EnableNetworkIsolation": True,
            "HyperParameters": {
                "ML2P_ENV.ML2P_PROJECT": '"my-models"',
                "ML2P_ENV.ML2P_S3_URL": '"s3://my-bucket/my-models/"',
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": ("s3://my-bucket/my-models/datasets/ds-20201012"),
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": "s3://my-bucket/my-models/models/"},
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.large",
                "VolumeSizeInGB": 20,
            },
            "RoleArn": "arn:aws:iam::12345:role/role-name",
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
            "Tags": [{"Key": "ml2p-project", "Value": "my-models"}],
        }
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.eu-west-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
            "train": {"instance_type": "ml.m5.large"},
        }
        return training_job, cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["training-job", "--help"],
            output_startswith=[
                "Usage: ml2p training-job [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and inspect training jobs.",
            ],
        )

    def test_list_empty(self, cli_helper):
        cli_helper.invoke(
            ["training-job", "list"], output_jsonl=[],
        )

    def test_create_and_list(self, cli_helper):
        training_job, cfg = self.example_1()
        cli_helper.invoke(
            ["training-job", "create", "tj-0-1-11", "ds-20201012"],
            output_jsonl=[training_job],
            cfg=cfg,
        )
        cli_helper.invoke(
            ["training-job", "list"], output_jsonl=[training_job],
        )

    def test_create_and_describe(self, cli_helper):
        training_job, cfg = self.example_1()
        cli_helper.invoke(
            ["training-job", "create", "tj-0-1-11", "ds-20201012"],
            output_jsonl=[training_job],
            cfg=cfg,
        )
        cli_helper.invoke(
            ["training-job", "describe", "tj-0-1-11"], output_jsonl=[training_job],
        )

    def test_create_and_wait(self, cli_helper):
        training_job, cfg = self.example_1()
        cli_helper.invoke(
            ["training-job", "create", "tj-0-1-11", "ds-20201012"],
            output_jsonl=[training_job],
            cfg=cfg,
        )
        cli_helper.invoke(
            ["training-job", "wait", "tj-0-1-11"], output=[],
        )


class TestModel:
    def example_1(self):
        model = {
            "ModelName": "my-models-mdl-0-1-12",
            "PrimaryContainer": {
                "Image": "12345.dkr.ecr.eu-west-1.amazonaws.com/docker-image:0.0.2",
                "ModelDataUrl": (
                    "s3://my-bucket/my-models/models/"
                    "my-models-mdl-0-1/output/model.tar.gz"
                ),
                "Environment": {
                    "ML2P_MODEL_VERSION": "my-models-mdl-0-1-12",
                    "ML2P_PROJECT": "my-models",
                    "ML2P_S3_URL": "s3://my-bucket/my-models/",
                },
            },
            "ExecutionRoleArn": "arn:aws:iam::12345:role/role-name",
            "Tags": [{"Key": "ml2p-project", "Value": "my-models"}],
            "EnableNetworkIsolation": False,
        }
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.eu-west-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
        }
        return model, cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["model", "--help"],
            output_startswith=[
                "Usage: ml2p model [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and inspect models.",
            ],
        )

    def test_list_empty(self, cli_helper):
        cli_helper.invoke(
            ["model", "list"], output_jsonl=[],
        )

    def test_create_and_list(self, cli_helper):
        model, cfg = self.example_1()
        cli_helper.invoke(
            ["model", "create", "mdl-0-1-12"], output_jsonl=[model], cfg=cfg,
        )
        cli_helper.invoke(
            ["model", "list"], output_jsonl=[model],
        )

    def test_create_and_describe(self, cli_helper):
        model, cfg = self.example_1()
        cli_helper.invoke(
            ["model", "create", "mdl-0-1-12"], output_jsonl=[model], cfg=cfg,
        )
        cli_helper.invoke(
            ["model", "describe", "mdl-0-1-12"], output_jsonl=[model],
        )

    def test_create_and_delete(self, cli_helper):
        model, cfg = self.example_1()
        cli_helper.invoke(
            ["model", "create", "mdl-0-1-12"], output_jsonl=[model], cfg=cfg,
        )
        cli_helper.invoke(
            ["model", "delete", "mdl-0-1-12"], output_jsonl=[model], cfg=cfg,
        )


class TestEndpoint:
    def example_1(self):
        endpoint = {
            "EndpointName": "my-models-endpoint-0-1-12",
            "EndpointConfigName": "my-models-endpoint-0-1-12-config",
            "Tags": [{"Key": "ml2p-project", "Value": "my-models"}],
            "EndpointArn": (
                "arn:aws:sagemaker:us-east-1:12345:endpoint/my-models-endpoint-0-1-12"
            ),
        }
        endpoint_cfg = {
            "EndpointConfigName": "my-models-endpoint-0-1-12-config",
            "ProductionVariants": [
                {
                    "VariantName": "my-models-endpoint-0-1-12-variant-1",
                    "ModelName": "my-models-endpoint-0-1-12",
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.t2.medium",
                    "InitialVariantWeight": 1.0,
                }
            ],
            "Tags": [{"Key": "ml2p-project", "Value": "my-models"}],
        }
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.eu-west-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
            "deploy": {"instance_type": "ml.t2.medium"},
        }

        return endpoint, endpoint_cfg, cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["endpoint", "--help"],
            output_startswith=[
                "Usage: ml2p endpoint [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and inspect endpoints.",
            ],
        )

    def test_list_empty(self, cli_helper):
        cli_helper.invoke(["endpoint", "list"], output_jsonl=[])

    def test_create_and_list(self, cli_helper):
        endpoint, endpoint_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["endpoint", "create", "endpoint-0-1-12"], output_jsonl=[endpoint], cfg=cfg,
        )
        cli_helper.invoke(
            ["endpoint", "list"], output_jsonl=[endpoint],
        )
        pages = list(
            cli_helper.sagefaker.get_paginator("list_endpoint_configs").paginate()
        )
        assert pages == [{"EndpointConfigs": [endpoint_cfg]}]

    def test_create_and_describe(self, cli_helper):
        endpoint, endpoint_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["endpoint", "create", "endpoint-0-1-12"], output_jsonl=[endpoint], cfg=cfg,
        )
        endpoint_with_url = dict(
            **endpoint,
            EndpointUrl=(
                "https://runtime.sagemaker.us-east-1.amazonaws.com"
                "/endpoints/my-models-endpoint-0-1-12/invocations"
            )
        )
        cli_helper.invoke(
            ["endpoint", "describe", "endpoint-0-1-12"],
            output_jsonl=[endpoint_with_url],
        )
        assert (
            cli_helper.sagefaker.describe_endpoint_config(
                EndpointConfigName="my-models-endpoint-0-1-12-config"
            )
            == endpoint_cfg
        )

    def test_create_and_delete(self, cli_helper):
        endpoint, endpoint_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["endpoint", "create", "endpoint-0-1-12"], output_jsonl=[endpoint], cfg=cfg,
        )
        cli_helper.invoke(
            ["endpoint", "delete", "endpoint-0-1-12"],
            output_jsonl=[endpoint, endpoint_cfg],
            cfg=cfg,
        )

    def test_create_and_invoke(self, cli_helper):
        endpoint, endpoint_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["endpoint", "create", "endpoint-0-1-12"], output_jsonl=[endpoint], cfg=cfg,
        )
        cli_helper.invoke(
            ["endpoint", "invoke", "endpoint-0-1-12", json.dumps({"j": "son"})],
            output_jsonl=[{"Body": {"inputs": {"j": "son"}}}],
            cfg=cfg,
        )

    def test_create_and_wait(self, cli_helper):
        endpoint, endpoint_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["endpoint", "create", "endpoint-0-1-12"], output_jsonl=[endpoint], cfg=cfg,
        )
        cli_helper.invoke(
            ["endpoint", "wait", "endpoint-0-1-12"], output=[], cfg=cfg,
        )


class TestNotebook:
    def example_1(self):
        notebook = {
            "NotebookInstanceName": "my-models-notebook-test",
            "InstanceType": "ml.t2.medium",
            "RoleArn": "arn:aws:iam::12345:role/role-name",
            "Tags": [{"Key": "ml2p-project", "Value": "my-models"}],
            "LifecycleConfigName": "my-models-notebook-test-lifecycle-config",
            "VolumeSizeInGB": 8,
            "DirectInternetAccess": "Disabled",
            "DefaultCodeRepository": None,
            "NotebookInstanceStatus": "Pending",
        }
        lifecycle_cfg = {
            "NotebookInstanceLifecycleConfigName": (
                "my-models-notebook-test-lifecycle-config"
            ),
        }
        cfg = {
            "defaults": {
                "image": "12345.dkr.ecr.eu-west-1.amazonaws.com/docker-image:0.0.2",
                "role": "arn:aws:iam::12345:role/role-name",
            },
            "notebook": {"instance_type": "ml.t2.medium", "volume_size": 8},
        }
        return notebook, lifecycle_cfg, cfg

    def example_2_repo_url(self):
        notebook, lifecycle_cfg, cfg = self.example_1()
        notebook["DefaultCodeRepository"] = "my-models-notebook-test-repo"
        cfg["notebook"].update(
            **{
                "repo_url": "https://example.com/repo-1234",
                "repo_branch": "master",
                "repo_secret_arn": "arn:secret:1234",
            }
        )
        return notebook, lifecycle_cfg, cfg

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["notebook", "--help"],
            output_startswith=[
                "Usage: ml2p notebook [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Create and manage notebooks.",
            ],
        )

    def test_list_empty(self, cli_helper):
        cli_helper.invoke(["notebook", "list"], output_jsonl=[])

    def test_create_and_list(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "list"], output_jsonl=[notebook],
        )
        pages = list(
            cli_helper.sagefaker.get_paginator(
                "list_notebook_instance_lifecycle_configs"
            ).paginate()
        )
        assert pages == [{"NotebookInstanceLifecycleConfigs": [lifecycle_cfg]}]

    def test_create_and_list_with_repo_url(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_2_repo_url()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "list"], output_jsonl=[notebook],
        )
        pages = list(
            cli_helper.sagefaker.get_paginator(
                "list_notebook_instance_lifecycle_configs"
            ).paginate()
        )
        assert pages == [{"NotebookInstanceLifecycleConfigs": [lifecycle_cfg]}]

    def test_create_and_describe(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "describe", "notebook-test"], output_jsonl=[notebook],
        )
        assert (
            cli_helper.sagefaker.describe_notebook_instance_lifecycle_config(
                NotebookInstanceLifecycleConfigName=(
                    "my-models-notebook-test-lifecycle-config"
                )
            )
            == lifecycle_cfg
        )

    def test_create_and_delete(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "delete", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        pages = list(
            cli_helper.sagefaker.get_paginator(
                "list_notebook_instance_lifecycle_configs"
            ).paginate()
        )
        assert pages == [{"NotebookInstanceLifecycleConfigs": []}]

    def test_create_and_delete_while_in_service(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "start", "notebook-test"], output_jsonl=[], cfg=cfg,
        )
        notebook["NotebookInstanceStatus"] = "Stopped"
        cli_helper.invoke(
            ["notebook", "delete", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        pages = list(
            cli_helper.sagefaker.get_paginator(
                "list_notebook_instance_lifecycle_configs"
            ).paginate()
        )
        assert pages == [{"NotebookInstanceLifecycleConfigs": []}]

    def test_create_and_delete_with_repo(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_2_repo_url()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "delete", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        lifecycle_pages = list(
            cli_helper.sagefaker.get_paginator(
                "list_notebook_instance_lifecycle_configs"
            ).paginate()
        )
        assert lifecycle_pages == [{"NotebookInstanceLifecycleConfigs": []}]
        repo_pages = list(
            cli_helper.sagefaker.get_paginator("list_code_repositories").paginate()
        )
        assert repo_pages == [{"CodeRepositorySummaryList": []}]

    def test_presigned_url(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "presigned-url", "notebook-test"],
            output_jsonl=[
                {"AuthorizedUrl": "https://example.com/my-models-notebook-test"}
            ],
            cfg=cfg,
        )

    def test_stop(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "stop", "notebook-test"], output_jsonl=[], cfg=cfg,
        )
        notebook["NotebookInstanceStatus"] = "Stopped"
        cli_helper.invoke(
            ["notebook", "describe", "notebook-test"], output_jsonl=[notebook],
        )

    def test_start(self, cli_helper):
        notebook, lifecycle_cfg, cfg = self.example_1()
        cli_helper.invoke(
            ["notebook", "create", "notebook-test"], output_jsonl=[notebook], cfg=cfg,
        )
        cli_helper.invoke(
            ["notebook", "start", "notebook-test"], output_jsonl=[], cfg=cfg,
        )
        notebook["NotebookInstanceStatus"] = "InService"
        cli_helper.invoke(
            ["notebook", "describe", "notebook-test"], output_jsonl=[notebook],
        )


class TestRepo:
    def example_1(self):
        repo = {
            "CodeRepositoryName": "my-models-repo-1234",
            "GitConfig": {
                "RepositoryUrl": "https://example.com/repo-1234",
                "Branch": "master",
                "SecretArn": "arn:secret:repo-1234",
            },
        }
        return repo

    def test_help(self, cli_helper):
        cli_helper.invoke(
            ["repo", "--help"],
            output_startswith=[
                "Usage: ml2p repo [OPTIONS] COMMAND [ARGS]...",
                "",
                "  Describe and list code repositories.",
            ],
        )

    def test_list_empty(self, cli_helper):
        cli_helper.invoke(
            ["repo", "list"], output_jsonl=[],
        )

    def test_list(self, cli_helper):
        repo = self.example_1()
        cli_helper.sagefaker.create_code_repository(**repo)
        cli_helper.invoke(
            ["repo", "list"], output_jsonl=[repo],
        )

    def test_describe(self, cli_helper):
        repo = self.example_1()
        cli_helper.sagefaker.create_code_repository(**repo)
        cli_helper.invoke(
            ["repo", "describe", "repo-1234"], output_jsonl=[repo],
        )
