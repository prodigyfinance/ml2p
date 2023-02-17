# -*- coding: utf-8 -*-

""" Tests for ml2p.cli_utils. """

import base64
import datetime
from unittest.mock import patch

import click
import pytest
from pkg_resources import resource_filename

from ml2p.cli import ModellingProject
from ml2p.cli_commands import utils
from ml2p.errors import ConfigError, NamingError


@pytest.fixture
def prj():
    with patch("boto3.client"):
        cfg = resource_filename("tests.fixture_files", "ml2p.yml")
        prj = ModellingProject(cfg)
    return prj


@pytest.fixture
def prj_no_vpc():
    with patch("boto3.client"):
        cfg_no_vpc = resource_filename("tests.fixture_files", "ml2p-no-vpc.yml")
        prj_no_vpc = ModellingProject(cfg_no_vpc)
    return prj_no_vpc


def on_start_fixture():
    with open(resource_filename("tests.fixture_files", "on_start.sh"), "rb") as f:
        return f.read()


def on_create_fixture():
    with open(resource_filename("tests.fixture_files", "on_create.sh"), "rb") as f:
        return f.read()


class TestCliUtils:
    def test_date_to_string_serializer(self):
        value = datetime.datetime(1, 1, 1)
        assert utils.date_to_string_serializer(value) == "0001-01-01 00:00:00"
        with pytest.raises(TypeError) as exc_info:
            utils.date_to_string_serializer("test")
        assert str(exc_info.value) == "Serializing 'test' to JSON not supported."

    def test_click_echo_json(self, capsys):
        response = {"NotebookInstanceName": "notebook-1"}
        utils.click_echo_json(response)
        assert (
            capsys.readouterr().out == '{\n  "NotebookInstanceName": "notebook-1"\n}\n'
        )

    def test_endpoint_url_for_arn(self):
        endpoint_arn = (
            "arn:aws:sagemaker:eu-west-1:123456789012:endpoint/endpoint-20190612"
        )
        assert utils.endpoint_url_for_arn(endpoint_arn) == (
            "https://runtime.sagemaker.eu-west-1.amazonaws.com/"
            "endpoints/endpoint-20190612/invocations"
        )
        assert utils.endpoint_url_for_arn("") is None

    def test_mk_vpc_config_no_config(self, prj):
        assert utils.mk_vpc_config(prj.train) is None
        assert utils.mk_vpc_config(prj.deploy) is None

    def test_mk_vpc_config_valid_config(self, prj):
        prj.train["vpc_config"] = {"security_groups": ["a"], "subnets": ["b"]}
        assert utils.mk_vpc_config(prj.train) == {
            "SecurityGroupIds": ["a"],
            "Subnets": ["b"],
        }

    def test_mk_vpc_config_non_dict(self, prj):
        prj.train["vpc_config"] = 5
        with pytest.raises(ConfigError) as err:
            utils.mk_vpc_config(prj.train)
        assert str(err.value) == (
            "The vpc_config requires a dictionary with keys 'security_groups' and"
            " 'subnets'. Both the security_groups and subnets should contain lists"
            " of IDs."
        )

    def test_mk_vpc_config_missing_security_groups(self, prj):
        prj.train["vpc_config"] = {"subnets": ["a"]}
        with pytest.raises(ConfigError) as err:
            utils.mk_vpc_config(prj.train)
        assert str(err.value) == (
            "The vpc_config requires a dictionary with keys 'security_groups' and"
            " 'subnets'. Both the security_groups and subnets should contain lists"
            " of IDs."
        )

    def test_mk_vpc_config_non_list_security_groups(self, prj):
        prj.train["vpc_config"] = {"security_groups": 5, "subnets": ["a"]}
        with pytest.raises(ConfigError) as err:
            utils.mk_vpc_config(prj.train)
        assert str(err.value) == (
            "The vpc_config requires a dictionary with keys 'security_groups' and"
            " 'subnets'. Both the security_groups and subnets should contain lists"
            " of IDs."
        )

    def test_mk_vpc_config_empty_security_groups(self, prj):
        prj.train["vpc_config"] = {"security_groups": [], "subnets": ["a"]}
        with pytest.raises(ConfigError) as err:
            utils.mk_vpc_config(prj.train)
        assert str(err.value) == (
            "The vpc_config must contain at least one security group id."
        )

    def test_mk_vpc_config_missing_subnets(self, prj):
        prj.train["vpc_config"] = {"security_groups": ["a"]}
        with pytest.raises(ConfigError) as err:
            utils.mk_vpc_config(prj.train)
        assert str(err.value) == (
            "The vpc_config requires a dictionary with keys 'security_groups' and"
            " 'subnets'. Both the security_groups and subnets should contain lists"
            " of IDs."
        )

    def test_mk_vpc_config_non_list_subnets(self, prj):
        prj.train["vpc_config"] = {"security_groups": ["a"], "subnets": 5}
        with pytest.raises(ConfigError) as err:
            utils.mk_vpc_config(prj.train)
        assert str(err.value) == (
            "The vpc_config requires a dictionary with keys 'security_groups' and"
            " 'subnets'. Both the security_groups and subnets should contain lists"
            " of IDs."
        )

    def test_mk_vpc_config_empty_subnets(self, prj):
        prj.train["vpc_config"] = {"security_groups": ["a"], "subnets": []}
        with pytest.raises(ConfigError) as err:
            utils.mk_vpc_config(prj.train)
        assert str(err.value) == ("The vpc_config must contain at least one subnet id.")

    def test_mk_training_job(self, prj):
        training_job_cfg = utils.mk_training_job(prj, "training-job-1", "dataset-1")
        assert training_job_cfg == {
            "TrainingJobName": "modelling-project-training-job-1",
            "AlgorithmSpecification": {
                "TrainingImage": "123456789012.dkr.ecr.eu-west-1.amazonaws.com"
                "/modelling-project-sagemaker:latest",
                "TrainingInputMode": "File",
            },
            "EnableNetworkIsolation": True,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": "s3://prodigyfinance-modelling-project-"
                            "sagemaker-production/datasets/dataset-1",
                        }
                    },
                }
            ],
            "Environment": {
                "ML2P_TRAINING_JOB": "modelling-project-training-job-1",
                "ML2P_PROJECT": "modelling-project",
                "ML2P_S3_URL": "s3://prodigyfinance-modelling-project-"
                "sagemaker-production/",
            },
            "OutputDataConfig": {
                "S3OutputPath": "s3://prodigyfinance-modelling-project-"
                "sagemaker-production/models/"
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.2xlarge",
                "VolumeSizeInGB": 20,
            },
            "RoleArn": "arn:aws:iam::111111111111:role/modelling-project",
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
            "Tags": [{"Key": "ml2p-project", "Value": "modelling-project"}],
        }

    def test_mk_training_job_with_vpc_config(self, prj):
        prj.train["vpc_config"] = {"security_groups": ["sg-1"], "subnets": ["net-2"]}
        training_job_cfg = utils.mk_training_job(prj, "training-job-1", "dataset-1")
        assert training_job_cfg["VpcConfig"] == {
            "SecurityGroupIds": ["sg-1"],
            "Subnets": ["net-2"],
        }

    def test_mk_training_job_with_model_type(self, prj):
        prj.models["model-type-1"] = "my.pkg.model"
        training_job_cfg = utils.mk_training_job(
            prj, "training-job-1", "dataset-1", "model-type-1"
        )
        assert training_job_cfg["Environment"] == {
            "ML2P_TRAINING_JOB": "modelling-project-training-job-1",
            "ML2P_MODEL_CLS": "my.pkg.model",
            "ML2P_PROJECT": "modelling-project",
            "ML2P_S3_URL": (
                "s3://prodigyfinance-modelling-project-sagemaker-production/"
            ),
        }

    def test_mk_training_job_with_missing_model_type(self, prj):
        with pytest.raises(KeyError) as err:
            utils.mk_training_job(prj, "training-job-1", "dataset-1", "model-type-1")
        assert str(err.value) == "'model-type-1'"

    def test_mk_model(self, prj):
        model_cfg = utils.mk_model(prj, "model-1", "training-job-1")
        assert model_cfg == {
            "ModelName": "modelling-project-model-1",
            "PrimaryContainer": {
                "Image": "123456789012.dkr.ecr.eu-west-1.amazonaws.com/"
                "modelling-project-sagemaker:latest",
                "ModelDataUrl": "s3://prodigyfinance-modelling-project-sagemaker"
                "-production/models/modelling-project-training-job-1/"
                "output/model.tar.gz",
                "Environment": {
                    "ML2P_MODEL_VERSION": "modelling-project-model-1",
                    "ML2P_PROJECT": "modelling-project",
                    "ML2P_S3_URL": (
                        "s3://prodigyfinance-modelling-project-sagemaker-production/"
                    ),
                },
            },
            "ExecutionRoleArn": "arn:aws:iam::111111111111:role/modelling-project",
            "Tags": [{"Key": "ml2p-project", "Value": "modelling-project"}],
            "EnableNetworkIsolation": False,
        }

    def test_mk_model_with_vpc_config(self, prj):
        prj.deploy["vpc_config"] = {"security_groups": ["sg-1"], "subnets": ["net-2"]}
        model_cfg = utils.mk_model(prj, "model-1", "training-job-1")
        assert model_cfg["VpcConfig"] == {
            "SecurityGroupIds": ["sg-1"],
            "Subnets": ["net-2"],
        }

    def test_mk_model_with_model_type(self, prj):
        prj.models["model-type-1"] = "my.pkg.model"
        model_cfg = utils.mk_model(prj, "model-1", "training-job-1", "model-type-1")
        assert model_cfg["PrimaryContainer"]["Environment"] == {
            "ML2P_MODEL_CLS": "my.pkg.model",
            "ML2P_MODEL_VERSION": "modelling-project-model-1",
            "ML2P_PROJECT": "modelling-project",
            "ML2P_S3_URL": (
                "s3://prodigyfinance-modelling-project-sagemaker-production/"
            ),
        }

    def test_mk_model_with_missing_model_type(self, prj):
        with pytest.raises(KeyError) as err:
            utils.mk_model(prj, "model-1", "training-job-1", "model-type-1")
        assert str(err.value) == "'model-type-1'"

    def test_mk_model_with_record_invokes(self, prj):
        prj.deploy["record_invokes"] = True
        model_cfg = utils.mk_model(prj, "model-1", "training-job-1")
        assert (
            model_cfg["PrimaryContainer"]["Environment"]["ML2P_RECORD_INVOKES"]
            == "true"
        )

    def test_mk_endpoint_config(self, prj):
        endpoint_cfg = utils.mk_endpoint_config(prj, "endpoint-1", "model-1")
        assert endpoint_cfg == {
            "EndpointConfigName": "modelling-project-endpoint-1-config",
            "ProductionVariants": [
                {
                    "VariantName": "modelling-project-model-1-variant-1",
                    "ModelName": "modelling-project-model-1",
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.t2.medium",
                    "InitialVariantWeight": 1.0,
                }
            ],
            "Tags": [{"Key": "ml2p-project", "Value": "modelling-project"}],
        }

    def test_mk_notebook(self, prj):
        notebook_cfg_no_repo = utils.mk_notebook(prj, "notebook-1")
        assert notebook_cfg_no_repo == {
            "NotebookInstanceName": "modelling-project-notebook-1",
            "DirectInternetAccess": "Disabled",
            "InstanceType": "ml.t2.medium",
            "RoleArn": "arn:aws:iam::111111111111:role/modelling-project",
            "Tags": [{"Key": "ml2p-project", "Value": "modelling-project"}],
            "LifecycleConfigName": "modelling-project-notebook-1-lifecycle-config",
            "VolumeSizeInGB": 8,
            "SubnetId": "subnet-1",
            "SecurityGroupIds": ["sg-1"],
        }
        notebook_cfg_repo = utils.mk_notebook(
            prj, "notebook-1", repo_name="notebook-1-repo"
        )
        assert notebook_cfg_repo == {
            "NotebookInstanceName": "modelling-project-notebook-1",
            "InstanceType": "ml.t2.medium",
            "DirectInternetAccess": "Disabled",
            "RoleArn": "arn:aws:iam::111111111111:role/modelling-project",
            "Tags": [{"Key": "ml2p-project", "Value": "modelling-project"}],
            "LifecycleConfigName": "modelling-project-notebook-1-lifecycle-config",
            "VolumeSizeInGB": 8,
            "DefaultCodeRepository": "modelling-project-notebook-1-repo",
            "SubnetId": "subnet-1",
            "SecurityGroupIds": ["sg-1"],
        }

    def test_mk_notebook_no_scripts(self, prj_no_vpc):
        notebook_cfg_no_vpc = utils.mk_notebook(prj_no_vpc, "notebook-1")
        assert notebook_cfg_no_vpc == {
            "NotebookInstanceName": "modelling-project-notebook-1",
            "InstanceType": "ml.t2.medium",
            "DirectInternetAccess": "Disabled",
            "RoleArn": "arn:aws:iam::111111111111:role/modelling-project",
            "Tags": [{"Key": "ml2p-project", "Value": "modelling-project"}],
            "LifecycleConfigName": "modelling-project-notebook-1-lifecycle-config",
            "VolumeSizeInGB": 8,
        }

    def test_mk_notebook_with_direct_internet_access_enabled(self, prj):
        prj.cfg["notebook"]["direct_internet_access"] = "Enabled"
        notebook_cfg = utils.mk_notebook(prj, "notebook-1")
        assert notebook_cfg["DirectInternetAccess"] == "Enabled"

    def test_mk_notebook_with_direct_internet_access_disabled_by_default(self, prj):
        notebook_cfg = utils.mk_notebook(prj, "notebook-1")
        assert notebook_cfg["DirectInternetAccess"] == "Disabled"

    def test_mk_lifecycle_config_on_start(self, prj):
        notebook_lifecycle_cfg = utils.mk_lifecycle_config(prj, "notebook-1")
        assert (
            base64.b64decode(notebook_lifecycle_cfg["OnStart"][0]["Content"])
            == on_start_fixture()
        )

    def test_mk_lifecycle_config_on_create(self, prj):
        notebook_lifecycle_cfg = utils.mk_lifecycle_config(prj, "notebook-1")
        assert (
            base64.b64decode(notebook_lifecycle_cfg["OnCreate"][0]["Content"])
            == on_create_fixture()
        )

    def test_mk_lifecycle_config(self, prj):
        notebook_lifecycle_cfg = utils.mk_lifecycle_config(prj, "notebook-1")
        assert notebook_lifecycle_cfg == {
            "NotebookInstanceLifecycleConfigName": "modelling-project-"
            "notebook-1-lifecycle-config",
            "OnCreate": [
                {"Content": base64.b64encode(on_create_fixture()).decode("utf-8")}
            ],
            "OnStart": [
                {"Content": base64.b64encode(on_create_fixture()).decode("utf-8")}
            ],
        }

    def test_mk_lifecycle_config_no_onstart_or_oncreate(self, prj_no_vpc):
        notebook_lifecycle_cfg_no_scripts = utils.mk_lifecycle_config(
            prj_no_vpc, "notebook-1"
        )
        assert notebook_lifecycle_cfg_no_scripts == {
            "NotebookInstanceLifecycleConfigName": "modelling-project-"
            "notebook-1-lifecycle-config"
        }

    def test_mk_repo(self, prj):
        repo_cfg = utils.mk_repo(prj, "repo-1")
        assert repo_cfg == {
            "CodeRepositoryName": "modelling-project-repo-1",
            "GitConfig": {
                "RepositoryUrl": "https://github.example.com/modelling-project",
                "Branch": "master",
                "SecretArn": "arn:aws:secretsmanager:eu-west-1:111111111111:"
                "secret:sagemaker-github-authentication-fLJGfa",
            },
        }

    def test_mk_processing_job(self, prj):
        processing_job_cfg = utils.mk_processing_job(prj, "my-dataset-2022-12-25")
        assert processing_job_cfg == {
            "ProcessingJobName": "modelling-project-my-dataset-2022-12-25",
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.2xlarge",
                    "VolumeSizeInGB": 20,
                }
            },
            "AppSpecification": {
                "ImageUri": "123456789012.dkr.ecr.eu-west-1.amazonaws.com"
                "/modelling-project-sagemaker:latest",
                "ContainerArguments": ["generate-dataset"],
            },
            "Environment": {
                "ML2P_DATASET": "my-dataset-2022-12-25",
                "ML2P_PROJECT": "modelling-project",
                "ML2P_S3_URL": "s3://prodigyfinance-modelling-"
                "project-sagemaker-production/",
            },
            "NetworkConfig": {
                "EnableInterContainerTrafficEncryption": False,
                "EnableNetworkIsolation": False,
            },
            "RoleArn": "arn:aws:iam::111111111111:role/modelling-project",
            "StoppingCondition": {"MaxRuntimeInSeconds": 36000},
            "Tags": [{"Key": "ml2p-project", "Value": "modelling-project"}],
        }


class TestNamingValidation:
    def test_naming_validation_noncompliance(self):
        with pytest.raises(NamingError) as exc_info:
            utils.validate_name("a wrong name", "dataset")
        assert (
            str(exc_info.value) == "Dataset names should be in the "
            "format <model-name>-YYYYMMDD"
        )
        with pytest.raises(NamingError) as exc_info:
            utils.validate_name("a wrong name", "training-job")
        assert (
            str(exc_info.value) == "Training job names should be in the "
            "format <model-name>-X-Y-Z-[dev]"
        )
        with pytest.raises(NamingError) as exc_info:
            utils.validate_name("a wrong name", "model")
        assert (
            str(exc_info.value) == "Model names should be in the"
            " format <model-name>-X-Y-Z-[dev]"
        )
        with pytest.raises(NamingError) as exc_info:
            utils.validate_name("a wrong name", "endpoint")
        assert (
            str(exc_info.value) == "Endpoint names should be in the"
            " format <model-name>-X-Y-Z-[dev]-[live|analysis|test]"
        )

    def test_naming_validation_compliance(self):
        utils.validate_name("test-model-20191011", "dataset")
        utils.validate_name("test-model-0-0-dev", "training-job")
        utils.validate_name("test-model-0-0", "training-job")
        utils.validate_name("test-model-10-11-12", "training-job")
        utils.validate_name("test-model-0-0-0-dev", "model")
        utils.validate_name("test-model-0-0-0", "model")
        utils.validate_name("test-model-10-11-12", "model")
        utils.validate_name("test-model-0-0-0-dev", "endpoint")
        utils.validate_name("test-model-0-0-0-dev-live", "endpoint")
        utils.validate_name("test-model-0-0-0-dev-analysis", "endpoint")
        utils.validate_name("test-model-0-0-0-dev-test", "endpoint")
        utils.validate_name("test-model-0-0-0", "endpoint")
        utils.validate_name("test-model-10-11-12", "endpoint")
        utils.validate_name("test-model-0-0-0-live", "endpoint")
        utils.validate_name("test-model-0-0-0-analysis", "endpoint")
        utils.validate_name("test-model-0-0-0-test", "endpoint")


class TestTrainingJobNameForModel:
    def test_training_job_name_for_model(self):
        assert utils.training_job_name_for_model("model-1-0-2") == "model-1-0"
        assert utils.training_job_name_for_model("model-1-0-2-dev") == "model-1-0"

    def test_invalid_model_name(self):
        with pytest.raises(NamingError) as err:
            utils.training_job_name_for_model("not a valid model name")
        assert str(err.value) == "Invalid model name 'not a valid model name'"


class TestModelNameForEndpoint:
    def test_model_name_for_endpoint(self):
        assert utils.model_name_for_endpoint("model-1-0-2-live") == "model-1-0-2"
        assert (
            utils.model_name_for_endpoint("model-1-0-2-dev-live") == "model-1-0-2-dev"
        )

    def test_invalid_endpoint_name(self):
        with pytest.raises(NamingError) as err:
            utils.model_name_for_endpoint("not a valid endpoint name")
        assert str(err.value) == "Invalid endpoint name 'not a valid endpoint name'"


class TestValidateModelType:
    def test_value_in_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert utils.validate_model_type(ctx, "model_type", "model-1") == "model-1"

    def test_value_not_in_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        with pytest.raises(click.BadParameter) as err:
            utils.validate_model_type(ctx, "model_type", "model-2")
        assert str(err.value) == "Unknown model type."

    def test_value_is_none_no_model_types(self, cli_helper):
        ctx = cli_helper.ctx(models={})
        assert utils.validate_model_type(ctx, "model_type", None) is None

    def test_value_is_none_single_model_type(self, cli_helper):
        ctx = cli_helper.ctx(models={"model-1": "my_models.ml2p.Model1"})
        assert utils.validate_model_type(ctx, "model_type", None) == "model-1"

    def test_value_is_none_multiple_model_types(self, cli_helper):
        ctx = cli_helper.ctx(
            models={
                "model-1": "my_models.ml2p.Model1",
                "model-2": "my_models.ml2p.Model2",
            }
        )
        with pytest.raises(click.BadParameter) as err:
            utils.validate_model_type(ctx, "model_type", None)
        assert str(err.value) == (
            "Model type may only be omitted if zero or one models are listed"
            " in the ML2P config YAML file."
        )
