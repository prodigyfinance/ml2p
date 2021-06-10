# -*- coding: utf-8 -*-

""" Helper functions for the ml2p CLI. """

import base64
import datetime
import json
import re

import click

from . import errors, hyperparameters


def date_to_string_serializer(value):
    """ JSON serializer for datetime objects. """
    if isinstance(value, datetime.datetime):
        return str(value)
    raise TypeError("Serializing {!r} to JSON not supported.".format(value))


def click_echo_json(response):
    """ Echo JSON via click.echo. """
    click.echo(json.dumps(response, indent=2, default=date_to_string_serializer))


def endpoint_url_for_arn(endpoint_arn):
    """ Return the URL for an endpoint ARN. """
    match = re.match(
        r"^arn:aws:sagemaker:(?P<region>[^:]*):(?P<account>[^:]*):"
        r"endpoint/(?P<endpoint>.*)$",
        endpoint_arn,
    )
    if not match:
        return None
    return (
        "https://runtime.sagemaker.{region}.amazonaws.com/"
        "endpoints/{endpoint}/invocations".format(**match.groupdict())
    )


def mk_vpc_config(subcfg):
    """ Parse VPC configuration for training job or model creation.
    """
    vpc_config = subcfg.get("vpc_config", None)
    if vpc_config is None:
        return None
    if (
        not isinstance(vpc_config, dict)
        or not isinstance(vpc_config.get("security_groups"), list)
        or not isinstance(vpc_config.get("subnets"), list)
    ):
        raise errors.ConfigError(
            "The vpc_config requires a dictionary with keys 'security_groups'"
            " and 'subnets'. Both the security_groups and subnets should contain lists"
            " of IDs."
        )
    security_groups = vpc_config["security_groups"]
    if not security_groups:
        raise errors.ConfigError(
            "The vpc_config must contain at least one security group id."
        )
    subnets = vpc_config["subnets"]
    if not subnets:
        raise errors.ConfigError("The vpc_config must contain at least one subnet id.")
    return {
        "SecurityGroupIds": security_groups,
        "Subnets": subnets,
    }


def mk_training_job(prj, training_job, dataset, model_type=None):
    """ Return training job creation parameters. """
    model_path = prj.s3.url("/models/")
    train_path = prj.s3.url("/datasets/" + dataset)
    extra_env = {}
    if model_type is not None:
        extra_env["ML2P_MODEL_CLS"] = prj.models[model_type]
    extra_training_params = {}
    vpc_config = mk_vpc_config(prj.train)
    if vpc_config is not None:
        extra_training_params["VpcConfig"] = vpc_config
    return {
        "TrainingJobName": prj.full_job_name(training_job),
        "AlgorithmSpecification": {
            "TrainingImage": prj.train.image,
            "TrainingInputMode": "File",
        },
        # training shouldn't make network calls
        "EnableNetworkIsolation": True,
        "HyperParameters": hyperparameters.encode(
            {
                "ML2P_ENV": {
                    "ML2P_PROJECT": prj.project,
                    "ML2P_S3_URL": prj.s3.url(),
                    **extra_env,
                }
            }
        ),
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": train_path}
                },
            }
        ],
        "OutputDataConfig": {"S3OutputPath": model_path},
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": prj.train.instance_type,
            "VolumeSizeInGB": 20,
        },
        "RoleArn": prj.train.role,
        "StoppingCondition": {"MaxRuntimeInSeconds": 60 * 60},
        "Tags": prj.tags(),
        **extra_training_params,
    }


def mk_model(prj, model_name, training_job, model_type=None):
    """ Return model creation parameters. """
    model_path = prj.s3.url("/models")
    model_tgz_path = (
        model_path + "/" + prj.full_job_name(training_job) + "/output/model.tar.gz"
    )
    extra_env = {}
    if model_type is not None:
        extra_env["ML2P_MODEL_CLS"] = prj.models[model_type]
    if prj.deploy.get("record_invokes", False):
        extra_env["ML2P_RECORD_INVOKES"] = "true"
    extra_model_params = {}
    vpc_config = mk_vpc_config(prj.deploy)
    if vpc_config is not None:
        extra_model_params["VpcConfig"] = vpc_config
    return {
        "ModelName": prj.full_job_name(model_name),
        "PrimaryContainer": {
            "Image": prj.deploy.image,
            "ModelDataUrl": model_tgz_path,
            "Environment": {
                "ML2P_MODEL_VERSION": prj.full_job_name(model_name),
                "ML2P_PROJECT": prj.project,
                "ML2P_S3_URL": prj.s3.url(),
                **extra_env,
            },
        },
        "ExecutionRoleArn": prj.deploy.role,
        "Tags": prj.tags(),
        "EnableNetworkIsolation": False,
        **extra_model_params,
    }


def mk_endpoint_config(prj, endpoint_name, model_name):
    """ Return endpoint config creation parameters. """
    return {
        "EndpointConfigName": prj.full_job_name(endpoint_name) + "-config",
        "ProductionVariants": [
            {
                "VariantName": prj.full_job_name(model_name) + "-variant-1",
                "ModelName": prj.full_job_name(model_name),
                "InitialInstanceCount": 1,
                "InstanceType": prj.deploy.instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
        "Tags": prj.tags(),
    }


def mk_notebook(prj, notebook_name, repo_name=None):
    """ Return a notebook configuration. """
    notebook_params = {
        "NotebookInstanceName": prj.full_job_name(notebook_name),
        "InstanceType": prj.notebook.instance_type,
        "RoleArn": prj.notebook.role,
        "Tags": prj.tags(),
        "LifecycleConfigName": prj.full_job_name(notebook_name) + "-lifecycle-config",
        "VolumeSizeInGB": prj.notebook.volume_size,
        "DirectInternetAccess": prj.notebook.get("direct_internet_access", "Disabled"),
    }
    if repo_name is not None:
        notebook_params["DefaultCodeRepository"] = prj.full_job_name(repo_name)
    if prj.notebook.get("subnet_id"):
        notebook_params["SubnetId"] = prj.notebook.subnet_id
    if prj.notebook.get("security_group_ids"):
        notebook_params["SecurityGroupIds"] = prj.notebook.security_group_ids
    return notebook_params


def mk_lifecycle_config(prj, notebook_name):
    """ Return a notebook instance lifecycle configuration. """
    lifecycle_config = {
        "NotebookInstanceLifecycleConfigName": prj.full_job_name(notebook_name)
        + "-lifecycle-config"
    }
    if prj.notebook.get("on_start"):
        with open(prj.notebook.on_start, "r") as f:
            on_start = f.read()
        on_start = base64.b64encode(on_start.encode("utf-8")).decode("utf-8")
        lifecycle_config["OnStart"] = [{"Content": on_start}]
    if prj.notebook.get("on_create"):
        with open(prj.notebook.on_create, "r") as f:
            on_create = f.read()
        on_create = base64.b64encode(on_create.encode("utf-8")).decode("utf-8")
        lifecycle_config["OnCreate"] = [{"Content": on_create}]
    return lifecycle_config


def mk_repo(prj, repo_name):
    """ Return parameters for creating a repo. """
    return {
        "CodeRepositoryName": prj.full_job_name(repo_name),
        "GitConfig": {
            "RepositoryUrl": prj.notebook.repo_url,
            "Branch": prj.notebook.repo_branch,
            "SecretArn": prj.notebook.repo_secret_arn,
        },
    }


VALIDATION_REGEXES = {
    "dataset": r"^(?P<model>[a-zA-Z0-9\-]+)-(?P<date>[0-9]{8})$",
    "training-job": (
        r"^(?P<model>[a-zA-Z0-9\-]+)-(?P<major>[0-9]+)-(?P<minor>[0-9]+)"
        r"(?P<training_suffix>\-dev)?$"
    ),
    "model": (
        r"^(?P<model>[a-zA-Z0-9\-]+)-(?P<major>[0-9]+)-(?P<minor>[0-9]+)"
        r"-(?P<patch>[0-9]+)(?P<model_suffix>\-dev)?$"
    ),
    "endpoint": (
        r"^(?P<model>[a-zA-Z0-9\-]+)-(?P<major>[0-9]+)-(?P<minor>[0-9]+)"
        r"-(?P<patch>[0-9]+)"
        r"(?P<model_suffix>\-dev)?(?P<endpoint_suffix>\-(live|analysis|test))?$"
    ),
}


def validate_name(name, resource):
    """ Validate that the name of the SageMaker resource complies with
        convention.

        :param str name:
            The name of the SageMaker resource to validate.
        :param str resource:
            The type of SageMaker resource to validate. One of "dataset",
            "training-job", "model", "endpoint".
    """
    message_dict = {
        "dataset": "Dataset names should be in the format <model-name>-YYYYMMDD",
        "training-job": "Training job names should be in the"
        " format <model-name>-X-Y-Z-[dev]",
        "model": "Model names should be in the format <model-name>-X-Y-Z-[dev]",
        "endpoint": "Endpoint names should be in the"
        " format <model-name>-X-Y-Z-[dev]-[live|analysis|test]",
    }
    if re.match(VALIDATION_REGEXES[resource], name) is None:
        raise errors.NamingError(message_dict[resource])


def training_job_name_for_model(model_name):
    """ Return a default training job name for the given model. """
    match = re.match(VALIDATION_REGEXES["model"], model_name)
    if match is None:
        raise errors.NamingError("Invalid model name {!r}".format(model_name))
    grps = match.groupdict()
    return "{model}-{major}-{minor}".format(**grps)


def model_name_for_endpoint(endpoint_name):
    """ Return a default model name for the given endpoint. """
    match = re.match(VALIDATION_REGEXES["endpoint"], endpoint_name)
    if match is None:
        raise errors.NamingError("Invalid endpoint name {!r}".format(endpoint_name))
    grps = match.groupdict()
    grps["model_suffix"] = grps["model_suffix"] or ""
    return "{model}-{major}-{minor}-{patch}{model_suffix}".format(**grps)
