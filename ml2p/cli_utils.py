# -*- coding: utf-8 -*-

""" Helper functions for the ml2p CLI. """

import base64
import datetime
import json
import re

import click


def date_to_string_serializer(value):
    """ JSON serializer for datetime objects. """
    if isinstance(value, datetime.datetime):
        return str(value)
    raise TypeError()


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


def mk_training_job(prj, training_job, dataset):
    """ Return training job creation parameters. """
    model_path = prj.s3.url("/models/")
    train_path = prj.s3.url("/datasets/" + dataset)
    return {
        "TrainingJobName": prj.full_job_name(training_job),
        "AlgorithmSpecification": {
            "TrainingImage": prj.train.image,
            "TrainingInputMode": "File",
        },
        # training shouldn't make network calls
        "EnableNetworkIsolation": True,
        "HyperParameters": {},
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
        # TODO: specify security groups
        # "VpcConfig": {"SecurityGroupIds": ["XXX"], "Subnets": ["XXX"]},
    }


def mk_model(prj, model_name, training_job):
    """ Return model creation parameters. """
    model_path = prj.s3.url("/models")
    model_tgz_path = (
        model_path + "/" + prj.full_job_name(training_job) + "/output/model.tar.gz"
    )
    return {
        "ModelName": prj.full_job_name(model_name),
        "PrimaryContainer": {
            "Image": prj.deploy.image,
            "ModelDataUrl": model_tgz_path,
            "Environment": {"ML2P_MODEL_VERSION": prj.full_job_name(model_name)},
        },
        "ExecutionRoleArn": prj.deploy.role,
        "Tags": prj.tags(),
        # TODO: specify security groups
        # "VpcConfig": {"SecurityGroupIds": ["XXX"], "Subnets": ["XXX"]},
        "EnableNetworkIsolation": False,
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
    }
    if repo_name is not None:
        notebook_params["DefaultCodeRepository"] = prj.full_job_name(repo_name)
    if prj.get("subnet_id"):
        notebook_params["SubnetId"] = prj.notebook.subnet_id
    if prj.get("security_group_ids"):
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
        on_start = base64.b64encode(prj.notebook.on_start.encode("utf-8")).decode(
            "utf-8"
        )
        lifecycle_config["OnStart"] = [{"Content": on_start}]
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
