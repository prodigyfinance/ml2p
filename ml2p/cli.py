# -*- coding: utf-8 -*-

""" CLI for Minimal Lovable Machine Learning Pipeline. """

import datetime
import json
import re
import urllib.parse

import boto3
import click
import yaml


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


class ModellingProject:
    """ Object for holding CLI context. """

    def __init__(self, cfg):
        with open(cfg) as f:
            self.cfg = yaml.safe_load(f)
        self.s3 = S3Path(self.cfg["s3folder"])
        self.client = boto3.client("sagemaker")
        self.train = ModellingSubCfg(self.cfg, "train")
        self.deploy = ModellingSubCfg(self.cfg, "deploy")

    def full_job_name(self, job_name):
        return "{}-{}".format(self.cfg["project"], job_name)

    def tags(self):
        return [{"Key": "ml2p-project", "Value": self.cfg["project"]}]


class S3Path:
    """ Holder for S3 folder. """

    def __init__(self, s3folder):
        self._s3url = urllib.parse.urlparse(s3folder)
        self._s3root = self._s3url.path.strip("/")

    def bucket(self):
        return self._s3url.netloc

    def path(self, suffix):
        path = self._s3root + "/" + suffix.lstrip("/")
        return path.lstrip("/")  # handles empty s3root

    def url(self, suffix):
        return "s3://{}/{}".format(self._s3url.netloc, self.path(suffix))


class ModellingSubCfg:
    """ Holder for training or deployment config. """

    def __init__(self, cfg, section, defaults="defaults"):
        self._cfg = cfg
        self._defaults = cfg.get(defaults, {})
        self._section = cfg.get(section, {})

    def __getattr__(self, name):
        if name in self._section:
            return self._section[name]
        return self._defaults[name]


# alias pass_obj for readability
pass_prj = click.pass_obj


@click.group()
@click.option(
    "--cfg",
    default="./ml2p.yml",
    help="Project configuration file. Default: ./ml2p.yml.",
)
@click.pass_context
def ml2p(ctx, cfg):
    """ Minimal Lovable Machine Learning Pipeline.

        A friendlier interface to AWS SageMaker.
    """
    ctx.obj = ModellingProject(cfg=cfg)


@ml2p.command("init")
@pass_prj
def init(prj):
    """ Initialize the project S3 bucket. """
    client = boto3.client("s3")
    client.put_object(
        Bucket=prj.s3.bucket(),
        Key=prj.s3.path("/models/README.rst"),
        Body="Models for {}.".format(prj.cfg["project"]).encode("utf-8"),
    )
    client.put_object(
        Bucket=prj.s3.bucket(),
        Key=prj.s3.path("/datasets/README.rst"),
        Body="Datasets for {}.".format(prj.cfg["project"]).encode("utf-8"),
    )


@ml2p.group("training-job")
def training_job():
    """ Create and inspect training jobs. """


@training_job.command("list")
@pass_prj
def training_job_list(prj):
    """ List training jobs for this project.
    """
    paginator = prj.client.get_paginator("list_training_jobs")
    for page in paginator.paginate():
        for job in page["TrainingJobSummaries"]:
            if job["TrainingJobName"].startswith(prj.cfg["project"]):
                click_echo_json(job)


@training_job.command("create")
@click.argument("training_job")
@click.argument("dataset")
@pass_prj
def training_job_create(prj, training_job, dataset):
    """ Create a training job.
    """
    training_job_params = mk_training_job(prj, training_job, dataset)
    response = prj.client.create_training_job(**training_job_params)
    click_echo_json(response)


@training_job.command("describe")
@click.argument("training-job")
@pass_prj
def training_job_describe(prj, training_job):
    """ Describe a training job.
    """
    response = prj.client.describe_training_job(
        TrainingJobName=prj.full_job_name(training_job)
    )
    click_echo_json(response)


@training_job.command("wait")
@click.argument("training-job")
@pass_prj
def training_job_wait(prj, training_job):
    """ Wait for a training job to complete or stop.
    """
    waiter = prj.client.get_waiter("training_job_completed_or_stopped")
    waiter.wait(
        TrainingJobName=prj.full_job_name(training_job),
        WaiterConfig={"Delay": 10, "MaxAttempts": 30},
    )


@ml2p.group("model")
def model():
    """ Create and inspect models. """


@model.command("list")
@pass_prj
def model_list(prj):
    """ List models for this project.
    """
    paginator = prj.client.get_paginator("list_models")
    for page in paginator.paginate():
        for job in page["Models"]:
            if job["ModelName"].startswith(prj.cfg["project"]):
                click_echo_json(job)


@model.command("create")
@click.argument("model-name")
@click.argument("training-job")
@pass_prj
def model_create(prj, model_name, training_job):
    """ Create a model.
    """
    model_params = mk_model(prj, model_name, training_job)
    response = prj.client.create_model(**model_params)
    click_echo_json(response)


@model.command("delete")
@click.argument("model-name")
@pass_prj
def model_delete(prj, model_name):
    """ Delete a model.
    """
    full_model_name = prj.full_job_name(model_name)
    response = prj.client.delete_model(ModelName=full_model_name)
    click_echo_json(response)


@model.command("describe")
@click.argument("model-name")
@pass_prj
def model_describe(prj, model_name):
    """ Describe a model.
    """
    response = prj.client.describe_model(ModelName=prj.full_job_name(model_name))
    click_echo_json(response)


@ml2p.group("endpoint")
def endpoint():
    """ Create and inspect endpoints. """


@endpoint.command("list")
@pass_prj
def endpoint_list(prj):
    """ List endpoints for this project.
    """
    paginator = prj.client.get_paginator("list_endpoints")
    for page in paginator.paginate():
        for job in page["Endpoints"]:
            if job["EndpointName"].startswith(prj.cfg["project"]):
                click_echo_json(job)


@endpoint.command("create")
@click.argument("endpoint-name")
@click.argument("model-name")
@pass_prj
def endpoint_create(prj, endpoint_name, model_name):
    """ Create an endpoint for a model.
    """
    endpoint_config_params = mk_endpoint_config(prj, endpoint_name, model_name)
    prj.client.create_endpoint_config(**endpoint_config_params)
    response = prj.client.create_endpoint(
        EndpointName=prj.full_job_name(endpoint_name),
        EndpointConfigName=endpoint_config_params["EndpointConfigName"],
        Tags=prj.tags(),
    )
    click_echo_json(response)


@endpoint.command("delete")
@click.argument("endpoint-name")
@pass_prj
def endpoint_delete(prj, endpoint_name):
    """ Delete an endpoint.
    """
    full_endpoint_name = prj.full_job_name(endpoint_name)
    full_endpoint_config_name = prj.full_job_name(endpoint_name) + "-config"
    response = prj.client.delete_endpoint(EndpointName=full_endpoint_name)
    click_echo_json(response)
    response = prj.client.delete_endpoint_config(
        EndpointConfigName=full_endpoint_config_name
    )
    click_echo_json(response)


@endpoint.command("describe")
@click.argument("endpoint-name")
@pass_prj
def endpoint_describe(prj, endpoint_name):
    """ Describe an endpoint.
    """
    response = prj.client.describe_endpoint(
        EndpointName=prj.full_job_name(endpoint_name)
    )
    response["EndpointUrl"] = endpoint_url_for_arn(response["EndpointArn"])
    click_echo_json(response)


@endpoint.command("wait")
@click.argument("endpoint-name")
@pass_prj
def endpoint_wait(prj, endpoint_name):
    """ Wait for an endpoint to be ready or dead.
    """
    waiter = prj.client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=prj.full_job_name(endpoint_name),
        WaiterConfig={"Delay": 10, "MaxAttempts": 30},
    )


@endpoint.command("invoke")
@click.argument("endpoint-name")
@click.argument("json-data")
@pass_prj
def endpoint_invoke(prj, endpoint_name, json_data):
    """ Invoke an endpoint (i.e. make a prediction).
    """
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=prj.full_job_name(endpoint_name),
        Body=json_data.encode("utf-8"),
        ContentType="application/json",
        Accept="application/json",
    )
    response["Body"] = json.loads(response["Body"].read().decode("utf-8"))
    click_echo_json(response)
