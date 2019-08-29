# -*- coding: utf-8 -*-

""" CLI for Minimal Lovable Machine Learning Pipeline. """

import json

import boto3
import click
import yaml

from . import cli_utils
from .cli_utils import click_echo_json
from .core import S3URL


class ModellingProject:
    """ Object for holding CLI context. """

    def __init__(self, cfg):
        with open(cfg) as f:
            self.cfg = yaml.safe_load(f)
        self.project = self.cfg["project"]
        self.s3 = S3URL(self.cfg["s3folder"])
        self.client = boto3.client("sagemaker")
        self.train = ModellingSubCfg(self.cfg, "train")
        self.deploy = ModellingSubCfg(self.cfg, "deploy")
        self.notebook = ModellingSubCfg(self.cfg, "notebook")
        self.models = ModellingSubCfg(self.cfg, "models", defaults="models")

    def full_job_name(self, job_name):
        return "{}-{}".format(self.project, job_name)

    def tags(self):
        return [{"Key": "ml2p-project", "Value": self.cfg["project"]}]


class ModellingSubCfg:
    """ Holder for training or deployment config. """

    def __init__(self, cfg, section, defaults="defaults"):
        self._defaults = cfg.get(defaults, {})
        self._section = cfg.get(section, {})

    def __getattr__(self, name):
        if name in self._section:
            return self._section[name]
        return self._defaults[name]

    def __getitem__(self, name):
        if name in self._section:
            return self._section[name]
        return self._defaults[name]

    def __setitem__(self, name, value):
        self._section[name] = value

    def get(self, name, default=None):
        if name in self._section:
            return self._section[name]
        return self._defaults.get(name, default)


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
@click.option("--model-type", "-m", default=None, help="The name of the type of model.")
@pass_prj
def training_job_create(prj, training_job, dataset, model_type):
    """ Create a training job.
    """
    training_job_params = cli_utils.mk_training_job(
        prj, training_job, dataset, model_type
    )
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
@click.option("--model-type", "-m", default=None, help="The name of the type of model.")
@pass_prj
def model_create(prj, model_name, training_job, model_type):
    """ Create a model.
    """
    model_params = cli_utils.mk_model(prj, model_name, training_job, model_type)
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
    endpoint_config_params = cli_utils.mk_endpoint_config(
        prj, endpoint_name, model_name
    )
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
    response["EndpointUrl"] = cli_utils.endpoint_url_for_arn(response["EndpointArn"])
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


@ml2p.group("notebook")
def notebook():
    """ Create notebooks. """


@notebook.command("list")
@pass_prj
def notebook_list(prj):
    paginator = prj.client.get_paginator("list_notebook_instances")
    for page in paginator.paginate():
        for notebook in page["NotebookInstances"]:
            if notebook["NotebookInstanceName"].startswith(prj.cfg["project"]):
                click_echo_json(notebook)


@notebook.command("create")
@click.argument("notebook-name")
@pass_prj
def notebook_create(prj, notebook_name):
    """ Create a notebook instance.
    """
    notebook_instance_lifecycle_config = cli_utils.mk_lifecycle_config(
        prj, notebook_name
    )
    repo_name = None
    if prj.notebook.get("repo_url"):
        repo_name = "{}-repo".format(notebook_name)
        repo_params = cli_utils.mk_repo(prj, repo_name)
        prj.client.create_code_repository(**repo_params)
    prj.client.create_notebook_instance_lifecycle_config(
        **notebook_instance_lifecycle_config
    )
    notebook_params = cli_utils.mk_notebook(prj, notebook_name, repo_name=repo_name)
    response = prj.client.create_notebook_instance(**notebook_params)
    click_echo_json(response)


@notebook.command("presigned-url")
@click.argument("notebook-name")
@pass_prj
def presigned_url(prj, notebook_name):
    """ Create a URL to connect to the Jupyter server from a notebook instance.
    """
    response = prj.client.create_presigned_notebook_instance_url(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
    click_echo_json(response)


@notebook.command("describe")
@click.argument("notebook-name")
@pass_prj
def notebook_describe(prj, notebook_name):
    """ Describe a notebook instance.
    """
    response = prj.client.describe_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
    click_echo_json(response)


@notebook.command("delete")
@click.argument("notebook-name")
@pass_prj
def notebook_delete(prj, notebook_name):
    """ Delete a notebook instance.
    """
    describe_response = prj.client.describe_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
    repo_name = describe_response["DefaultCodeRepository"]
    if describe_response["NotebookInstanceStatus"] == "InService":
        prj.client.stop_notebook_instance(
            NotebookInstanceName=prj.full_job_name(notebook_name)
        )
        waiter = prj.client.get_waiter("notebook_instance_stopped")
        waiter.wait(
            NotebookInstanceName=prj.full_job_name(notebook_name),
            WaiterConfig={"Delay": 30, "MaxAttempts": 30},
        )
    response = prj.client.delete_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
    prj.client.delete_notebook_instance_lifecycle_config(
        NotebookInstanceLifecycleConfigName=prj.full_job_name(notebook_name)
        + "-lifecycle-config"
    )
    if repo_name:
        prj.client.delete_code_repository(CodeRepositoryName=repo_name)
    click_echo_json(response)


@notebook.command("stop")
@click.argument("notebook-name")
@pass_prj
def notebook_stop(prj, notebook_name):
    """ Stop a notebook instance.
    """
    prj.client.stop_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )


@notebook.command("start")
@click.argument("notebook-name")
@pass_prj
def notebook_start(prj, notebook_name):
    """ Start a notebook instance.
    """
    prj.client.start_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )


@ml2p.group("repo")
def repo():
    """ Describe and list code repositories. """


@repo.command("list")
@pass_prj
def repo_list(prj):
    """ List code repositories. """
    paginator = prj.client.get_paginator("list_code_repositories")
    for page in paginator.paginate():
        for repo in page["CodeRepositorySummaryList"]:
            if repo["CodeRepositoryName"].startswith(prj.cfg["project"]):
                click_echo_json(repo)


@repo.command("describe")
@click.argument("repo-name")
@pass_prj
def repo_describe(prj, repo_name):
    """ Describe a code repository SageMaker resource.
    """
    response = prj.client.describe_code_repository(
        CodeRepositoryName=prj.full_job_name(repo_name)
    )
    click_echo_json(response)
