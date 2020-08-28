# -*- coding: utf-8 -*-

""" CLI for Minimal Lovable Machine Learning Pipeline. """

import json
import pathlib

import boto3
import click

from . import __version__ as ml2p_version
from . import cli_utils
from .cli_utils import click_echo_json, validate_name
from .core import ModellingProject


def validate_model_type(ctx, param, value):
    """ Custom validator for --model-type. """
    model_types = ctx.obj.models.keys()
    if value is not None:
        if model_types and value not in model_types:
            raise click.BadParameter("Unknown model type.")
        return value
    if len(model_types) == 0:
        return None
    if len(model_types) == 1:
        return model_types[0]
    raise click.BadParameter(
        "Model type may only be omitted if zero or one models are listed in the ML2P"
        " config YAML file."
    )


class ModellingProjectWithSagemakerClient(ModellingProject):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.client = boto3.client("sagemaker")


# alias pass_obj for readability
pass_prj = click.pass_obj


@click.group()
@click.option(
    "--cfg",
    default="./ml2p.yml",
    help="Project configuration file. Default: ./ml2p.yml.",
)
@click.version_option(version=ml2p_version)
@click.pass_context
def ml2p(ctx, cfg):
    """ Minimal Lovable Machine Learning Pipeline.

        A friendlier interface to AWS SageMaker.
    """
    ctx.obj = ModellingProjectWithSagemakerClient(cfg=cfg)


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


@ml2p.group("dataset")
def dataset():
    """ Create and manage datasets. """


@dataset.command("list")
@pass_prj
def dataset_list(prj):
    """ List datasets for this project.
    """
    client = boto3.client("s3")
    prefix = prj.s3.path("/datasets/")
    len_prefix = len(prefix)
    response = client.list_objects_v2(
        Bucket=prj.s3.bucket(), Prefix=prefix, Delimiter="/"
    )
    for item in response["CommonPrefixes"]:
        dataset = item["Prefix"][len_prefix:].rstrip("/")
        click_echo_json(dataset)


@dataset.command("create")
@click.argument("dataset")
@pass_prj
def dataset_create(prj, dataset):
    """ Create a dataset.
    """
    validate_name(dataset, "dataset")
    client = boto3.client("s3")
    client.put_object(
        Bucket=prj.s3.bucket(),
        Key=prj.s3.path("/datasets/{}/README.rst".format(dataset)),
        Body="Dataset {} for project {}.".format(dataset, prj.cfg["project"]).encode(
            "utf-8"
        ),
    )


@dataset.command("delete")
@click.argument("dataset")
@pass_prj
def dataset_delete(prj, dataset):
    """ Delete a dataset.
    """
    validate_name(dataset, "dataset")
    client = boto3.client("s3")
    prefix = prj.s3.path("/datasets/{}/".format(dataset))
    response = client.list_objects_v2(Bucket=prj.s3.bucket(), Prefix=prefix)
    objects_to_delete = [{"Key": item["Key"]} for item in response["Contents"]]
    client.delete_objects(
        Bucket=prj.s3.bucket(), Delete={"Objects": objects_to_delete},
    )


@dataset.command("ls")
@click.argument("dataset")
@pass_prj
def dataset_ls(prj, dataset):
    """ List the contents of a dataset.
    """
    validate_name(dataset, "dataset")
    client = boto3.client("s3")
    prefix = prj.s3.path("/datasets/{}/".format(dataset))
    len_prefix = len(prefix)
    response = client.list_objects_v2(Bucket=prj.s3.bucket(), Prefix=prefix)
    for item in response["Contents"]:
        filename = item["Key"][len_prefix:]
        filesize = item["Size"]
        click_echo_json({"filename": filename, "size": filesize})


@dataset.command("up")
@click.argument("dataset")
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.argument("dst", required=False, default=None)
@pass_prj
def dataset_up(prj, dataset, src, dst):
    """ Upload a file SRC to a dataset as DST.

        If DST is omitted, the source file is uploaded under its own name.
    """
    validate_name(dataset, "dataset")
    filepath = pathlib.Path(src)
    if dst is None:
        dst = filepath.name
    client = boto3.client("s3")
    with filepath.open("rb") as f:
        client.upload_fileobj(
            Fileobj=f,
            Bucket=prj.s3.bucket(),
            Key=prj.s3.path("/datasets/{}/{}".format(dataset, dst)),
        )


@dataset.command("dn")
@click.argument("dataset")
@click.argument("src")
@click.argument(
    "dst",
    type=click.Path(exists=False, writable=True, dir_okay=False),
    required=False,
    default=None,
)
@pass_prj
def dataset_dn(prj, dataset, src, dst):
    """ Download a file SRC from the dataset and save it in DST.

        If DST is omitted, the source file is downloaded as its own name.
    """
    validate_name(dataset, "dataset")
    if dst is None:
        dst = src
    download_path = pathlib.Path(dst)
    client = boto3.client("s3")
    with download_path.open("wb") as f:
        client.download_fileobj(
            Bucket=prj.s3.bucket(),
            Key=prj.s3.path("/datasets/{}/{}".format(dataset, src)),
            Fileobj=f,
        )


@dataset.command("rm")
@click.argument("dataset")
@click.argument("filename")
@pass_prj
def dataset_rm(prj, dataset, filename):
    """ Delete a file from a dataset.
    """
    validate_name(dataset, "dataset")
    client = boto3.client("s3")
    client.delete_object(
        Bucket=prj.s3.bucket(),
        Key=prj.s3.path("/datasets/{}/{}".format(dataset, filename)),
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
@click.option(
    "--model-type",
    "-m",
    default=None,
    callback=validate_model_type,
    help="The name of the type of model.",
)
@pass_prj
def training_job_create(prj, training_job, dataset, model_type):
    """ Create a training job.
    """
    validate_name(training_job, "training-job")
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
@click.option(
    "--training-job",
    "-t",
    default=None,
    help=(
        "The name of the training job to base the model on. Defaults to the model name"
        " without the patch version number."
    ),
)
@click.option(
    "--model-type",
    "-m",
    default=None,
    callback=validate_model_type,
    help="The name of the type of model.",
)
@pass_prj
def model_create(prj, model_name, training_job, model_type):
    """ Create a model.
    """
    validate_name(model_name, "model")
    if training_job is None:
        training_job = cli_utils.training_job_name_for_model(model_name)
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
@click.option(
    "--model-name",
    "-m",
    default=None,
    help=(
        "The name of the model to base the endpoint on. Defaults to the endpoint name"
        " without the live/analysis/test suffix."
    ),
)
@pass_prj
def endpoint_create(prj, endpoint_name, model_name):
    """ Create an endpoint for a model.
    """
    validate_name(endpoint_name, "endpoint")
    if model_name is None:
        model_name = cli_utils.model_name_for_endpoint(endpoint_name)
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
    """ Create and manage notebooks. """


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
def notebook_presigned_url(prj, notebook_name):
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
