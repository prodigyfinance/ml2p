# -*- coding: utf-8 -*-

""" CLI commands for interating with SageMaker models. """

import click

from .utils import (
    click_echo_json,
    mk_model,
    training_job_name_for_model,
    validate_model_type,
    validate_name,
)


@click.group("model")
def model():
    """Create and inspect models."""


@model.command("list")
@click.pass_obj
def model_list(prj):
    """List models for this project."""
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
@click.pass_obj
def model_create(prj, model_name, training_job, model_type):
    """Create a model."""
    validate_name(model_name, "model")
    if training_job is None:
        training_job = training_job_name_for_model(model_name)
    model_params = mk_model(prj, model_name, training_job, model_type)
    response = prj.client.create_model(**model_params)
    click_echo_json(response)


@model.command("delete")
@click.argument("model-name")
@click.pass_obj
def model_delete(prj, model_name):
    """Delete a model."""
    full_model_name = prj.full_job_name(model_name)
    response = prj.client.delete_model(ModelName=full_model_name)
    click_echo_json(response)


@model.command("describe")
@click.argument("model-name")
@click.pass_obj
def model_describe(prj, model_name):
    """Describe a model."""
    response = prj.client.describe_model(ModelName=prj.full_job_name(model_name))
    click_echo_json(response)
