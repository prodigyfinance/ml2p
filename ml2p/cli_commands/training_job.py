# -*- coding: utf-8 -*-

""" CLI commands for interating with SageMaker training jobs. """

import click

from .utils import click_echo_json, mk_training_job, validate_model_type, validate_name


@click.group("training-job")
def training_job():
    """Create and inspect training jobs."""


@training_job.command("list")
@click.pass_obj
def training_job_list(prj):
    """List training jobs for this project."""
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
@click.pass_obj
def training_job_create(prj, training_job, dataset, model_type):
    """Create a training job."""
    validate_name(training_job, "training-job")
    training_job_params = mk_training_job(prj, training_job, dataset, model_type)
    response = prj.client.create_training_job(**training_job_params)
    click_echo_json(response)


@training_job.command("describe")
@click.argument("training-job")
@click.pass_obj
def training_job_describe(prj, training_job):
    """Describe a training job."""
    response = prj.client.describe_training_job(
        TrainingJobName=prj.full_job_name(training_job)
    )
    click_echo_json(response)


@training_job.command("wait")
@click.argument("training-job")
@click.pass_obj
def training_job_wait(prj, training_job):
    """Wait for a training job to complete or stop."""
    waiter = prj.client.get_waiter("training_job_completed_or_stopped")
    waiter.wait(
        TrainingJobName=prj.full_job_name(training_job),
        WaiterConfig={"Delay": 10, "MaxAttempts": 30},
    )
