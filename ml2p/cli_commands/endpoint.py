# -*- coding: utf-8 -*-

""" CLI commands for interating with SageMaker endpoints. """

import json

import boto3
import click

from .utils import (
    click_echo_json,
    endpoint_url_for_arn,
    mk_endpoint_config,
    model_name_for_endpoint,
    validate_name,
)


@click.group("endpoint")
def endpoint():
    """Create and inspect endpoints."""


@endpoint.command("list")
@click.pass_obj
def endpoint_list(prj):
    """List endpoints for this project."""
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
@click.pass_obj
def endpoint_create(prj, endpoint_name, model_name):
    """Create an endpoint for a model."""
    validate_name(endpoint_name, "endpoint")
    if model_name is None:
        model_name = model_name_for_endpoint(endpoint_name)
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
@click.pass_obj
def endpoint_delete(prj, endpoint_name):
    """Delete an endpoint."""
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
@click.pass_obj
def endpoint_describe(prj, endpoint_name):
    """Describe an endpoint."""
    response = prj.client.describe_endpoint(
        EndpointName=prj.full_job_name(endpoint_name)
    )
    response["EndpointUrl"] = endpoint_url_for_arn(response["EndpointArn"])
    click_echo_json(response)


@endpoint.command("wait")
@click.argument("endpoint-name")
@click.pass_obj
def endpoint_wait(prj, endpoint_name):
    """Wait for an endpoint to be ready or dead."""
    waiter = prj.client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=prj.full_job_name(endpoint_name),
        WaiterConfig={"Delay": 10, "MaxAttempts": 30},
    )


@endpoint.command("invoke")
@click.argument("endpoint-name")
@click.argument("json-data")
@click.pass_obj
def endpoint_invoke(prj, endpoint_name, json_data):
    """Invoke an endpoint (i.e. make a prediction)."""
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=prj.full_job_name(endpoint_name),
        Body=json_data.encode("utf-8"),
        ContentType="application/json",
        Accept="application/json",
    )
    response["Body"] = json.loads(response["Body"].read().decode("utf-8"))
    click_echo_json(response)
