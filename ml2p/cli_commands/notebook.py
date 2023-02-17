# -*- coding: utf-8 -*-

""" CLI commands for interating with SageMaker notebooks. """

import click

from .utils import click_echo_json, mk_lifecycle_config, mk_notebook, mk_repo


@click.group("notebook")
def notebook():
    """Create and manage notebooks."""


@notebook.command("list")
@click.pass_obj
def notebook_list(prj):
    paginator = prj.client.get_paginator("list_notebook_instances")
    for page in paginator.paginate():
        for notebook in page["NotebookInstances"]:
            if notebook["NotebookInstanceName"].startswith(prj.cfg["project"]):
                click_echo_json(notebook)


@notebook.command("create")
@click.argument("notebook-name")
@click.pass_obj
def notebook_create(prj, notebook_name):
    """Create a notebook instance."""
    notebook_instance_lifecycle_config = mk_lifecycle_config(prj, notebook_name)
    repo_name = None
    if prj.notebook.get("repo_url"):
        repo_name = "{}-repo".format(notebook_name)
        repo_params = mk_repo(prj, repo_name)
        prj.client.create_code_repository(**repo_params)
    prj.client.create_notebook_instance_lifecycle_config(
        **notebook_instance_lifecycle_config
    )
    notebook_params = mk_notebook(prj, notebook_name, repo_name=repo_name)
    response = prj.client.create_notebook_instance(**notebook_params)
    click_echo_json(response)


@notebook.command("presigned-url")
@click.argument("notebook-name")
@click.pass_obj
def notebook_presigned_url(prj, notebook_name):
    """Create a URL to connect to the Jupyter server from a notebook instance."""
    response = prj.client.create_presigned_notebook_instance_url(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
    click_echo_json(response)


@notebook.command("describe")
@click.argument("notebook-name")
@click.pass_obj
def notebook_describe(prj, notebook_name):
    """Describe a notebook instance."""
    response = prj.client.describe_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
    click_echo_json(response)


@notebook.command("delete")
@click.argument("notebook-name")
@click.pass_obj
def notebook_delete(prj, notebook_name):
    """Delete a notebook instance."""
    describe_response = prj.client.describe_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
    repo_name = describe_response.get("DefaultCodeRepository", None)
    if repo_name:
        prj.client.delete_code_repository(CodeRepositoryName=repo_name)
    prj.client.delete_notebook_instance_lifecycle_config(
        NotebookInstanceLifecycleConfigName=prj.full_job_name(notebook_name)
        + "-lifecycle-config"
    )
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
    click_echo_json(response)


@notebook.command("stop")
@click.argument("notebook-name")
@click.pass_obj
def notebook_stop(prj, notebook_name):
    """Stop a notebook instance."""
    prj.client.stop_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )


@notebook.command("start")
@click.argument("notebook-name")
@click.pass_obj
def notebook_start(prj, notebook_name):
    """Start a notebook instance."""
    prj.client.start_notebook_instance(
        NotebookInstanceName=prj.full_job_name(notebook_name)
    )
