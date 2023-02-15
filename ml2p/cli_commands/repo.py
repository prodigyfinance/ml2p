# -*- coding: utf-8 -*-

""" CLI commands for interating with repositories. """

import click

from .utils import click_echo_json


@click.group("repo")
def repo():
    """Describe and list code repositories."""


@repo.command("list")
@click.pass_obj
def repo_list(prj):
    """List code repositories."""
    paginator = prj.client.get_paginator("list_code_repositories")
    for page in paginator.paginate():
        for repo in page["CodeRepositorySummaryList"]:
            if repo["CodeRepositoryName"].startswith(prj.cfg["project"]):
                click_echo_json(repo)


@repo.command("describe")
@click.argument("repo-name")
@click.pass_obj
def repo_describe(prj, repo_name):
    """Describe a code repository SageMaker resource."""
    response = prj.client.describe_code_repository(
        CodeRepositoryName=prj.full_job_name(repo_name)
    )
    click_echo_json(response)
