# -*- coding: utf-8 -*-

""" CLI for Minimal Lovable Machine Learning Pipeline. """

import boto3
import click

from . import __version__ as ml2p_version
from .cli_commands import dataset, endpoint, init, model, notebook, repo, training_job
from .core import ModellingProject


class ModellingProjectWithSagemakerClient(ModellingProject):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.client = boto3.client("sagemaker")


@click.group()
@click.option(
    "--cfg",
    default="./ml2p.yml",
    help="Project configuration file. Default: ./ml2p.yml.",
)
@click.version_option(version=ml2p_version)
@click.pass_context
def ml2p(ctx, cfg):
    """Minimal Lovable Machine Learning Pipeline.

    A friendlier interface to AWS SageMaker.
    """
    ctx.obj = ModellingProjectWithSagemakerClient(cfg=cfg)


ml2p.add_command(init)
ml2p.add_command(dataset)
ml2p.add_command(training_job)
ml2p.add_command(model)
ml2p.add_command(endpoint)
ml2p.add_command(notebook)
ml2p.add_command(repo)
