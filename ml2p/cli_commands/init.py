# -*- coding: utf-8 -*-

""" CLI command for initializing ML2P projects. """

import boto3
import click


@click.command("init")
@click.pass_obj
def init(prj):
    """Initialize the project S3 bucket."""
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
