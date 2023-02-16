# -*- coding: utf-8 -*-

""" CLI commands for interating with S3 datasets. """

import pathlib

import boto3
import click

from .utils import (
    click_echo_json,
    mk_processing_job,
    validate_model_type,
    validate_name,
)


@click.group("dataset")
def dataset():
    """Create and manage datasets."""


@dataset.command("list")
@click.pass_obj
def dataset_list(prj):
    """List datasets for this project."""
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
@click.pass_obj
def dataset_create(prj, dataset):
    """Create a dataset."""
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
@click.pass_obj
def dataset_delete(prj, dataset):
    """Delete a dataset."""
    validate_name(dataset, "dataset")
    client = boto3.client("s3")
    prefix = prj.s3.path("/datasets/{}/".format(dataset))
    response = client.list_objects_v2(Bucket=prj.s3.bucket(), Prefix=prefix)
    objects_to_delete = [{"Key": item["Key"]} for item in response["Contents"]]
    client.delete_objects(
        Bucket=prj.s3.bucket(),
        Delete={"Objects": objects_to_delete},
    )


@dataset.command("ls")
@click.argument("dataset")
@click.pass_obj
def dataset_ls(prj, dataset):
    """List the contents of a dataset."""
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
@click.pass_obj
def dataset_up(prj, dataset, src, dst):
    """Upload a file SRC to a dataset as DST.

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
@click.pass_obj
def dataset_dn(prj, dataset, src, dst):
    """Download a file SRC from the dataset and save it in DST.

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
@click.pass_obj
def dataset_rm(prj, dataset, filename):
    """Delete a file from a dataset."""
    validate_name(dataset, "dataset")
    client = boto3.client("s3")
    client.delete_object(
        Bucket=prj.s3.bucket(),
        Key=prj.s3.path("/datasets/{}/{}".format(dataset, filename)),
    )


@dataset.command("generate")
@click.argument("dataset")
@click.option(
    "--model-type",
    "-m",
    default=None,
    callback=validate_model_type,
    help="The name of the type of model.",
)
@click.pass_obj
def dataset_generate(prj, dataset, model_type):
    """Launch a processing job that generates a dataset."""
    validate_name(dataset, "dataset")
    processing_job_params = mk_processing_job(prj, dataset, model_type)
    response = prj.client.create_processing_job(**processing_job_params)
    click_echo_json(response)
