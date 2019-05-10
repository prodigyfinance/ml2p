# -*- coding: utf-8 -*-

""" ML2P SageMaker training and prediction server for use in SageMaker docker
    containers.
"""

import json
import os
import pathlib

import click
from flask_api import FlaskAPI


app = FlaskAPI(__name__)


@app.route("/invocations", methods=["POST"])
def invocations():
    # TODO: implement generic invocations
    response = {
        "model_version": "XXX",
    }
    return response


@app.route("/ping", methods=["GET"])
def ping():
    return {}


@click.group()
def ml2p_docker():
    """ ML2P Sagemaker Docker container helper CLI. """


@ml2p_docker.command("train")
@click.option(
    "--ml-folder",
    default="/opt/ml/",
    help="The base folder to read datasets from and write the model to.",
)
def train(ml_folder):
    """ Train the model.
    """
    click.echo("Training ...")

    # TODO: implement generic training

    click.echo("Done.")


@ml2p_docker.command("serve")
@click.option(
    "--ml-folder", default="/opt/ml", help="The base folder to read the model from."
)
def serve(ml_folder):
    """ Serve the model.
    """
    model_version = os.environ.get("ML2P_MODEL_VERSION", "Unknown")
    click.echo("Starting model version {}.".format(model_version))

    # TODO: implement generic app setup

    app.run(host="0.0.0.0", port=8080, debug=True)  # TODO: Turn off debug!


if __name__ == "__main__":
    ml2p_docker()
