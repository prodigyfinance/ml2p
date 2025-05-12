# -*- coding: utf-8 -*-
"""ML2P SageMaker training and prediction server for use in SageMaker docker
containers, without Flask-API."""

import atexit
import collections
import traceback
from functools import update_wrapper

import click
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from . import __version__ as ml2p_version
from .core import SageMakerEnv, import_string
from .errors import APIError


class ML2PAPI(Flask):
    """Improved error handling for ML2P API using Flask and Werkzeug."""

    def handle_user_exception(self, exc):
        # Handle our APIError with JSON response
        if isinstance(exc, APIError):
            return jsonify(message=exc.message, details=exc.details), exc.status_code
        # Handle generic HTTP exceptions (including those from werkzeug)
        if isinstance(exc, HTTPException):
            # Use description for message
            return jsonify(message=exc.description), exc.code
        # Default Flask behavior for other exceptions
        return super().handle_user_exception(exc)


app = ML2PAPI(__name__)


@app.route("/invocations", methods=["POST"])
def invocations():
    if "instances" in request.get_json(force=True):
        response = app.predictor.batch_invoke(request.get_json(force=True)["instances"])
    else:
        response = app.predictor.invoke(request.get_json(force=True))
    return response


@app.route("/execution-parameters", methods=["GET"])
def execution_parameters():
    return {
        "MaxConcurrentTransforms": 1,
        "BatchStrategy": "MULTI_RECORD",
        "MaxPayloadInMB": 6,
    }


@app.route("/ping", methods=["GET"])
def ping():
    return {
        "model_version": app.predictor.env.model_version,
        "ml2p_version": ml2p_version,
    }


def pass_sagemaker_env(f):
    """Pass the current SageMakerEnv into a click command."""

    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj["env"], *args, **kwargs)

    return update_wrapper(new_func, f)


ML2POptions = collections.namedtuple("ML2POptions", ["model"])


def pass_ml2p_docker_options(f):
    """Pass the group options into a click command."""

    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj["opt"], *args, **kwargs)

    return update_wrapper(new_func, f)


@click.group()
@click.option(
    "--ml-folder",
    default="/opt/ml/",
    help="The base folder for the datasets and models.",
)
@click.option(
    "--model",
    default=None,
    help="The fully qualified name of the ML2P model interface to use.",
)
@click.version_option(version=ml2p_version)
@click.pass_context
def ml2p_docker(ctx, ml_folder, model):
    """ML2P Sagemaker Docker container helper CLI."""
    ctx.ensure_object(dict)
    ctx.obj["env"] = env = SageMakerEnv(ml_folder)
    if model is None:
        model = env.model_cls
    if model is not None:
        model = import_string(model)
    ctx.obj["opt"] = ML2POptions(model=model)


@ml2p_docker.command("train")
@pass_sagemaker_env
@pass_ml2p_docker_options
def train(opt, env):
    """Train the model."""
    if opt.model is None:
        raise click.UsageError(
            "The global parameter --model must either be given when calling the train"
            " command or --model-type must be given when creating the training job."
        )
    click.echo(f"Starting training job {env.training_job_name}.")
    try:
        trainer = opt.model().trainer(env)
        trainer.train()
    except Exception:
        env.write_failure(traceback.format_exc())
        raise
    click.echo("Done.")


@ml2p_docker.command("serve")
@click.option("--debug/--no-debug", default=False)
@pass_sagemaker_env
@pass_ml2p_docker_options
def serve(opt, env, debug):
    """Serve the model and make predictions."""
    if opt.model is None:
        raise click.UsageError(
            "The global parameter --model must either be given when calling the serve"
            " command or --model-type must be given when creating the model."
        )
    click.echo(f"Starting server for model version {env.model_version}.")
    predictor = opt.model().predictor(env)
    predictor.setup()
    app.predictor = predictor
    atexit.register(predictor.teardown)
    app.run(host="0.0.0.0", port=8080, debug=debug)
    click.echo("Done.")


@ml2p_docker.command("generate-dataset")
@pass_sagemaker_env
@pass_ml2p_docker_options
def generate_dataset(opt, env):
    """Generates a dataset for training the model."""
    if opt.model is None:
        raise click.UsageError(
            "The global parameter --model must either be given when calling the"
            " generate-dataset command or --model-type must be given when"
            " generating the dataset."
        )
    click.echo(f"Starting generation of dataset {env.dataset_name}.")
    try:
        dataset_generator = opt.model().dataset_generator(env)
        dataset_generator.generate()
    except Exception:
        env.write_failure(traceback.format_exc())
        raise
    click.echo("Done.")
