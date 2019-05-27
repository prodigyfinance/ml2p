# -*- coding: utf-8 -*-

""" ML2P SageMaker training and prediction server for use in SageMaker docker
    containers.
"""

import atexit
import collections
import traceback
from functools import update_wrapper

import click
from flask import request

from flask_api import FlaskAPI

from .core import SageMakerEnv, import_string

app = FlaskAPI(__name__)


@app.route("/invocations", methods=["POST"])
def invocations():
    response = app.predictor.invoke(request.data)
    return response


@app.route("/ping", methods=["GET"])
def ping():
    return {"model_version": app.predictor.env.model_version}


def pass_sagemaker_env(f):
    """ Pass the current SageMakerEnv into a click command. """

    @click.pass_context
    def new_func(ctx, *args, **kwargs):
        return ctx.invoke(f, ctx.obj["env"], *args, **kwargs)

    return update_wrapper(new_func, f)


ML2POptions = collections.namedtuple("ML2POptions", ["model"])


def pass_ml2p_docker_options(f):
    """ Pass the group options into a click command. """

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
@click.pass_context
def ml2p_docker(ctx, ml_folder, model):
    """ ML2P Sagemaker Docker container helper CLI. """
    ctx.ensure_object(dict)
    ctx.obj["env"] = SageMakerEnv(ml_folder)
    if model is not None:
        model = import_string(model)
    ctx.obj["opt"] = ML2POptions(model=model)


@ml2p_docker.command("train")
@pass_sagemaker_env
@pass_ml2p_docker_options
def train(opt, env):
    """ Train the model.
    """
    if opt.model is None:
        raise click.UsageError(
            "The global parameter --model must be given when calling the train command."
        )
    click.echo("Training model version {}.".format(env.model_version))
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
    """ Serve the model and make predictions.
    """
    if opt.model is None:
        raise click.UsageError(
            "The global parameter --model must be given when calling the serve command."
        )
    click.echo("Starting server for model version {}.".format(env.model_version))
    predictor = opt.model().predictor(env)
    predictor.setup()
    app.predictor = predictor
    atexit.register(predictor.teardown)
    app.run(host="0.0.0.0", port=8080, debug=debug)
    click.echo("Done.")
