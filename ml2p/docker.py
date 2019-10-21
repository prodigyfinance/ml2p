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

from . import __version__ as ml2p_version
from .core import SageMakerEnv, import_string
from .errors import APIError


class ML2PAPI(FlaskAPI):
    """ Improved error handling for ML2P API. """

    def __init__(self, *args, **kw):
        super(ML2PAPI, self).__init__(*args, **kw)
        # FlaskAPI requires error_handler_spec to have a key for None (i.e. undefined
        # error) but does not set it itself, so we do so here:
        self.error_handler_spec.setdefault(None, {})

    def handle_api_exception(self, exc):
        if not isinstance(exc, APIError):
            return super(ML2PAPI, self).handle_api_exception(exc)
        # Enhanced error support for errors raised by the ML2P API:
        content = {"message": exc.message, "details": exc.details}
        return self.response_class(content, status=exc.status_code)


app = ML2PAPI(__name__)


@app.route("/invocations", methods=["POST"])
def invocations():
    if "instances" in request.data:
        response = app.predictor.batch_invoke(request.data["instances"])
    else:
        response = app.predictor.invoke(request.data)
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
@click.version_option(version=ml2p_version)
@click.pass_context
def ml2p_docker(ctx, ml_folder, model):
    """ ML2P Sagemaker Docker container helper CLI. """
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
    """ Train the model.
    """
    if opt.model is None:
        raise click.UsageError(
            "The global parameter --model must either be given when calling the train"
            " command or --model-type must be given when creating the training job."
        )
    click.echo("Starting training job {}.".format(env.training_job_name))
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
            "The global parameter --model must either be given when calling the serve"
            " command or --model-type must be given when creating the model."
        )
    click.echo("Starting server for model version {}.".format(env.model_version))
    predictor = opt.model().predictor(env)
    predictor.setup()
    app.predictor = predictor
    atexit.register(predictor.teardown)
    app.run(host="0.0.0.0", port=8080, debug=debug)
    click.echo("Done.")
