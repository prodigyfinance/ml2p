# -*- coding: utf-8 -*-

""" ML2P SageMaker training and prediction server for use in SageMaker docker
    containers.
"""

import atexit
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


@click.group()
@click.option(
    "--ml-folder",
    default="/opt/ml/",
    help="The base folder for the datasets and models.",
)
@click.pass_context
def ml2p_docker(ctx, ml_folder):
    """ ML2P Sagemaker Docker container helper CLI. """
    ctx.ensure_object(dict)
    ctx.obj["env"] = SageMakerEnv(ml_folder)


@ml2p_docker.command("train")
@click.argument("model_trainer")
@pass_sagemaker_env
def train(env, model_trainer):
    """ Train the model.
    """
    click.echo("Training model version {}.".format(env.model_version))
    try:
        trainer_cls = import_string(model_trainer)
        trainer = trainer_cls(env)
        trainer.train()
    except Exception:
        env.write_failure(traceback.format_exc())
        raise
    click.echo("Done.")


@ml2p_docker.command("serve")
@click.argument("model_predictor")
@click.option("--debug/--no-debug", default=False)
@pass_sagemaker_env
def serve(env, model_predictor, debug):
    """ Serve the model.
    """
    click.echo("Starting server for model version {}.".format(env.model_version))
    predictor_cls = import_string(model_predictor)
    predictor = predictor_cls(env)
    predictor.setup()
    app.predictor = predictor
    atexit.register(predictor.teardown)
    app.run(host="0.0.0.0", port=8080, debug=debug)
    click.echo("Done.")


if __name__ == "__main__":
    ml2p_docker()
