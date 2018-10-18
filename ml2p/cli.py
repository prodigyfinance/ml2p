# -*- coding: utf-8 -*-

""" CLI for ml2p. """

import click


@click.group()
def cli():
    """ A minimum-lovable machine-learning pipeline on top of Amazon
        SageMaker.
    """


@cli.command()
def dataset():
    """ Create and list datasets. """


@cli.command()
def train():
    """ Train and list trained models. """


@cli.command()
def metrics():
    """ Examine metrics for trained models. """


@cli.command()
def docker_build():
    """ Build a docker image for the model. """


@cli.command()
def deploy():
    """ Deploy a trained model. """
