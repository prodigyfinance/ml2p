# -*- coding: utf-8 -*-

""" ML2P SageMaker context objects.

    These are convenience methods for accessing the SageMaker Docker
    container environment.
"""

import functools
import json
import os
import pathlib

from . import errors as ml2p_errors

# SageMaker data channel modes:

CHANNEL_MODE_FILE = "File"
CHANNEL_MODE_PIPE = "Pipe"


class SageMakerTrainContext:
    """ Provides access to a SageMaker Docker training environment,
        including:

        * Hyperparameters
        * Training job name
        * Input Data Configuration
        * Training Data
        * Distributed Training Configuration

        Not yet implemented:

        * Pipe data channels.
    """

    def job_name(self):
        """ Return the training job name. """
        return os.environ["TRAINING_JOB_NAME"]

    @functools.lru_cache()
    def hyperparameters(self):
        """ Return the hyperparameters. """
        # XXX: Should this return a better object than a dictionary?
        with open("/opt/ml/input/config/hyperparameters.json") as f:
            return json.load(f)

    @functools.lru_cache()
    def input_data_config(self):
        """ Return the input data configuration. """
        # XXX: Do we need this method to be exposed?
        with open("/opt/ml/input/config/inputdataconfig.json") as f:
            return json.load(f)

    @functools.lru_cache()
    def resource_config(self):
        """ Return the distributed training configuration. """
        # XXX: Should this return a better object than a dictionary?
        with open("/opt/ml/input/config/resourceconfig.json") as f:
            return json.load(f)

    def data_channel(self, data_channel):
        """ Return a path to the data channel folder.

            Checks that the data chanell exists and has input mode set to FILE
            (i.e. is not a pipelined channel).

            :param str data_channel:
                The name of the data channel, e.g. "training".
            :return pathlib.Path:
                The path of the data channel.
        """
        channel = self.input_data_config().get(data_channel)
        if channel is None:
            raise ml2p_errors.ML2PDataChannelMissing(
                "Data channel {} is not configured.".format(data_channel))
        channel_mode = channel.get("TrainingInputMode")
        if channel_mode != CHANNEL_MODE_FILE:
            raise ml2p_errors.ML2PDataChannelIncorrectedMode(
                "Data channel {} has mode {} but {} is required.".format(
                    data_channel, channel_mode, CHANNEL_MODE_FILE))
        return pathlib.Path("/opt/ml/input/data") / data_channel


class SageMakerPredictContext:
    """ Provides access to a SageMaker Docker predict environment,
        including:

        * Model folder
    """

    def model_folder(self):
        """ Return the model folder.

            :return pathlib.Path:
                The path of the model folder.
        """
        return pathlib.Path("/opt/ml/model")
