====
ML2P
====

ML2P -- or (ML)^2P -- is the minimal lovable machine-learning pipeline and a
friendlier interface to `AWS SageMaker <https://aws.amazon.com/sagemaker/>`_.

Design goals:

* support the full machine learning lifecyle
* support custom feature engineering
* support building custom models in Python
* provide reproducible training and deployment of models
* support the use of customised base Docker images for training and deployment

Concretely it provides a command line interface and a Python library to assist
with:

* S3:
    * Managing training data
* SageMaker:
    * Launching training jobs
    * Deploying trained models
    * Creating notebook instances
* On your local machine or in a SageMaker notebook:
    * Downloading training datasets from S3
    * Training models
    * Loading trained models from SageMaker / S3


Installing
==========

Install ML2P with::

  $ pip install ml2p


Mailing list
============

If you have questions about ML2P, or would like to contribute or have
suggestions for improvements, you are welcome to join the project mailing
list at https://groups.google.com/g/ml2p and write us a letter there.


Overview
========

ML2P helps manage a machine learning project. You'll define your project
by writing a small YAML file named `ml2p.yml`::

  project: "ml2p-tutorial"
  s3folder: "s3://your-s3-bucket/"
  models:
    bob: "models.RegressorModel"
  defaults:
    image: "XXXXX.dkr.ecr.REGION.amazonaws.com/your-docker-image:X.Y.Z"
    role: "arn:aws:iam::XXXXX:role/your-role"
  train:
    instance_type: "ml.m5.large"
  deploy:
    instance_type: "ml.t2.medium"
    record_invokes: true

This specifies:

* `project`: the name of your project
* `s3folder`: the S3 bucket that will hold the models and data sets for your
  project
* `models`: a list of model names and the Python classes that will be used to
  train the models and make predictions
* `defaults`:

  * `image`: the docker image that your project will use for training and
    prediction
  * `role`: the AWS role your project will run under

* `train`:

  * `instance_type`: the AWS instance type that will be used when training
    your model

* `deploy`:

  * `instance_type`: the AWS instance type that will be used when deploying
    your model
  * `record_invokes`: whether to record prediction requests in S3

The name of your project functions as a prefix to the names of SageMaker training jobs,
models and endpoints that ML2P creates (since these names are global within a SageMaker
account).

ML2P also tags all of the AWS objects it creates with your project name.


Tutorial
========

See `<https://ml2p.readthedocs.io/en/latest/tutorial/>`_ for a step-by-step tutorial.
