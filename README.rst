====
ML2P
====

ML2P -- or (ML)^2P -- is the minimal lovable machine-learning pipeline and a friendlier
interface to AWS SageMaker.

Install ML2P with::

  $ pip install ml2p


Overview
========

ML2P helps manage a machine learning project. You'll define your project
by writing a small YAML file named `ml2p.yml`::

  project: "zendesk-complaint-flagger"
  s3folder: "s3://zendesk-complaint-flagger-sagemaker-test/"
  defaults:
    image: "670223297817.dkr.ecr.eu-west-1.amazonaws.com/zendesk-complaint-flagger-sagemaker:latest"
    role: "arn:aws:iam::319756771752:role/service-role/AmazonSageMaker-ExecutionRole-20190410T162807"
  train:
    instance_type: "ml.m5.xlarge"
  deploy:
    instance_type: "ml.t2.medium"

This specifies:

  * the name of your project,
  * the docker image that your project will use for training and prediction,
  * the S3 bucket that will hold the models and data sets for your project,
  * and the AWS role your project will run under.

The name of your project functions as a prefix to the names of SageMaker training jobs,
models and endpoints that ML2P creates (since these names are global within a SageMaker
account).

ML2P also tags all of the AWS objects it creates with your project name.


Setting up your project
=======================

To set up your project you'll need to create:

  * the docker image,
  * the S3 bucket,
  * and the AWS role

yourself. ML2P does not manage these for you.

When you run `ml2p.py init` (see below),  ML2p will create the following folder
structure in your S3 bucket::

  s3://my-project-bucket/
    models/
      ... ml2p with place the outputs of your training jobs here ...
    datasets/
      ... you should place your data sets here, the name of the
          subfolder is the name of the data set ...


Assumptions made by ML2P
========================

ML2P is minimal, opinionated and probably wrong in some cases. It currently assumes:

* You want your datasets and models stored in the folder structure described above.

* You've uploaded your training data to folders under `datasets/`.


Training and deploying a model
==============================

First set the AWS profile you'd like to use::

  $ export AWS_PROFILE="prodigy-development"

If you haven't initialized your project before, run::

  $ python ml2p.py init

which will create the S3 model and dataset folder for you.

Next start a training job to train your model::

  $ python ml2p.py training-job create zcf-train-6 zcf-20190412

The first argument is the name of the training job, the second is name of the data
set (i.e. the folder under ``/datasets/`` in your project's S3 bucket). You will need
to have uploaded some training data.

Wait for your training job to finish. To check up on it you can run::

  $ python ml2p.py training-job wait zcf-train-6  # wait for job to finish
  $ python ml2p.py training-job describe zcf-train-6  # inspect job

Once your training job is done, create a model from the output of the training job::

  $ python ml2p.py model create zcf-model-6 zcf-train-6

The first argument is the name of the model to create, the second is the training job
the model should be created from.

The model is just an object in S3 -- it doesn't run any instances -- so it will be
created immediately.

Now its time to deploy your model by creating an endpoint for it::

  $ python ml2p.py endpoint create zcf-endpoint-6 zcf-model-6

The first argument is the name of the endpoint to create, the second is the name of
the model to create the endpoint form.

Setting up the endpoint takes awhile. To check up on it you can run::

  $ python ml2p.py endpoint wait zcf-endpoint-6  # wait for endpoint to be ready
  $ python ml2p.py endpoint describe zcf-endpoint-6  # inspect endpoint

Once the endpoint is ready, your model is deployed!

You can make a test prediction using::

  $ python ml2p.py endpoint invoke zcf-endpoint-6 '{"your": "data"}'

And you're done!
