.. _ml2p-tutorial:

Tutorial
========

Welcome to ML2P! In this tutorial we'll take you through:

* setting up your first project
* uploading a training dataset to S3
* training a model
* deploying a model
* making predictions

Throughout all of this we'll be working with the classic Boston house prices
dataset that is available within `scikit-learn <https://scikit-learn.org/>`_.


Setting up your project
-----------------------

Before running ML2P you'll need to create:

  * a docker image,
  * a S3 bucket,
  * and an AWS role

yourself. ML2P does not manage these for you.

Once you have the docker image, bucket and role set up, you are ready to
create your ML2P configuration file. Save the following as `ml2p.yml`:

.. literalinclude:: ml2p.yml
   :language: yaml
   :emphasize-lines: 2,6-7
   :linenos:

The yellow high-lights show the lines where you'll need to fill in the details
for the docker image, S3 bucket and AWS role.


Initialize the ML2P project
---------------------------

You're now ready to run your first ML2P command!

First set the AWS profile you'd like to use:

.. code-block:: console

   $ export AWS_PROFILE="my-profile"

You'll need to set the `AWS_PROFILE` or otherwise provide your AWS credentials
whenever you run an `ml2p` command.

If you haven't initialized your project before, run:

.. code-block:: console

   $ ml2p init

which will create the S3 model and dataset folders for you.

Once you've run `ml2p init`,  ML2P have will created the following folder
structure in your S3 bucket::

  s3://your-s3-bucket/
    models/
      ... ml2p will place the outputs of your training jobs here ...
    datasets/
      ... ml2p will place your datasets here, the name of the
          subfolder is the name of the dataset ...


Creating a training dataset
---------------------------

First create a CSV file containing the Boston house prices that you'll be
using to train your model. You can do this by saving the file below as
`create_boston_prices_csv.py`:

.. literalinclude:: create_boston_prices_csv.py
    :language: python
    :linenos:

and running:

.. code-block:: console

   $ python create_boston_prices_csv.py

This will write the file `house-prices.csv` to the current folder.

Now create a training dataset and upload the CSV file to it:

.. code-block:: console

   $ ml2p dataset create boston-20200901
   $ ml2p dataset up boston-20200901 house-prices.csv

And check that the contents of the training dataset is as expected by
listing the files in it:

.. code-block:: console

   $ ml2p dataset ls boston-20200901


Training a model
----------------

You'll need to start by implementing a subclass of `ml2p.core.ModelTrainer`.
Your subclass needs to define a `.train(...)` method that will load the
training set, train the model, and save it.

A simple implementation for the Boston house price model can be found in
`model.py`:

.. literalinclude:: model.py
   :language: python
   :emphasize-lines: 14-27
   :linenos:

The training data should be read from `self.env.dataset_folder()`. This is the
folder that SageMaker will load your training dataset into.

Once the model is trained, you should write your output files to
`self.env.model_folder()`. SageMaker will read the contents of this folder once
training has finished and store them as a `.tar.gz` file in S3.

Before you train your model in SageMaker, you can try it locally
as shown in `local.py`:

.. literalinclude:: local.py
   :language: python
   :emphasize-lines: 11-14
   :linenos:

ML2P provides `ml2p.core.LocalEnv` which you can use to emulate a real SageMaker
environment. SageMaker will read the training data from `input/data/training/`
so you will need to place a copy of `house-prices.csv` there for the
script to run successfully.

Later in the tutorial you will learn how to download a dataset directly from
S3 for use in a local environment.

Once your model works locally, you are ready to train it in SageMaker by
creating a training job with:

.. code-block:: console

  $ ml2p training-job create boston-train boston-20200901 --model-type boston

The first argument is the name of the training job, the second is name of the
dataset. You will need to have uploaded some training data. The `--model-type`
argument is optional -- if you have only a single model defined in `ml2p.yml`,
ML2P will automatically select that one for you.

Wait for your training job to finish. To check up on it you can run:

.. code-block:: console

  $ ml2p training-job wait boston-train  # wait for job to finish
  $ ml2p training-job describe boston-train  # inspect job

Once your training job is done, there is one more step. The training job
records the trained model parameters, but we also need to specify the Docker
image that should be used along with those parameters. We do this by creating a
SageMaker model from the output of the training job:

.. code-block:: console

  $ ml2p model create boston-model boston-train --model-type boston

The first argument is the name of the model to create, the second is the training job
the model should be created from.

The docker image to use is read from the `image` parameter in `ml2p.yml` so
you don't have to specify it here.

The model is just an object in SageMaker -- it doesn't run any instances -- so it will be
created immediately.

Now its time to deploy your model by creating an endpoint for it!


Deploying a model
-----------------

To deploy a model you'll need to implement a subclass of `ml2p.core.ModelPredictor`.

You might have seen the implementation for the Boston house price model in
`model.py` while looking at the code for training, but here it is again:

.. literalinclude:: model.py
   :language: python
   :emphasize-lines: 30-51
   :linenos:

The `.setup()` method is called only once when starting up a prediction instance.
It should read the model from `self.env.model_folder()` -- SageMaker will have
placed them in the same location where they were stored while running `.train()`.
Other kinds of setup can be done in this function too if you need to.

The `.result(data)` method is called when a prediction needs to be made. It will
be passed the data that was sent to the prediction API endpoint (usually a
dictionary with the features as the keys) and should return the prediction.

As you can see in `local.py`, `.result()` is usually not called directly. Instead,
when a prediction needs to be made, ML2P will call `.invoke()`, which will
then call `.result()` and add some metadata to the result before returning it.

If you ran `local.py` earlier, you've already successfully run a local prediction.

Once you're ready to deploy your model, you can create an endpoint by running:

.. code-block:: console

  $ ml2p endpoint create boston-endpoint --model-name boston-model

The first argument is the name of the endpoint to create, the second is the name of
the model to create the endpoint from.

Note that endpoints can be quite expensive to run, so check the pricing for the
instance type you have specified before pressing enter!

Setting up the endpoint takes awhile. To check up on it you can run:

.. code-block:: console

  $ ml2p endpoint wait boston-endpoint  # wait for endpoint to be ready
  $ ml2p endpoint describe boston-endpoint  # inspect endpoint

Once the endpoint is ready, your model is deployed!

You can make a test prediction using:

.. code-block:: console

  $ ml2p endpoint invoke boston-endpoint '{"CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0.0, "NOX": 0.5379999999999999, "RM": 6.575, "AGE": 65.2, "DIS": 4.09, "RAD": 1.0, "TAX": 296.0, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98, "target": 24.0}'

Congratulations! You have trained and deployed your first model using ML2P!


Working with models locally
---------------------------

At times it may be convenient to work with ML2P models on a local machine, rather than
within SageMaker. ML2P supports both training models locally and loading models trained
in SageMaker for local analysis.

In either case, first create a local environment:

.. code-block:: python
  :linenos:

  # set up a connection to AWS, specifying an appropriate AWS profile name:
  import boto3
  session = boto3.session.Session(profile_name="aws-profile")

  # create a local environment
  from ml2p.core import LocalEnv
  env = LocalEnv(".", "./ml2p.yml", session)

  # import your ml2p model class:
  from model import BostonModel

The first argument to `LocalEnv` is the local folder to store the environment
in, and the second is the path to the `ml2p.yml` config file for the project.

The third argument, `session`, is an optional boto3 session and is only needed
if you wish to download datasets or models from S3 to your local environment.

To download a dataset from S3 into the local environment used:

.. code-block:: python

  env.download_dataset("dataset-name")

If you prefer not to download a dataset, you can also copy a local file into::

  input/data/training/

For example, for this tutorial it may be useful to copy the `house-prices.csv`
training file into this folder using:

.. code-block:: console

  $ mkdir -p input/data/training/
  $ cp house-prices.csv input/data/training/

Once you have a dataset you can train a model locally using:

.. code-block:: python

  env.clean_model_folder()
  trainer = BostonModel().trainer(env)
  trainer.train()

The first line, `env.clean_model_folder()` just deletes any old files created
by previous local training runs.

You can list the model files created during training using:

.. code-block:: console

  $ ls model/

If you have already trained a model in SageMaker with `ml2p create training-job`
and would like to examine it locally you can download it into the model folder
by running:

.. code-block:: python

  env.download_model("training-job-name")

Once you have a model available locally, either by training it locally or by
downloading it, you can make predictions with:

.. code-block:: python

  predictor = BostonModel().predictor(env)
  predictor.setup()
  predictor.invoke(data)

Happy local analyzing and debugging!
