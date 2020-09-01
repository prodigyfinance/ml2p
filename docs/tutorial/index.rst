.. _ml2p-tutorial:

Tutorial
========

Welcome to ML2P! In this tutorial we'll take you through:

* setting up your first project
* uploading a training dataset to S3
* training a model
* deploying a model
* making predictions

Through out all of this we'll be working with the classic Boston house prices
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


Training and deploying a model
------------------------------

The code for the model:

.. literalinclude:: model.py
   :language: python
   :linenos:

Next start a training job to train your model:

.. code-block:: console

  $ ml2p training-job create boston-train boston-20200901 --model-type boston

The first argument is the name of the training job, the second is name of the dataset
(i.e. the folder under ``/datasets/`` in your project's S3 bucket). You will need
to have uploaded some training data. The `--model-type` argument is optional -- the
model type to use may also be specified directly in the docker image.

Wait for your training job to finish. To check up on it you can run:

.. code-block:: console

  $ ml2p training-job wait boston-train  # wait for job to finish
  $ ml2p training-job describe boston-train  # inspect job

Once your training job is done, create a model from the output of the training job:

.. code-block:: console

  $ ml2p model create boston-model boston-train --model-type boston

The first argument is the name of the model to create, the second is the training job
the model should be created from.  The `--model-type` argument is optional -- the
model type to use may also be specified directly in the docker image.

The model is just an object in SageMaker -- it doesn't run any instances -- so it will be
created immediately.

Now its time to deploy your model by creating an endpoint for it:

.. code-block:: console

  $ ml2p endpoint create boston-endpoint --model-name boston-model

The first argument is the name of the endpoint to create, the second is the name of
the model to create the endpoint from.

Setting up the endpoint takes awhile. To check up on it you can run:

.. code-block:: console

  $ ml2p endpoint wait boston-endpoint  # wait for endpoint to be ready
  $ ml2p endpoint describe boston-endpoint  # inspect endpoint

Once the endpoint is ready, your model is deployed!

You can make a test prediction using:

.. code-block:: console

  $ ml2p endpoint invoke boston-endpoint '{"your": "data"}'

And you're done!


Working with models locally
---------------------------

At times it may be convenient to work with ML2P models on a local machine, rather than
within SageMaker. ML2P supports both training models locally and loading models trained
in SageMaker for local analysis.

In either case, first create a local environment::

  # set up a connection to AWS, specifying an appropriate AWS profile name:
  import boto3
  session = boto3.session.Session(profile_name="aws-profile")

  # create a local environment, the arguments are the local folder to store the
  # environment in, the path the ml2p.yml config file, and an optional boto3
  # session to use for retrieving files from S3.
  from ml2p.core import LocalEnv
  env = LocalEnv("./local", "./sagemaker/ml2p.yml", session)

  # import your ml2p model class:
  from my_package import MyModel

Then to train a model locally::

  env.download_dataset("dataset-name")
  env.clean_model_folder()
  trainer = MyModel().trainer(env)
  trainer.train()

And to load an already trained model::

  env.download_model("training-job-name")
  predictor = MyModel().predictor(env)
  predictor.setup()

Happy local analyzing and debugging!
