History
=======

0.3.2 (2023-02-16)
------------------

* Fix dataset name.

0.3.1 (2023-02-16)
------------------

* Disable network isolation for generating datasets.
* Change order in which notebooks resources are deleted.

0.3.0 (2023-02-16)
------------------

* Replace Sagefaker by moto sagemaker client to mock sagemaker.
* Refactor CLI commands.
* Add ml2p dataset generate command to generate datasets using the CLI.
* Add ModelDatasetGenerator class.
* Simplify how environment variables are passed to training job.
* Add ml2p-docker generate-dataset command.
* Update documentation.

0.2.4 (2022-02-21)
------------------

* Unpin all dependencies to stay up to date with Pallets packages.

0.2.3 (2021-07-12)
------------------

* Added the on-create feature to notebook servers.

0.2.2 (2021-05-18)
------------------

* Pinned Flask to a stable version.

0.2.1 (2021-05-13)
------------------

* Pinned Flask, Flask-API and click.

0.2.0 (2020-09-16)
------------------

* Added reference documentation.
* Added a tutorial.
* Added tests for the ML2P command line utilities.
* Added support for attaching training and deployment instances to VPCs.
* Open sourced ML2P under the ISCL.

0.1.5 (2020-06-12)
------------------

* Added `ml2p dataset delete` which deletes an entire dataset.
* Added `ml2p dataset ls` which lists the contents of a dataset.
* Added `ml2p dataset up` which uploads a local file to a dataset.
* Added `ml2p dataset dn` which downloads a file from a dataset.
* Added `ml2p dataset rm` which deletes a file from a dataset.

0.1.4 (2020-02-21)
------------------

* Correctly handle folder keys when downloading datasets from S3. Previously folder
  keys created files, now they created folders.

0.1.3 (2020-02-20)
------------------

* Added support for local environments. These allow ML2P models to be trained and used
  to make predictions locally, as though they were being loaded in SageMaker.
* Added support for downloading datasets and models from S3 into local environments.

0.1.2 (2020-01-23)
------------------

* Fix support for recording predictions in S3 (in first release of this feature, the code
  attempted to pass a boolean value as an environment variable, which failed as expected).

0.1.1 (2020-01-22)
------------------

* Add support for recording predictions in S3.

0.1.0 (2019-10-22)
------------------

* Improve batch prediction support to allow models to separately implement batch
  prediction (e.g. a model might want to implement batch prediction separately to
  improve performance).
* Tweak training job version format to only include major and minor versions numbers.
  Patch version numbers are now reserved for models and intended for use in the case
  where the code used to make predictions changes but the underlying model is the same.
* Model creation now defaults to using the training job with the same version as the model
  but with the patch number removed.
* Endpoint creation now defaults to using the model with the same version as the endpoint.
* When creating training jobs or models, specifying the model type is now required if
  the ml2p configuration file contains more than one model. If there is exactly one model
  type listed, that is the default. If there are no model types, the docker file
  must specify the model on the command line.
* Metadata returned by predictions now includes the ML2P version number.
* Version bumped to 0.1.0 now that versioning support is complete(-ish).

0.0.9 (2019-10-15)
------------------

* Add support for client and server error exception handling.
* Deprecate passing a channel name to dataset_folder and add a new data_channel_folder
  method to allow data in other channels to be accessed.
* Add dataset create and list commands to ml2p CLI.
* Add --version to ml2p and ml2p-docker CLIs.
* Allow model and endpoint version numbers to be multiple digits.

0.0.8 (2019-09-11)
------------------

* Added validation of naming convention

0.0.7 (2019-08-29)
------------------

* Added Sphinx requirements to build file.

0.0.6 (2019-08-29)
------------------

* Cleaned up support for passing ML2P environment data into training jobs and
  model deployments. Environment settings such as the S3 URL and the project name
  are now passed into training jobs via hyperparameters and into model deployments
  via model environment variables.
* Added support for training and serving multiple models using the same docker
  image by optionally passing the model to use into training jobs and endpoint
  deployments.
* Added support for rich hyperparameters. This sidesteps SageMaker API's limited
  hyperparameter support (it only supports string values) by encoding any
  JSON-compatible Python dictionary to a flattened formed and then decoding
  it when it is read by the training job.
* Added skeleton for Sphinx documentation.
* Removed old pre-0.0.1 example files.

0.0.5 (2019-07-23)
------------------

* Disabled direct internet access from notebooks by default.
* Added tests for cli_utils.

0.0.4 (2019-06-26)
------------------

* Fixed bug in setting of ML2P_S3_URL on model creation.

0.0.3 (2019-06-26)
------------------

* Added new ml2p notebook command group for creating, inspecting,
  and deleting SageMaker Notebook instances.
* Added new ml2p repo command group for inspecting code repository SageMaker resources.

0.0.2 (2019-05-24)
------------------

* Complete re-write.
* Added new ml2p-docker command added that assists with training and deploying models
  in SageMaker.


0.0.1 (2018-10-19)
------------------

* Initial hackathon release.
