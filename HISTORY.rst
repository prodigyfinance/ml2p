History
=======

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

* Complete re-write. Initially implemented for zendesk_complaint_flagger.
* Added new ml2p-docker command added that assists with training and deploying models
  in SageMaker.


0.0.1 (2018-10-19)
------------------

* Initial hackathon release.
