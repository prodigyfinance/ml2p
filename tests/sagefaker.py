# -*- coding: utf-8 -*-

""" A fake SageMaker client until moto has one of its own. """

import copy
import io
import json


class Paginator:
    """ A fake paginator.

        :param str summary_title:
            The key to store the list of items under in the page dictionary.
        :param list items:
            The list of items to paginate. Individual items are usually
            dictionaries.
        :param int per_page:
            How many items to include in each page. Default: 2.
    """

    def __init__(self, summary_title, items, per_page=2):
        self._summary_title = summary_title
        self._items = items
        self._per_page = per_page

    def paginate(self):
        if not self._items:
            yield {self._summary_title: []}
        for i in range(0, len(self._items), self._per_page):
            start, end = i, i + self._per_page
            yield {self._summary_title: copy.deepcopy(self._items[start:end])}


class Waiter:
    """ A fake waiter.

        :param str selector_name:
            The name of the keyword argument passed to .wait() that
            will select the item to wait on.
        :param function selector:
            The function to retrieve the item to wait on. It should accept
            a single argument (the value passed as selector_name) and return
            the item or None if the item does not exist.
    """

    def __init__(self, selector_name, selector):
        self._selector_name = selector_name
        self._selector = selector

    def wait(self, **kw):
        expected_kws = {
            self._selector_name,
            "WaiterConfig",
        }
        assert set(kw) == expected_kws
        assert self._selector(kw[self._selector_name]) is not None


class SageFakerClient:
    """ A fake SageMaker client. """

    def __init__(self, aws_region):
        self._aws_region = aws_region
        self._training_jobs = []
        self._models = []
        self._endpoint_configs = []
        self._endpoints = []
        self._notebooks = []
        self._notebook_lifecycle_configs = []
        self._repos = []

    def get_paginator(self, name):
        if name == "list_training_jobs":
            return self._list_training_jobs()
        elif name == "list_models":
            return self._list_models()
        elif name == "list_endpoints":
            return self._list_endpoints()
        elif name == "list_endpoint_configs":
            return self._list_endpoint_configs()
        elif name == "list_notebook_instances":
            return self._list_notebook_instances()
        elif name == "list_notebook_instance_lifecycle_configs":
            return self._list_notebook_instance_lifecycle_configs()
        elif name == "list_code_repositories":
            return self._list_code_repositories()
        raise NotImplementedError(
            f"SageFakerClient.get_paginator does not yet support {name}"
        )

    def _list_training_jobs(self):
        return Paginator("TrainingJobSummaries", self._training_jobs)

    def _list_models(self):
        return Paginator("Models", self._models)

    def _list_endpoints(self):
        return Paginator("Endpoints", self._endpoints)

    def _list_endpoint_configs(self):
        return Paginator("EndpointConfigs", self._endpoint_configs)

    def _list_notebook_instances(self):
        return Paginator("NotebookInstances", self._notebooks)

    def _list_notebook_instance_lifecycle_configs(self):
        return Paginator(
            "NotebookInstanceLifecycleConfigs", self._notebook_lifecycle_configs
        )

    def _list_code_repositories(self):
        return Paginator("CodeRepositorySummaryList", self._repos)

    def get_waiter(self, name):
        if name == "training_job_completed_or_stopped":
            return self._training_job_completed_or_stopped()
        elif name == "endpoint_in_service":
            return self._endpoint_in_service()
        elif name == "notebook_instance_stopped":
            return self._notebook_instance_stopped()
        raise NotImplementedError(
            f"SageFakerClient.get_waiter does not yet support {name}"
        )

    def _training_job_completed_or_stopped(self):
        return Waiter("TrainingJobName", self._get_training_job)

    def _endpoint_in_service(self):
        return Waiter("EndpointName", self._get_endpoint)

    def _notebook_instance_stopped(self):
        return Waiter("NotebookInstanceName", self._get_notebook)

    def _get_training_job(self, name):
        jobs = [t for t in self._training_jobs if t["TrainingJobName"] == name]
        if not jobs:
            return None
        if len(jobs) == 1:
            return jobs[0]
        raise RuntimeError(
            f"TrainingJobNames should be unique but {len(jobs)}"
            f" jobs were discovered with the name {name}"
        )

    def create_training_job(self, **kw):
        expected_kws = {
            "TrainingJobName",
            "AlgorithmSpecification",
            "EnableNetworkIsolation",
            "HyperParameters",
            "InputDataConfig",
            "OutputDataConfig",
            "ResourceConfig",
            "RoleArn",
            "StoppingCondition",
            "Tags",
        }
        assert set(kw) == expected_kws
        assert self._get_training_job(kw["TrainingJobName"]) is None
        self._training_jobs.append(kw)
        return copy.deepcopy(kw)

    def describe_training_job(self, TrainingJobName):
        training_job = self._get_training_job(TrainingJobName)
        assert training_job is not None
        return copy.deepcopy(training_job)

    def _get_model(self, name):
        models = [m for m in self._models if m["ModelName"] == name]
        if not models:
            return None
        if len(models) == 1:
            return models[0]
        raise RuntimeError(
            f"ModelNames should be unique but {len(models)}"
            f" models were discovered with the name {name}"
        )

    def create_model(self, **kw):
        expected_kws = {
            "ModelName",
            "PrimaryContainer",
            "ExecutionRoleArn",
            "Tags",
            "EnableNetworkIsolation",
        }
        assert set(kw) == expected_kws
        assert self._get_model(kw["ModelName"]) is None
        self._models.append(kw)
        return copy.deepcopy(kw)

    def describe_model(self, ModelName):
        model = self._get_model(ModelName)
        assert model is not None
        return copy.deepcopy(model)

    def delete_model(self, ModelName):
        model = self._get_model(ModelName)
        assert model is not None
        self._models.remove(model)
        return copy.deepcopy(model)

    def _get_endpoint_config(self, name):
        config = [e for e in self._endpoint_configs if e["EndpointConfigName"] == name]
        if not config:
            return None
        if len(config) == 1:
            return config[0]
        raise RuntimeError(
            f"EndpointConfigName should be unique but {len(config)}"
            f" endpoint configs were discovered with the name {name}"
        )

    def create_endpoint_config(self, **kw):
        expected_kws = {
            "EndpointConfigName",
            "Tags",
            "ProductionVariants",
        }
        assert set(kw) == expected_kws
        assert self._get_endpoint_config(kw["EndpointConfigName"]) is None
        self._endpoint_configs.append(kw)
        return copy.deepcopy(kw)

    def describe_endpoint_config(self, EndpointConfigName):
        endpoint_cfg = self._get_endpoint_config(EndpointConfigName)
        assert endpoint_cfg is not None
        return copy.deepcopy(endpoint_cfg)

    def delete_endpoint_config(self, EndpointConfigName):
        endpoint_cfg = self._get_endpoint_config(EndpointConfigName)
        assert endpoint_cfg is not None
        self._endpoint_configs.remove(endpoint_cfg)
        return copy.deepcopy(endpoint_cfg)

    def _get_endpoint(self, name):
        endpoint = [e for e in self._endpoints if e["EndpointName"] == name]
        if not endpoint:
            return None
        if len(endpoint) == 1:
            return endpoint[0]
        raise RuntimeError(
            f"EndpointName should be unique but {len(endpoint)}"
            f" endpoints were discovered with the name {name}"
        )

    def create_endpoint(self, **kw):
        expected_kws = {"EndpointConfigName", "EndpointName", "Tags"}
        assert set(kw) == expected_kws
        assert self._get_endpoint(kw["EndpointName"]) is None
        kw[
            "EndpointArn"
        ] = f"arn:aws:sagemaker:{self._aws_region}:12345:endpoint/{kw['EndpointName']}"
        self._endpoints.append(kw)
        return copy.deepcopy(kw)

    def describe_endpoint(self, EndpointName):
        endpoint = self._get_endpoint(EndpointName)
        assert endpoint is not None
        return copy.deepcopy(endpoint)

    def delete_endpoint(self, EndpointName):
        endpoint = self._get_endpoint(EndpointName)
        assert endpoint is not None
        self._endpoints.remove(endpoint)
        return copy.deepcopy(endpoint)

    def _get_lifecycle_config(self, name):
        lifecycle_cfgs = [
            lifecyle
            for lifecyle in self._notebook_lifecycle_configs
            if lifecyle["NotebookInstanceLifecycleConfigName"] == name
        ]
        if not lifecycle_cfgs:
            return None
        if len(lifecycle_cfgs) == 1:
            return lifecycle_cfgs[0]
        raise RuntimeError(
            f"NotebookInstanceLifecycleConfigName should be unique but"
            f" {len(lifecycle_cfgs)} lifecycle configs were discovered with"
            f" the name {name}"
        )

    def create_notebook_instance_lifecycle_config(self, **kw):
        expected_kws = {"NotebookInstanceLifecycleConfigName"}
        assert set(kw) == expected_kws
        assert (
            self._get_lifecycle_config(kw["NotebookInstanceLifecycleConfigName"])
            is None
        )
        self._notebook_lifecycle_configs.append(kw)
        return copy.deepcopy(kw)

    def describe_notebook_instance_lifecycle_config(
        self, NotebookInstanceLifecycleConfigName
    ):
        lifecycle_config = self._get_lifecycle_config(
            NotebookInstanceLifecycleConfigName
        )
        assert lifecycle_config is not None
        return copy.deepcopy(lifecycle_config)

    def delete_notebook_instance_lifecycle_config(
        self, NotebookInstanceLifecycleConfigName
    ):
        lifecycle_config = self._get_lifecycle_config(
            NotebookInstanceLifecycleConfigName
        )
        assert lifecycle_config is not None
        self._notebook_lifecycle_configs.remove(lifecycle_config)
        return copy.deepcopy(lifecycle_config)

    def _get_notebook(self, name):
        notebooks = [nb for nb in self._notebooks if nb["NotebookInstanceName"] == name]
        if not notebooks:
            return None
        if len(notebooks) == 1:
            return notebooks[0]
        raise RuntimeError(
            f"NotebookInstanceName should be unique but {len(notebooks)} notebooks were"
            f" discovered with the name {name}"
        )

    def create_notebook_instance(self, **kw):
        expected_kws = {
            "NotebookInstanceName",
            "InstanceType",
            "RoleArn",
            "Tags",
            "LifecycleConfigName",
            "VolumeSizeInGB",
            "DirectInternetAccess",
        }
        repo = kw.pop("DefaultCodeRepository", None)  # optional argument
        assert set(kw) == expected_kws
        assert self._get_notebook(kw["NotebookInstanceName"]) is None
        kw["DefaultCodeRepository"] = repo
        kw["NotebookInstanceStatus"] = "Pending"
        self._notebooks.append(kw)
        return copy.deepcopy(kw)

    def describe_notebook_instance(self, NotebookInstanceName):
        notebook = self._get_notebook(NotebookInstanceName)
        assert notebook is not None
        return copy.deepcopy(notebook)

    def delete_notebook_instance(self, NotebookInstanceName):
        notebook = self._get_notebook(NotebookInstanceName)
        assert notebook is not None
        self._notebooks.remove(notebook)
        return copy.deepcopy(notebook)

    def create_presigned_notebook_instance_url(self, NotebookInstanceName):
        notebook = self._get_notebook(NotebookInstanceName)
        assert notebook is not None
        return {
            "AuthorizedUrl": f"https://example.com/{notebook['NotebookInstanceName']}"
        }

    def start_notebook_instance(self, NotebookInstanceName):
        notebook = self._get_notebook(NotebookInstanceName)
        assert notebook is not None
        notebook["NotebookInstanceStatus"] = "InService"

    def stop_notebook_instance(self, NotebookInstanceName):
        notebook = self._get_notebook(NotebookInstanceName)
        assert notebook is not None
        notebook["NotebookInstanceStatus"] = "Stopped"

    def _get_repo(self, name):
        repos = [r for r in self._repos if r["CodeRepositoryName"] == name]
        if not repos:
            return None
        if len(repos) == 1:
            return repos[0]
        raise RuntimeError(
            f"CodeRepositoryName should be unique but {len(repos)} repos were"
            f" discovered with the name {name}"
        )

    def create_code_repository(self, **kw):
        expected_kws = {
            "CodeRepositoryName",
            "GitConfig",
        }
        assert set(kw) == expected_kws
        assert self._get_repo(kw["CodeRepositoryName"]) is None
        self._repos.append(kw)
        return copy.deepcopy(kw)

    def describe_code_repository(self, CodeRepositoryName):
        repo = self._get_repo(CodeRepositoryName)
        assert repo is not None
        return copy.deepcopy(repo)

    def delete_code_repository(self, CodeRepositoryName):
        repo = self._get_repo(CodeRepositoryName)
        assert repo is not None
        self._repos.remove(repo)
        return copy.deepcopy(repo)


class SageFakerRuntimeClient:
    """ A fake SageMaker Runtime client. """

    def __init__(self, sagefaker):
        self._sagefaker = sagefaker
        self._invokes = []

    def invoke_endpoint(self, **kw):
        expected_kws = {"EndpointName", "Body", "ContentType", "Accept"}
        assert set(kw) == expected_kws
        assert kw["ContentType"] == "application/json"
        assert kw["Accept"] == "application/json"
        data = json.loads(kw["Body"])
        self._invokes.append({"EndpointName": kw["EndpointName"], "Data": data})
        return {"Body": io.BytesIO(json.dumps({"inputs": data}).encode("utf-8"))}
