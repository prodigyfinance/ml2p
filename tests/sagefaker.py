# -*- coding: utf-8 -*-

""" A fake SageMaker client until moto has one of its own. """

import copy


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

    def __init__(self):
        self._training_jobs = []
        self._models = []

    def get_paginator(self, name):
        if name == "list_training_jobs":
            return self._list_training_jobs()
        elif name == "list_models":
            return self._list_models()
        raise NotImplementedError(
            f"SageFakerClient.get_paginator does not yet support {name}"
        )

    def _list_training_jobs(self):
        return Paginator("TrainingJobSummaries", self._training_jobs)

    def _list_models(self):
        return Paginator("Models", self._models)

    def get_waiter(self, name):
        if name == "training_job_completed_or_stopped":
            return self._training_job_completed_or_stopped()
        raise NotImplementedError(
            f"SageFakerClient.get_waiter does not yet support {name}"
        )

    def _training_job_completed_or_stopped(self):
        return Waiter("TrainingJobName", self._get_training_job)

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
