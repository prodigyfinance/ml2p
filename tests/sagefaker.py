# -*- coding: utf-8 -*-

""" A fake SageMaker client until moto has one of its own. """

import copy


class Paginator:
    """ A fake paginator. """

    def __init__(self, summary_title, items, per_page=2):
        self._summary_title = summary_title
        self._items = items
        self._per_page = per_page

    def paginate(self):
        for i in range(0, len(self._items), self._per_page):
            start, end = i, i + self._per_page
            yield {self._summary_title: copy.deepcopy(self._items[start:end])}


class Waiter:
    """ A fake waiter. """

    def __init__(self, selector_name):
        self._selector_name = selector_name

    def wait(self, **kw):
        expected_kws = {
            self._selector_name,
            "WaiterConfig",
        }
        assert set(kw) == expected_kws


class SageFakerClient:
    """ A fake SageMaker client. """

    def __init__(self):
        self._training_jobs = []

    def get_paginator(self, name):
        if name == "list_training_jobs":
            return self._list_training_jobs()
        raise NotImplementedError(
            f"SageFakerClient.get_paginator does not yet support {name}"
        )

    def _list_training_jobs(self):
        return Paginator("TrainingJobSummaries", self._training_jobs)

    def get_waiter(self, name):
        if name == "training_job_completed_or_stopped":
            return self._training_job_completed_or_stopped()
        raise NotImplementedError(
            f"SageFakerClient.get_waiter does not yet support {name}"
        )

    def _training_job_completed_or_stopped(self):
        return Waiter("TrainingJobName")

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
