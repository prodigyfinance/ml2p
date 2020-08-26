# -*- coding: utf-8 -*-

""" A fake SageMaker client until moto has one of its own. """


class Paginator:
    """ A fake paginator """

    def __init__(self, summary_title, items, per_page=2):
        self._summary_title = summary_title
        self._items = items
        self._per_page = per_page

    def paginate(self):
        for i in range(0, len(self._items), self._per_page):
            start, end = i, i + self._per_page
            yield {self._summary_title: self._items[start:end]}


class SageFakerClient:
    """ A fake SageMaker client. """

    def get_paginator(self, name):
        if name == "list_training_jobs":
            return self._list_training_jobs()
        raise NotImplementedError(
            f"SageFakerClient.get_paginator does not yet support {name}"
        )

    def _list_training_jobs(self):
        return Paginator(
            "TrainingJobSummaries",
            [
                {"TrainingJobName": "xxx"},
                {"TrainingJobName": "yyy"},
                {"TrainingJobName": "my-models-zzz"},
            ],
        )
