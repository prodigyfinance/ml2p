# -*- coding: utf-8 -*-

""" A small script for creating a Boston house price data traing set.
"""

import pandas
import sklearn.datasets


def write_boston_csv(csv_name):
    """ Write a Boston house price training dataset. """
    boston = sklearn.datasets.load_boston()
    df = pandas.DataFrame(boston["data"], columns=boston["feature_names"])
    df["target"] = boston["target"]
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    write_boston_csv("house-prices.csv")
