# -*- coding: utf-8 -*-

""" Train the Boston house prices model on your local machine.
"""

import pandas as pd
from ml2p.core import LocalEnv
import model


def train(env):
    """ Train and save the model locally. """
    trainer = model.BostonModel().trainer(env)
    trainer.train()


def predict(env):
    """ Load a model and make predictions locally. """
    predictor = model.BostonModel().predictor(env)
    predictor.setup()
    data = pd.read_csv("house-prices.csv")
    house = dict(data.iloc[0])
    result = predictor.invoke(house)
    print(result)


if __name__ == "__main__":
    env = LocalEnv(".", "ml2p.yml")
    train(env)
    predict(env)
