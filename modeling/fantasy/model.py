from config import *
from helpers import *
from pprint import pprint
import numpy as np
import os

from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from modeling.fantasy.wrapper import ModelWrapper
from modeling.fantasy.load import load_xy


class LinearNet(nn.Module):

    def __init__(self, n_feat, n_out=1):

        super().__init__()

        self.linear1 = nn.Linear(n_feat, 50)
        self.linear2 = nn.Linear(50, n_out)

    def forward(self, x):

        x = self.linear1(x)
        x = self.linear2(x)
        return x


def train(
    model,
    X, y,
    epochs=50000,
    batch_size=None,
    learning_rate=1e-5,
    score_func=lambda x, y: np.corrcoef(x, y)[0, 1],
    optimizer=optim.Adam,
    loss_func=nn.MSELoss(),
):
        
    y_comp = y.detach().cpu().numpy().reshape(-1,)
    
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    splits = floor(len(X) / batch_size) if batch_size != None else 1

    X_batches = torch.tensor_split(X, splits)
    y_batches = torch.tensor_split(y, splits)
    
    bar = tqdm(range(epochs))
    for epoch in bar:
        for x, y in zip(X_batches, y_batches):
            y_pred = model(x)
            loss = loss_func(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        y_pred_comp = model(X).detach().cpu().numpy().reshape(-1,)
        score = score_func(y_pred_comp, y_comp)
        bar.set_description(f"Score: {round(score, 5)}")
    
    return model

def load_fantasy_model(_log = True):

    model_dir = f'{fantasy_model_dir}/fantasy_point_model.pkl'
    if os.path.exists(model_dir):
        return torch.load(model_dir)

    if _log: print(f'Training Model | {model_dir}')

    X, y = load_xy()
    n_feat = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2) 

    net = LinearNet(n_feat = n_feat)
    model = ModelWrapper(net, train)
    
    model.train(X_train, y_train)
    model.save(model_dir)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_train = model.detach(y_train)
    
    if _log: print('Train Correlation: ', round(correlation(y_train, y_pred_train), 4))
    if _log: print('Test Correlation: ', round(correlation(y_test, y_pred), 4))
