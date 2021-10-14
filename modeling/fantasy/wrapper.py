from sklearn.pipeline import Pipeline
import torch

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

sk_pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),
    ]
)

class ModelWrapper(object):

    def __init__(self, model, train_func, feature_pipeline = sk_pipeline):

        self.model = model
        self.train_func = train_func
        self.feature_pipeline = feature_pipeline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, X, y):

        X = self.feature_pipeline.fit_transform(X)
        y = y.reshape(-1, 1)
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        X = X.to(self.device)
        y = y.to(self.device)
        self.model = self.model.to(self.device)
        self.model = self.train_func(self.model, X, y)

    def detach(self, y):
        return y.detach().cpu().numpy().reshape(-1,)

    def predict(self, X):
        y = model(X)
        return self.detach(y)  

    def save(self, dir):
    
        torch.save(self.model.state_dict(), dir)