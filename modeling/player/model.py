import numpy as np
from modeling.player.load import load_xy
from modeling.player.regression import (
    Mean,
    MeanLinear,
    Linear,
)

recorded_stats = {
    "G": Mean(),
    "GS": MeanLinear(),
    "MP": MeanLinear(),
    "FG": MeanLinear(),
    "FGA": MeanLinear(),
    "FG%": Mean(),
    "3P": MeanLinear(),
    "3PA": MeanLinear(),
    "3P%": Mean(),
    "2P": MeanLinear(),
    "2PA": MeanLinear(),
    "2P%": MeanLinear(),
    "eFG%": Mean(),
    "FT": MeanLinear(),
    "FTA": MeanLinear(),
    "FT%": Mean(),
    "ORB": MeanLinear(),
    "DRB": MeanLinear(),
    "TRB": MeanLinear(),
    "AST": MeanLinear(),
    "STL": MeanLinear(),
    "BLK": MeanLinear(),
    "TOV": MeanLinear(),
    "PF": MeanLinear(),
    "PTS": MeanLinear(),
    "TMP": MeanLinear(),
    "TFG": MeanLinear(),
    "TFGA": MeanLinear(),
    "T3P": MeanLinear(),
    "T3PA": MeanLinear(),
    "T2P": MeanLinear(),
    "T2PA": MeanLinear(),
    "TFT": MeanLinear(),
    "TFTA": MeanLinear(),
    "TORB": MeanLinear(),
    "TDRB": MeanLinear(),
    "TTRB": MeanLinear(),
    "TAST": MeanLinear(),
    "TSTL": MeanLinear(),
    "TBLK": MeanLinear(),
    "TTOV": MeanLinear(),
    "TPF": MeanLinear(),
    "TPTS": MeanLinear(),
}


class PlayerModel(object):

    def __init__(self):
        self.models = []

    def train(self, player):

        X, y = load_xy(player)
        skel = list(recorded_stats.values())
        for _y, s in zip(y, skel):
            s.fit(X, _y)
            self.models.append(s)
            

    def predict(self, X):
        return np.array([m.predict(X) for m in self.models])

