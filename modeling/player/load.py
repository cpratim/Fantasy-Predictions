import numpy as np
from helpers import *
from config import *


def load_xy(player):

    player = '_'.join([l.lower() for l in player.split(' ')])

    X, y = [], []
   
    stats_record = load_json(f'{player_dir}/{player}.json')
    for season, stats in stats_record.items():

        X.append(int(season[-2:]))
        y.append(list(stats.values()))

    return np.array(X), np.array(y).T
