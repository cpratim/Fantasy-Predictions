import numpy as np
from helpers import *
from config import *

def load_xy(seasons = ['20' + str(i) for i in range(14, 22)]):

    stat_cache = {}
    X, y = [], []
    for season in seasons:

        ranking = load_json(f'{fantasy_dir}/{season}.json')
        for player, score in ranking.items():
            if player not in stat_cache:
                fname = '_'.join([s.lower() for s in player.split(' ')]) + '.json'
                stat_cache[player] = load_json(f'{player_dir}/{fname}')
            try:
                stats = list(stat_cache[player][season].values())
            except:
                continue
            X.append(stats)
            y.append(score)
        
    return np.array(X), np.array(y)
    