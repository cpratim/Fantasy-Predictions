# from modeling.fantasy import model

# model.load_fantasy_model()
import numpy as np
from modeling.player.model import PlayerModel, recorded_stats
from config import *
from helpers import *

player = 'Lebron James'

f = '_'.join([l.lower() for l in player.split(' ')])
stats_record = load_json(f'{player_dir}/{f}.json')

model = PlayerModel()
model.train(player)
preds = model.predict(22)
average = []
for k, v in stats_record.items():
	average.append(list(v.values()))
mmax = np.max(np.array(average).T, axis=1)
average = np.mean(np.array(average).T, axis=1)


for k, v, p, a, m in zip(list(recorded_stats.keys()), preds, list(stats_record['2021'].values()), average, mmax):
	print(f'{k}: P: {round(v, 4)} | L: {round(p, 4)} | A: {round(a, 4)} | M: {round(m, 4)}')