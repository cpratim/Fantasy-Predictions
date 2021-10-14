import numpy as np
from modeling.player.model import PlayerModel, recorded_stats
from modeling.fantasy.model import load_fantasy_model
from helpers import *
from config import *
from pprint import pprint
from warnings import filterwarnings
import math

filterwarnings('ignore')

fantasy_model = load_fantasy_model()

latest_rankings = load_json(f'{fantasy_dir}/2021.json')
predictions = []
ranking_predictions = {}

for player in latest_rankings:
	model = PlayerModel()
	model.train(player)
	stats_pred = np.nan_to_num([model.predict(22),], posinf = 0, neginf = 0, nan = 0)
	X = np.array(stats_pred)

	pred = fantasy_model.predict(X).item()
	if math.isnan(pred):
		pred = 0
	predictions.append((player, pred))

print()
for player, score in sorted(predictions, key = lambda x: x[1], reverse = True):

	print(f'{player}: {score}')

