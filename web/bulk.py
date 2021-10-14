import json
import os
from config import *
from scraper import player_stats
from helpers import *
from pprint import pprint

special_cases = {
    'C.J. McCollum': 'CJ McCollum'
}

def bulk_load(seasons = ['20' + str(i) for i in range(14, 22)], _log = True):
    
    not_found = []
    for season in seasons:
        players = list(load_json(f'{fantasy_dir}/{season}.json').keys())
        for player in players:
            f_out = '_'.join([s.lower() for s in player.split(' ')]) + '.json'
            if player in special_cases:
                player = special_cases[player]
            
            if not os.path.exists(f'{player_dir}/{f_out}'):
                try:
                    stats = player_stats(player)
                    dump_json(stats, f'{player_dir}/{f_out}')

                    if _log: print(f'[{timestamp()}] Stats for "{player}" COLLECTED | {player_dir}/{f_out}')
                except:
                    if player not in not_found:
                        not_found.append(player)
                        if _log: print(f'[{timestamp()}] Stats for "{player}" NOT FOUND')
    return not_found

if __name__ == '__main__':

    not_found = bulk_load()
    pprint(not_found)
    #print(len(os.listdir(player_dir)))
    #print(player_stats('Kevin Durant'))