import json
from config import *
from helpers import *

def fantasy_rankings(season, nlines=100):

    file = f'{raw_dir}/{season}.txt'
    rankings = {}
    with open(file, 'r') as h:
        lines = h.readlines()
        players, stats = lines[:nlines], lines[nlines:]
        for p, s in zip(players, stats):
            name = p.split('\t')[1].strip()
            points = float(s.split('\t')[-1].strip())
            rankings[name] = points

    return rankings

def merge_raw(seasons = ['20' + str(i) for i in range(14, 22)], _log = True):

    for s in seasons:
        rankings = fantasy_rankings(s)
        dump_json(rankings, f'{fantasy_dir}/{s}.json')
        if _log: print(f'[{timestamp()}] MERGED: {s} | {fantasy_dir}/{s}.json')

if __name__ == '__main__':

    merge_raw()
