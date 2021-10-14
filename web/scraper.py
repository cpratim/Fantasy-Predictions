import requests
from pprint import pprint
from bs4 import BeautifulSoup
import lxml

recorded_stats = [
    'G', 
    'GS', 
    'MP', 
    'FG', 
    'FGA', 
    'FG%', 
    '3P', 
    '3PA', 
    '3P%', 
    '2P', 
    '2PA', 
    '2P%', 
    'eFG%', 
    'FT', 
    'FTA', 
    'FT%', 
    'ORB', 
    'DRB', 
    'TRB', 
    'AST', 
    'STL', 
    'BLK', 
    'TOV', 
    'PF', 
    'PTS'
]


def make_url(name):
    base = 'https://www.basketball-reference.com/search/search.fcgi?search=' + '+'.join(name.split(' '))
    raw = requests.get(base)
    if 'players' in raw.url:
        return raw.url 
    soup = BeautifulSoup(raw.text, 'lxml')
    url = soup.find('div', {'class': 'search-item-url'}).text
    return 'https://www.basketball-reference.com' + url

def player_stats(player):

    base = make_url(player)
    raw = requests.get(base).text
    soup = BeautifulSoup(raw, 'lxml')
    stats = {}
    table = soup.find('table', {'id': 'per_game'})
    for r in table.find_all('tr', {'class': 'full_table'}):
        szn = '20' + r.find('th').a.text[5:]
        stats[szn] = {}
        raw_stats = r.find_all('td')[4:]
        for n, s in zip(recorded_stats, raw_stats):
            try:
                stats[szn][n] = float(s.text)
            except:
                stats[szn][n] = 0

        games = stats[szn]['G']
        for n, s in zip(recorded_stats[2:], raw_stats[2:]):
            if '%' not in n:
                try:
                    tS = games * float(s.text)
                    stats[szn][f'T{n}'] = round(tS)
                except:
                    stats[szn][f'T{n}'] = 0

    return stats

if __name__ == '__main__':

    print(player_stats('Clint Capela'))