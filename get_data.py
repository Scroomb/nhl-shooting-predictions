import requests
import pymongo
import pandas as pd
import numpy as np
from make_graphs import load_shots_goals

def make_request_game(coll,year):
    if year == '2017':
        up_range = 1272
    elif year == '2012':
        up_range = 720
    else:
        up_range=1231
    for gmid in range(1,up_range):
        endpoint = 'https://statsapi.web.nhl.com/api/v1/game/'+ year +'02'+ str(gmid).zfill(4) + '/feed/live.json'
        json = single_query(endpoint)
        try:
            coll.insert_one(json)
            print(gmid, ' out of ', up_range)
        except pymongo.errors.DuplicateKeyError:
            print('duplicate found')
            continue
    # return scoring_coords

def make_request_player_stats(coll,season,p_id):
    endpoint = 'https://statsapi.web.nhl.com/api/v1/people/'+ p_id + '/stats?stats=statsSingleSeason&season=' + season
    json = single_query(endpoint)
    if bool(json):
        try:
            coll.insert_one({'player_id':p_id})
            coll.update_one({'player_id':p_id},{'$push':{'stats':json['stats'][0]['splits'][0]}})
            # print(gmid, ' out of ', up_range)
        except pymongo.errors.DuplicateKeyError:
            print('duplicate found')

def make_request_player_info(coll,p_id):
    endpoint = 'https://statsapi.web.nhl.com/api/v1/people/' + p_id
    json = single_query(endpoint)
    if bool(json):
        if bool(coll.find_one({'id':json['people'][0]['id']})):
            print('player in db')
            return
        try:
            coll.insert_one(json['people'][0])
        except pymongo.errors.DuplicateKeyError:
            print('duplicate found')

def single_query(endpoint):
    response = requests.get(endpoint)
    if response.status_code == 200:
        print('request successful')
        return response.json()
    else:
        print('WARNING status code {}'.format(response.status_code))
        return False

def get_players_info(db,players):
    for player in players:
        make_request_player_info(db['players'],str(player))


def get_player_data(db,year,shots):
    season = str(year) + str(year+1)
    collection = 'players_year_' + season
    coll = db[collection]
    for player in shots.shooter.unique():
        make_request_player_stats(coll,season,str(player))
    for player in shots.goalie.unique():
        make_request_player_stats(coll,season,str(player))

def save_dataframe(df,year):
    df.to_csv(year+'_coord.csv')

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

if __name__ == '__main__':
    db = _init_mongo()
    shots, goals = load_shots_goals(2017)

    # get_player_data(db,2017,shots)
    players = np.concatenate([goals.goalie.unique(),goals.scorer.unique()])
    get_players_info(db,players)
    #Get shots/goals data
    # for y in range(2010,2018):
    #     collection = 'year_'+str(y)
    #     coll = db[collection]
    #     make_request(coll,str(y))
