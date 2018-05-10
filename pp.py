import pymongo
import pandas as pd
import numpy as np

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def penalty_over(time,period):
    penalty_over = time.split(':')
    penalty_over[0] = str(int(time.split(':')[0])+2).zfill(2)
    if penalty_over[0]>'20':
        period +=1
        penalty_over[0] = str(int(penalty_over[0])-20).zfill(2)
    return ':'.join(penalty_over), period

def get_all_shot_coords(db,coll):
    shots = pd.DataFrame()
    blocks = pd.DataFrame()
    misses = pd.DataFrame()
    goals = pd.DataFrame()
    for gmid in coll.distinct('gamePk'):
        print(gmid)
        all_plays = coll.find_one({'gamePk':gmid})['liveData']['plays']['allPlays']
        home = coll.find_one({'gamePk':gmid})['gameData']['teams']['home']['id']
        away = coll.find_one({'gamePk':gmid})['gameData']['teams']['away']['id']
        on_penalty = False
        over='00:00'
        period = 1
        for play in all_plays:
            if play['result']['event']=='Penalty' and on_penalty == False:
                on_penalty = True
                over,period = penalty_over(play['about']['periodTime'],play['about']['period'])
                sh_team = play['team']['id']
            elif play['about']['periodTime']>over and period == play['about']['period']:
                on_penalty = False

            if play['about']['periodType']!='SHOOTOUT':
                play_type = play['result']['eventTypeId']
                if ('SHOT' in play_type or 'GOAL' in play_type) and bool(play['coordinates']):
                    if play_type=='GOAL' and play['result']['emptyNet']==True:
                        continue
                    coords = play['coordinates']
                    coords['type'] = play_type
                    coords['period'] = play['about']['period']
                    coords.update(play['about']['goals'])
                    if on_penalty:
                        if play['team']['id']==sh_team:
                            coords['pp_status']='sh'
                        else:
                            coords['pp_status']='pp'
                        if play_type == 'GOAL':
                            on_penalty=False
                    else:
                        coords['pp_status']='even'

                    coords['home'] = home
                    coords['away'] = away
                    for player in play['players']:
                        if not player['player']['id']:
                            print('no player')
                        if player['playerType']=='Shooter' and coll.find_one({'gamePk':gmid})['gameData']['players']['ID' + str(player['player']['id'])]['primaryPosition']['code']!='G':
                            coords['shooter']=int(player['player']['id'])
                        if player['playerType']=='Scorer' and coll.find_one({'gamePk':gmid})['gameData']['players']['ID' + str(player['player']['id'])]['primaryPosition']['code']!='G':
                            coords['scorer']=int(player['player']['id'])
                        if player['playerType']=='Goalie':
                            coords['goalie']=int(player['player']['id'])
                        if player['playerType']=='Blocker':
                            coords['defender']=int(player['player']['id'])
                    if play_type =='MISSED_SHOT':
                        coords['m_team']=play['team']['id']
                        if all(pos in coords.keys() for pos in misses.keys()):
                            misses = misses.append([coords])
                    elif play_type =='BLOCKED_SHOT':
                        coords['d_team']=play['team']['id']
                        if all(pos in coords.keys() for pos in blocks.keys()):
                            blocks = blocks.append([coords])
                    elif play_type =='SHOT':
                        if all(pos in coords.keys() for pos in shots.keys()):
                            shots = shots.append([coords])
                    elif play_type =='GOAL':
                        if all(pos in coords.keys() for pos in goals.keys()):
                            goals = goals.append([coords])
    return misses,blocks,shots,goals


if __name__ == '__main__':
    db = _init_mongo()
    coll = db.year_2017

    gmid = 2017020001
    t_missed, t_blocked,t_shots,t_goals = get_all_shot_coords(db,coll)
    #
    t_missed.to_csv('data/2017_missed.csv')
    t_blocked.to_csv('data/2017_blocked.csv')
    t_shots.to_csv('data/2017_shots.csv')
    t_goals.to_csv('data/2017_goals.csv')
