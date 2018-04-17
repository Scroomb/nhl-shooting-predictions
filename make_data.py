from imblearn.over_sampling import SMOTE
from make_graphs import player_shots_goals, goalie_shots_goals, load_shots_goals
import make_distributions
import pymongo
import pandas as pd
import numpy as np

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def make_data(shots,goals):
    goal_data = []
    for row in goals.iterrows():
        row_d = make_distributions.single_row(row[1],'scorer')
        row_d = np.append(row_d,1)
        goal_data.append(row_d)
    goal_data = np.array(goal_data)
    gd = goal_data[:,2:]

    shot_data = []
    for row in shots.iterrows():
        row_d = row[1]
        if bool('shot_dist' in coll.find_one({'player_id':str(int(row_d['goalie']))}).keys()):
            row_data = make_distributions.single_row(row_d,'shooter')
            row_data = np.append(row_data,0)
            shot_data.append(row_data)
    shot_data = np.array(shot_data)
    sd = shot_data[:,2:]
    td = np.concatenate((gd,sd),axis=0)
    td_x = td[:,:-1]
    td_y = td[:,-1]
    return td_x, td_y

if __name__ == '__main__':
    db = _init_mongo()
    shots, goals = load_shots_goals(2017)
    nm_shots, nm_goals = player_shots_goals(8477492,shots,goals)

    td_x, td_y = make_data(nm_shots,nm_goals)

    sm = SMOTE(kind='regular')

    X_res, y_res = sm.fit_sample(td_x,td_y)
