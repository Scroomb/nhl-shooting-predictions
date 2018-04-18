from imblearn.over_sampling import SMOTE
from make_graphs import player_shots_goals, goalie_shots_goals, load_shots_goals
import make_distributions
import pymongo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle as pkl

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def single_row(db,row,p_type):
    goalie = str(int(row['goalie']))
    x = int(row['x'])
    scorer = str(int(row[p_type]))
    y = int(row['y'])+42
    g_g_density = 1-retrieve_density(db,goalie,'goalie','save_dist').reshape(85,100)
    p_g_density = retrieve_density(db,scorer,'player','goal_dist').reshape(85,100)
    g_s_density = retrieve_density(db,goalie,'goalie','shot_dist').reshape(85,100)
    p_s_density = retrieve_density(db,scorer,'player','shot_dist').reshape(85,100)
    g_g_den = g_g_density[y][x]
    p_g_den = p_g_density[y][x]
    p_s_den = p_s_density[y][x]  # Use player shot density?
    return np.append(row,[p_g_den,g_g_den,p_s_den])

def make_data(db,shots,goals):
    coll = db['players_year_20172018']
    goal_data = []

    for row in goals.iterrows():
        if coll.find_one({'player_id':str(int(row[1]['scorer']))}) and coll.find_one({'player_id':str(int(row[1]['goalie']))}):
            shooter_goal = bool('goal_dist' in coll.find_one({'player_id':str(int(row[1]['scorer']))}).keys())
            shooter_shot = bool('shot_dist' in coll.find_one({'player_id':str(int(row[1]['scorer']))}).keys())
            goalie_save = bool('save_dist' in coll.find_one({'player_id':str(int(row[1]['goalie']))}).keys())
            if not(shooter_goal and goalie_save and shooter_shot):
                continue
            row_d = single_row(db,row[1],'scorer')
            row_d = np.append(row_d,1)
            goal_data.append(row_d)
    goal_data = np.array(goal_data)
    gd = goal_data[:,2:]

    shot_data = []
    for row in shots.iterrows():
        row_d = row[1]
        if coll.find_one({'player_id':str(int(row_d['shooter']))}) and coll.find_one({'player_id':str(int(row_d['goalie']))}):
            shooter_goal = bool('goal_dist' in coll.find_one({'player_id':str(int(row_d['shooter']))}).keys())
            goalie_save = bool('save_dist' in coll.find_one({'player_id':str(int(row_d['goalie']))}).keys())
            shooter_shot = bool('shot_dist' in coll.find_one({'player_id':str(int(row[1]['shooter']))}).keys())
            if not(shooter_goal and goalie_save and shooter_shot):
                continue
            row_data = single_row(db,row_d,'shooter')
            row_data = np.append(row_data,0)
            shot_data.append(row_data)
    shot_data = np.array(shot_data)
    sd = shot_data[:,2:]
    td = np.concatenate((gd,sd),axis=0)

    return td

def generate_prediction_data(shooter_id,goalie_id,scaler):
    xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
    xy = np.vstack([xx.ravel(),yy.ravel()])

    shooter = int(shooter_id)
    goalie = int(goalie_id)
    shooter = np.full((8500,1),shooter)
    goalie = np.full((8500,1),goalie)

    unseen = np.concatenate((shooter,goalie,xy.T),axis=1)
    unseen = pd.DataFrame(unseen,columns=['scorer','goalie','x','y'])

    unseen_data = []
    for row in unseen.iterrows():
        row_d = single_row(db,row[1],'scorer')
        unseen_data.append(row_d)
    unseen_data = np.array(unseen_data)
    unseen_data_for_model = unseen_data[:,2:]
    return scaler.transform(unseen_data_for_model)

def retrieve_density(db,player,position,dist_type,year=2017):
    coll = db['players_year_'+str(year)+str(year+1)]
    if position == 'goalie':
        y = coll.find_one({'player_id':player})[dist_type][0]
    else:
        y = coll.find_one({'player_id':player})[dist_type][0]
    return pkl.loads(y)

def scale_transform_split(td):
    td_x = td[:,:-1]
    td_y = td[:,-1]
    sm = SMOTE(kind='regular')
    x_res, y_res = sm.fit_sample(td_x,td_y)

    x_train,x_test,y_train,y_test = train_test_split(x_res,y_res,test_size=.2)
    x_scaler = StandardScaler()

    x_std = x_scaler.fit_transform(x_train)
    x_t_std = x_scaler.transform(x_test)
    return x_std,x_t_std,y_train,y_test,x_scaler

if __name__ == '__main__':
    db = _init_mongo()
    shots, goals = load_shots_goals(2017)
    nm_shots, nm_goals = player_shots_goals(8477492,shots,goals)

    td = make_data(db,shots,goals)
    np.savetxt('data/2017_shots_goals_goalie.csv',delimiter=',')
    # td = np.genfromtxt('data/2017_total_data.csv',delimiter=',')
    # x_std,x_t_std,y_train,y_test,x_scaler = scale_transform_split(td)
    # pred_data = generate_prediction_data(8477492,8471469,x_scaler)
