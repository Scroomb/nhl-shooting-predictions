from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from make_graphs import player_shots_goals, goalie_shots_goals, load_shots_goals
import pymongo
import pickle as pkl
import bson

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def make_shot_density(shots,cv=20):
    xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
    xy = np.vstack([xx.ravel(),yy.ravel()])
    grid = GridSearchCV(KernelDensity(),{'bandwidth':np.linspace(0.1,20,500)},cv=cv,n_jobs=-1,verbose=1)
    grid.fit(shots)
    return np.exp(grid.best_estimator_.score_samples(xy.T))

def save_density_to_db(year,player,density,dist_type,position):
    coll = db[year]
    y = bson.binary.Binary(pkl.dumps(density,protocol=2))
    if position == 'goalie':
        coll.update_one({'player_id':player},{'$push':{dist_type:y}})
    else:
        coll.update_one({'player_id':player},{'$push':{dist_type:y}})

def retrieve_density(player,position,dist_type,year=2017):
    coll = db['players_year_'+str(year)+str(year+1)]
    if position == 'goalie':
        y = coll.find_one({'player_id':player})[dist_type][0]
    else:
        y = coll.find_one({'player_id':player})[dist_type][0]
    return pkl.loads(y)

def generate_all_distributions(shots,goals):
    goalies = []
    for x in goals.goalie.unique():
        goalies.append(('goalie',x))
    goalies = np.array(goalies)
    scorers = []
    for x in goals.scorer.unique():
        scorers.append(('scorer',x))
    scorers = np.array(scorers)
    for goalie in goalies:
        shots_g, goals_g = goalie_shots_goals(int(goalie[1]),shots,goals)
        if goals_g.shape[0]<=20:
            if goals_g.shape[0]==1:
                continue
            g_cv = goals_g.shape[0]
        else:
            g_cv = 20
        if shots_g.shape[0]<=10:
            continue
        else:
            s_cv = 10
        if 'save_dist' not in db.players_year_20172018.find_one({'player_id':goalie[1]},{'save_dist':1}).keys():
            density = make_shot_density(goals_g[['x','y']].values,g_cv)
            save_density_to_db('players_year_20172018',goalie[1],density,'save_dist',goalie[0])
        else:
            print(goalie[1], ' goalie save dist exists')
        if 'shot_dist' not in db.players_year_20172018.find_one({'player_id':goalie[1]},{'shot_dist':1}).keys():
            density = make_shot_density(shots_g[['x','y']].values,s_cv)
            save_density_to_db('players_year_20172018',goalie[1],density,'shot_dist',goalie[0])
        else:
            print(goalie[1], ' goalie shot dist exists')

    for scorer in scorers:
    # for i in range(647,721):
        scorer = scorers[i]
        shots_p, goals_p = player_shots_goals(int(scorer[1]),shots,goals)
        if goals_p.shape[0]<=10:
            if goals_p.shape[0]==1:
                continue
            g_cv = goals_p.shape[0]
        else:
            g_cv = 10
        if shots_p.shape[0]<=10:
            continue
        else:
            s_cv = 10
        if 'goal_dist' not in db.players_year_20172018.find_one({'player_id':scorer[1]},{'goal_dist':1}).keys():
            density = make_shot_density(goals_p[['x','y']].values,g_cv)
            save_density_to_db('players_year_20172018',scorer[1],density,scorer[0])
        else:
            print(scorer[1], ' scorer goal dist exists')
        if 'shot_dist' not in db.players_year_20172018.find_one({'player_id':scorer[1]},{'shot_dist':1}).keys():
            density = make_shot_density(shots_p[['x','y']].values,s_cv)
            save_density_to_db('players_year_20172018',scorer[1],density,scorer[0])
        else:
            print(scorer[1], ' scorer shot dist exists')

def single_row(row,e_type):
    goalie = str(row['goalie'])
    scorer = str(row[e_type])
    x = row['x']
    y = row['y']+42
    g_density = 1-retrieve_density(goalie,'goalie',2017).reshape(85,100)
    p_density = retrieve_density(scorer,'player',2017).reshape(85,100)
    g_g_den = g_density[y][x]
    g_p_den = p_density[y][x]
    return np.append(row,[g_p_den,g_g_den])




if __name__ == '__main__':
    db = _init_mongo()
    shots, goals = load_shots_goals(2017)
    # goals = pd.read_csv('data/2017_goals.csv')
    # shots = pd.read_csv('data/2017_shots.csv')
    # nm_shots, nm_goals = player_shots_goals(8477492,shots,goals)
    # pr_shots, pr_goals = goalie_shots_goals(8471469,shots,goals)
    #
    # goal_d = make_shot_density(nm_goals[['x','y']].values)
    # save_d = 1 - make_shot_density(pr_goals[['x','y']].values)

    # feature = np.concatenate((goal_d.reshape(1,goal_d.shape[0]),save_d.reshape(1,save_d.shape[0])),axis=1)

    # generate_all_distributions(shots,goals)

    # model = Sequential()
    # model.add(Dense(32, input_dim=784))
    # model.add(Activation('relu'))
