from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from make_graphs import player_shots_goals, goalie_shots_goals, load_shots_goals, plot_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pymongo
from make_distributions import make_shot_density
import pickle as pkl

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def retrieve_team_density(db,team,year):
    coll = db['team']
    y = coll.find_one({'team':str(team)})['distribution'][0]['year_'+str(year)]
    return pkl.loads(y)

def retrieve_player_density(db,player,position,dist_type,year=2017):
    coll = db['players_year_'+str(year)+str(year+1)]
    if position == 'goalie':
        y = coll.find_one({'player_id':player})[dist_type][0]
    else:
        y = coll.find_one({'player_id':player})[dist_type][0]
    return pkl.loads(y)

def load_data(year):
    goals = pd.read_csv('data/'+str(year)+'_goals.csv')
    goals.x = goals.x.abs()
    shots = pd.read_csv('data/'+str(year)+'_shots.csv')
    shots.x = shots.x.abs()
    missed = pd.read_csv('data/'+str(year)+'_missed.csv')
    missed.x = missed.x.abs()
    return goals,shots,missed

def get_5_player_data(players,goals,shots,missed):
    goals_5 = goals[goals['scorer'].isin(players)]
    shots_5 = shots[shots['shooter'].isin(players)]
    missed_5 = missed[missed['shooter'].isin(players)]
    return goals_5,shots_5,missed_5

def get_player_id(player_name):
    db = _init_mongo()
    p_id = db['players'].find_one({'fullName':player_name},{'id':1})['id']
    return str(p_id)

def single_row(db,row,goalie,goal_den,shot_den,miss_den,d_team,year=2017):
    # goalie = str(int(row['goalie']))
    # scorer = str(int(row[p_type]))
    # team = str(int(row[d_team]))
    x = int(row['x'])
    y = int(row['y'])+42
    g_g_density = 1-retrieve_player_density(db,str(goalie),'goalie','save_dist').reshape(85,100) #goalie save
    t_b_density = retrieve_team_density(db,d_team,year)
    g_g_den = g_g_density[y][x]
    p_g_den = goal_den[y][x]
    p_s_den = shot_den[y][x]
    p_m_den = miss_den[y][x]
    t_b_den = t_b_density[y][x]
    return np.append(row,[p_g_den,g_g_den,p_s_den,p_m_den,t_b_den])

def generate_prediction_data(goal_den,shot_den,miss_den,goalie_id,d_team,scaler):
    xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
    xy = np.vstack([xx.ravel(),yy.ravel()])

    # shooter = int(shooter_id)
    goalie = int(goalie_id)
    # shooter = np.full((8500,1),shooter)
    # goalie = np.full((8500,1),goalie)

    # unseen = np.concatenate((shooter,goalie,xy.T),axis=1)
    # unseen = pd.DataFrame(unseen,columns=['scorer','goalie','x','y'])
    unseen = pd.DataFrame(xy.T,columns=['x','y'])

    unseen_data = []
    for row in unseen.iterrows():
        row_d = single_row(db,row[1],goalie,goal_den,shot_den,miss_den,d_team)
        unseen_data.append(row_d)
    unseen_data = np.array(unseen_data)
    # unseen_data_for_model = unseen_data[:,2:]
    return scaler.transform(unseen_data)

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
    td = np.genfromtxt('data/2017_g_s_m_b.csv',delimiter=',')
    x_std,x_t_std,y_train,y_test,x_scaler = scale_transform_split(td)

    goals,shots,missed = load_data(2017)
    lw = get_player_id('Gabriel Landeskog')
    c = get_player_id('Nathan MacKinnon')
    rw = get_player_id('Mikko Rantanen')
    d1 = get_player_id('Tyson Barrie')
    d2 = get_player_id('Nikita Zadorov')

    list_of_players = [lw,c,rw,d1,d2]

    g = get_player_id('Pekka Rinne')

    goals_5,shots_5,missed_5 = get_5_player_data(list_of_players,goals,shots,missed)

    den_5_g = make_shot_density(goals_5[['x','y']])
    den_5_s = make_shot_density(shots_5[['x','y']])
    den_5_m = make_shot_density(missed_5[['x','y']])

    avs_top_5 = generate_prediction_data(den_5_g,den_5_s,den_5_m,8471469,52,x_scaler)
