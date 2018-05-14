import numpy as np
import pandas as pd
from make_graphs import player_shots_goals, goalie_shots_goals, load_shots_goals, plot_kde
import pymongo
from make_distributions import make_shot_density

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

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

def single_row(db,row,goal,miss,save):
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

if __name__ == '__main__':
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
