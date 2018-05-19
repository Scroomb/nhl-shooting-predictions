import numpy as np
import pandas as pd
from make_graphs import plot_kde
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import pymongo
from keras.models import load_model
import pickle as pkl
from sklearn.externals import joblib

class SinglePlayer(object):
    '''
    Single Player Game Plan

    Initialize a player with player name and year
    Run populate_opponent with opposing team ID and opposing goalie
    Run get_player_data with full shots,goals,missed
    '''
    def __init__(self,player,year,pp='even'):
        self.year = year
        self.player = player
        self.db = self._init_mongo()
        self.player_id = self.get_player_id(player)
        self.opponent = None
        self.opponent_goalie = None
        self.opponent_goalie_id = None
        self.opponent_dist = None
        self.opponent_goalie_dist = None
        self.shot = None
        self.miss = None
        self.goal = None
        # self.shot_den = None
        # self.miss_den = None
        # self.goal_den = None
        # self.pred_data = None
        self.pp = pp

        self.model = load_model('data/test_model_515.h5')
        self.scaler = joblib.load('data/standard_scaler.pkl')

    def _init_mongo(self):
        '''
        Initialize mongoDB client
        '''
        client = pymongo.MongoClient()
        db = client.hockey
        return db

    def retrieve_team_density(self,team):
        '''
        Retrieve block distribution for a team
        '''
        coll = self.db['team']
        y = coll.find_one({'team':str(team)})['distribution'][0]['year_'+str(self.year)]
        return pkl.loads(y)

    def populate_opponent(self,opponent,opponent_goalie):
        '''
        Populate opposing team/goalie data
        '''
        self.opponent = opponent
        self.opponent_goalie = opponent_goalie
        self.opponent_goalie_id = self.get_player_id(opponent_goalie)
        self.opponent_dist = self.retrieve_team_density(opponent)
        self.opponent_goalie_dist = 1-self.retrieve_player_density( \
         str(self.opponent_goalie_id),'goalie','save_dist').reshape(85,100)

    def get_player_id(self,players):
        p_id = {self.db['players'].find_one({'fullName':player},{'id':1})['id'] \
          for player in players}
        return p_id

    def get_player_data(self,shots,goals,missed):
        self.shot = shots[shots['shooter'].isin(self.player_id)]
        self.miss = missed[missed['shooter'].isin(self.player_id)]
        self.goal = goals[goals['scorer'].isin(self.player_id)]
        self.gen_player_densities()

    def gen_player_densities(self,period):
        shot = self.shot[(self.shot['period']==period)& \
               (self.shot['pp_status']==self.pp)][['x','y']]
        miss = self.miss[(self.miss['period']==period)& \
               (self.miss['pp_status']==self.pp)][['x','y']]
        goal = self.goal[(self.goal['period']==period)& \
               (self.goal['pp_status']==self.pp)][['x','y']]

        self.shot_den = self.make_density(shot[['x','y']])
        self.miss_den = self.make_density(miss[['x','y']])
        self.goal_den = self.make_density(goal[['x','y']])

    def make_density(self,df,cv=10):
        xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
        xy = np.vstack([xx.ravel(),yy.ravel()])
        grid = GridSearchCV(KernelDensity(),{'bandwidth':np.linspace(0.1,20,500)}, \
         cv=cv,n_jobs=-1,verbose=1)
        grid.fit(df)
        return np.exp(grid.best_estimator_.score_samples(xy.T))

    def retrieve_player_density(self,player,position,dist_type):
        coll = self.db['players_year_'+str(self.year)+str(self.year+1)]
        if position == 'goalie':
            y = coll.find_one({'player_id':player})[dist_type][0]
        else:
            y = coll.find_one({'player_id':player})[dist_type][0]
        return pkl.loads(y)

    def single_row(self,row):
        x = int(row['x'])
        y = int(row['y'])+42
        g_g_den = self.opponent_goalie_dist[y][x]
        p_g_den = self.goal_den[y][x]
        p_s_den = self.shot_den[y][x]
        p_m_den = self.miss_den[y][x]
        t_b_den = self.opponent_dist[y][x]
        return np.append(row,[p_g_den,g_g_den,p_s_den,p_m_den,t_b_den])

    def generate_prediction_data(self,period):
        xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
        xy = np.vstack([xx.ravel(),yy.ravel()])
        goalie = int(self.opponent_goalie_id)
        unseen = pd.DataFrame(xy.T,columns=['x','y'])
        unseen_data = []
        for row in unseen.iterrows():
            row_d = single_row(db,row[1],goalie,self.goal_den,self.shot_den, \
                    self.miss_den,self.opposing_team)
            unseen_data.append(row_d)
        unseen_data = np.array(unseen_data)
        return self.scaler.transform(unseen_data)

    def generate_predictions(self):