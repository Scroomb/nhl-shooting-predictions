import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.opponent_name = None
        self.opponent_goalie = None
        self.opponent_goalie_id = None
        self.opponent_dist = None
        self.opponent_goalie_dist = None
        self.shot = None
        self.miss = None
        self.goal = None
        self.shot_den = None
        self.miss_den = None
        self.goal_den = None
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

    def populate_opponent(self,opponent,opponent_goalie,opponent_name):
        '''
        Populate opposing team/goalie data
        '''
        self.opponent_name = opponent_name
        self.opponent = opponent
        self.opponent_goalie = opponent_goalie
        self.opponent_goalie_id = self.get_player_id(opponent_goalie)
        for g in self.opponent_goalie_id:
            self.opponent_goalie_id = g
        self.opponent_dist = self.retrieve_team_density(opponent)
        self.opponent_goalie_dist = 1-self.retrieve_player_density('save_dist')

    def get_player_id(self,players):
        p_id = [self.db['players'].find_one({'fullName':player},{'id':1})['id'] \
          for player in players]
        # p_id = self.db['players'].find_one({'fullName':player},{'id':1})['id']
        return p_id

    def get_player_data(self,shots,goals,missed):
        self.shot = shots[shots['shooter'].isin(self.player_id)]
        self.miss = missed[missed['shooter'].isin(self.player_id)]
        self.goal = goals[goals['scorer'].isin(self.player_id)]
        # self.gen_player_densities()
        self.populate_full_densities()

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

    def retrieve_player_density(self,dist_type):
        coll = self.db['players_year_'+str(self.year)+str(self.year+1)]
        # y = coll.find_one({'player_id':player,'period.'+period+'.'+pp+'.'+ \
                # dist_type:{$exists:1}},{'period.'+period+'.'+pp+'.'+dist_type:1})
        if dist_type == 'save_dist':
            y = coll.find_one({'player_id':str(self.opponent_goalie_id)})[dist_type][0]
        else:
            for p in self.player_id:
                y = coll.find_one({'player_id':str(p)})[dist_type][0]
        return pkl.loads(y).reshape(85,100)

    def populate_full_densities(self):
        self.shot_den = self.retrieve_player_density('shot_dist')
        self.miss_den = self.retrieve_player_density('missed_dist')
        self.goal_den = self.retrieve_player_density('goal_dist')

    def single_row(self,row):
        x = int(row['x'])
        y = int(row['y'])+42
        g_g_den = self.opponent_goalie_dist[y][x]
        p_g_den = self.goal_den[y][x]
        p_s_den = self.shot_den[y][x]
        p_m_den = self.miss_den[y][x]
        t_b_den = self.opponent_dist[y][x]
        return np.append(row,[p_g_den,g_g_den,p_s_den,p_m_den,t_b_den])

    def generate_prediction_data(self,period=1):
        xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
        xy = np.vstack([xx.ravel(),yy.ravel()])
        goalie = int(self.opponent_goalie_id)
        unseen = pd.DataFrame(xy.T,columns=['x','y'])
        unseen_data = []
        for row in unseen.iterrows():
            row_d = self.single_row(row[1])
            unseen_data.append(row_d)
        unseen_data = np.array(unseen_data)
        return self.scaler.transform(unseen_data)

    def generate_predictions(self,dist_types='full'):
        pred_data = self.generate_prediction_data()
        pred = self.model.predict(pred_data)
        self.plot_kde(pred,self.player[0],'vs '+self.opponent_goalie[0]+' Full','player_pred',True)

    def plot_player_densities(self):
        self.plot_kde(self.shot_den,self.player[0],'Shots','player_shots',True)
        self.plot_kde(self.miss_den,self.player[0],'Misses','player_misses',True)
        self.plot_kde(self.goal_den,self.player[0],'Goals','player_goals',True)
        self.plot_kde(self.opponent_dist,str(self.opponent_name),'Blocks','opponent_blocks',True)
        self.plot_kde(self.opponent_goalie_dist,self.opponent_goalie[0],'Saves','opp_goalie',True)

    def plot_kde(self,density,player='test',type='test',save='test_vs_test',save_file=False):
        xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
        xy = np.vstack([xx.ravel(),yy.ravel()])
        z = density.reshape(xx.shape)
        plt.pcolormesh(xx,yy,z,cmap=plt.cm.jet,shading='gouraud')
        plt.title(player + ' ' + type)
        plt.xlabel('X-coordinate 1-ft')
        plt.ylabel('Y-coordinate 1-ft')
        plt.axvline(x=25,c='k')
        plt.axvline(x=89,c='k')
        circle1 = plt.Circle((69,-22),radius=15,clip_on=False, zorder=10, linewidth=1,
                        edgecolor='black', facecolor=(0, 0, 0, .0001))
        circle2 = plt.Circle((69,22),radius=15,clip_on=False, zorder=10, linewidth=1,
                        edgecolor='black', facecolor=(0, 0, 0, .0001))
        goal = plt.Rectangle((89,-3),44/12,6,edgecolor='black', facecolor=(0,0,0,.0001))
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(goal)
        if save_file:
            plt.savefig('static/figs/'+save+'.png')
            print(save, ' saved')
        else:
            plt.show()

if __name__ == '__main__':
    from make_distributions import load_data
    goals,shots,missed = load_data(2017)

    sp = SinglePlayer(['Nathan MacKinnon'],2017)
    sp.populate_opponent(52,['Pekka Rinne'])
    sp.get_player_data(shots,goals,missed)
    sp.generate_predictions()
