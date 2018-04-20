import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
from matplotlib.patches import Circle

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def get_scoring_coord(coll):
    scoring_coords = pd.DataFrame()
    for gmid in coll.distinct('gamePk'):
        scoring = coll.find_one({'gamePk':gmid})['liveData']['plays']['scoringPlays']
        plays = coll.find_one({'gamePk':gmid})['liveData']['plays']['allPlays']
        coords = [plays[i]['coordinates'] for i in scoring if plays[i]['result']['emptyNet']==False]
        if coords:
            scoring_coords = scoring_coords.append(coords)
    return scoring_coords

def get_shot_coords(coll):
    shots_df = pd.DataFrame()
    for gmid in coll.distinct('gamePk'):
        all_plays = coll.find_one({'gamePk':gmid})['liveData']['plays']['allPlays']
        shots = [play['coordinates'] for play in all_plays if play['result']['eventTypeId']=='SHOT']
        if shots:
            shots_df = shots_df.append(shots)
    return shots_df

def get_scorer_coords(db,coll):
    scorers = pd.DataFrame()
    for gmid in coll.distinct('gamePk'):
        scoring = coll.find_one({'gamePk':gmid})['liveData']['plays']['scoringPlays']
        plays = coll.find_one({'gamePk':gmid})['liveData']['plays']['allPlays']
        for i in scoring:
            if plays[i]['about']['periodType']!='SHOOTOUT':
                if plays[i]['result']['emptyNet']==False and bool(plays[i]['coordinates']):
                    coords = plays[i]['coordinates']
                    for player in plays[i]['players']:
                        if player['playerType']=='Scorer' and coll.find_one({'gamePk':gmid})['gameData']['players']['ID' + str(player['player']['id'])]['primaryPosition']['code']!='G':
                            coords['scorer']=player['player']['id']
                        if player['playerType']=='Goalie':
                            coords['goalie']=player['player']['id']
                    if all(pos in coords.keys() for pos in ['goalie','scorer','x','y']):
                        scorers = scorers.append([coords])
    return scorers

def get_shooter_coords(db,coll):
    shooters = pd.DataFrame()
    for gmid in coll.distinct('gamePk'):
        all_plays = coll.find_one({'gamePk':gmid})['liveData']['plays']['allPlays']
        for play in all_plays:
            if play['about']['periodType']!='SHOOTOUT':
                if play['result']['eventTypeId']=='SHOT' and bool(play['coordinates']):
                    coords = play['coordinates']
                    for player in play['players']:
                        if player['playerType']=='Shooter' and db.players.find_one({'id':player['player']['id']})['primaryPosition']['code']!='G':
                            coords['shooter']=int(player['player']['id'])
                        if player['playerType']=='Goalie':
                            coords['goalie']=int(player['player']['id'])
                    if all(pos in coords.keys() for pos in ['goalie','shooter','x','y']):
                        shooters = shooters.append([coords])

    return shooters


def plot_heat_map(df,color='bk'):
    x = df.x.abs().values
    y = df.y.values
    cmap = plt.cm.jet
    cmap.set_under('w',1)
    plt.hist2d(x[~np.isnan(x)],y[~np.isnan(y)],bins=[100,87],cmap=cmap,range=[[0,99],[-42,42]],vmin=1)
    plt.colorbar()
    # plt.show()

def get_single_player(id):
    shots_p = shots[shots.shooter == id][['x','y']]
    shots_p.x = shots_p.x.abs()
    goals_p = goals[goals.scorer == id][['x','y']]
    goals_p.x = goals_p.x.abs()
    return shots_p,goals_p

def make_density_plot(shots,goals):
    xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
    xy = np.vstack([xx.ravel(),yy.ravel()])
    k = kde.gaussian_kde(shots.T,bw_method='silverman')
    zk = k(xy)
    plt.pcolormesh(xx,yy,zk.reshape(xx.shape),cmap=plt.cm.jet)
    # plt.scatter(goals[:,0],goals[:,1],c='k')
    plt.show()

def get_player_stats(coll):
    stats = pd.DataFrame()
    for player in coll.find():
        print(player['player_id'])
        id_yr = dict([('player_id',player['player_id']),('season',player['stats'][0]['season'])])
        stats = stats.append([{**id_yr,**player['stats'][0]['stat']}])
    return stats

def make_shooting_pct(shots,goals):
    xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
    xy = np.vstack([xx.ravel(),yy.ravel()])
    all_shots = np.concatenate([shots,goals])
    shot_counts = np.unique(shots,return_counts=True,axis=0)
    goal_counts = np.unique(goals,return_counts=True,axis=0)
    all_counts = np.unique(all_shots,return_counts=True,axis=0)

    shot_pct=np.zeros(xx.shape)
    for i in zip(xx,yy):
        for j in range(len(i[0])):
             arr = np.array([i[0][j],np.unique(i[1])[0]])
             if np.any(np.equal(arr,goal_counts[0]).all(axis=1)):
                 loc_g = np.where(np.equal(arr,goal_counts[0]).all(axis=1))
                 loc_s = np.where(np.equal(arr,all_counts[0]).all(axis=1))
                 pct = goal_counts[1][loc_g]/all_counts[1][loc_s]
                 shot_pct[int(np.unique(i[1]))-42][j]=pct
             else:
                 shot_pct[int(np.unique(i[1]))-42][j]=0
    return shot_pct
#8477492

def player_shots_goals(p_id,shots,goals):
    p_shots = shots[shots.shooter==p_id]
    p_goals = goals[goals.scorer==p_id]
    return p_shots,p_goals

def goalie_shots_goals(p_id,shots,goals):
    g_shots = shots[shots.goalie==p_id]
    g_goals = goals[goals.goalie==p_id]
    return g_shots,g_goals

def load_shots_goals(year):
    shots = pd.read_csv('data/'+str(year)+'_shots.csv')
    goals = pd.read_csv('data/'+str(year)+'_goals.csv')
    shots.drop('Unnamed: 0',axis=1,inplace=True)
    goals.drop('Unnamed: 0',axis=1,inplace=True)
    shots.x = shots.x.abs()
    goals.x = goals.x.abs()
    return shots,goals

def plot_kde(density,player='test',type='test',save='test_vs_test'):
    xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
    xy = np.vstack([xx.ravel(),yy.ravel()])
    plt.pcolormesh(xx,yy,density.reshape(xx.shape),cmap=plt.cm.jet)
    plt.title(player + ' ' + type)
    plt.xlabel('X-coordinate 1-ft')
    plt.ylabel('Y-coordinate 1-ft')
    plt.axvline(x=25,c='k')
    plt.axvline(x=89,c='k')
    circle1 = plt.Circle((69,-22),radius=15,clip_on=False, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=(0, 0, 0, .0001))
    circle2 = plt.Circle((69,22),radius=15,clip_on=False, zorder=10, linewidth=1,
                    edgecolor='black', facecolor=(0, 0, 0, .0001))
    goal = plt.Rectangle((89,-3),2,6,edgecolor='black', facecolor=(0,0,0,.0001))
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(goal)
    plt.savefig('figs/'+save+'.png')
    print(save, ' saved')

if __name__ == '__main__':
    # db = _init_mongo()
    # coll = db.year_2017
    # coll = db.players_year_20172018
    # shots = get_shooter_coords(db,coll)
    # shots = pd.read_csv('data/2017_shooters.csv')
    # goals = get_scorer_coords(db,coll)

    # ham_shot, ham_goal = get_single_player(8477504)
    # make_density_plot(ham_shot,ham_goal)
    # df_2010_scoring = get_scoring_coord(coll)
    # plot_heat_map(df_2010_scoring)
    # df_2010_shots = get_shot_coords(coll)

    shots, goals = load_shots_goals(2017)
    #
    # # # Nathan MacKinnon 8477492
    # nm_shots, nm_goals = player_shots_goals(8477492,shots,goals)
    # # nm_pct = make_shooting_pct(nm_shots[['x','y']],nm_goals[['x','y']])
    #
    # # Pekka Rinne 8471469
    # pr_goals = goals[goals.goalie==8471469]
