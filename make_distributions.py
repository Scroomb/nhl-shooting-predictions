from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from make_graphs import player_shots_goals, goalie_shots_goals, load_shots_goals, plot_kde
import pymongo
import pickle as pkl
import bson
import triangle as t

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def make_shot_density(shots,cv=20):
    xx,yy = np.meshgrid(np.arange(0,100,1),np.arange(-42,43,1))
    xy = np.vstack([xx.ravel(),yy.ravel()])
    grid = GridSearchCV(KernelDensity(),{'bandwidth':np.linspace(0.1,20,500)},cv=cv,n_jobs=-1)
    grid.fit(shots)
    # print(grid.best_params_)
    return np.exp(grid.best_estimator_.score_samples(xy.T))

def save_density_to_db(year,player,density,dist_type,position,period,pp):
    coll = db[year]
    y = bson.binary.Binary(pkl.dumps(density,protocol=2))
    # if position == 'goalie':
    coll.update_one({'player_id':str(player)},{'$push':{'period.'+str(period)+'.'+pp+'.'+dist_type:y}})
    # else:
    #     coll.update_one({'player_id':player},{'$push':{dist_type:y}})

def save_block_to_db(year,team,density,db):
    coll = db['team']
    y = bson.binary.Binary(pkl.dumps(density,protocol=2))
    coll.update_one({'team':team},{'$push':{'distribution':{'year_'+year:y}}},upsert=True)

def retrieve_density(db,player,position,dist_type,year=2017):
    coll = db['players_year_'+str(year)+str(year+1)]
    if position == 'goalie':
        y = coll.find_one({'player_id':player})[dist_type][0]
    else:
        y = coll.find_one({'player_id':player})[dist_type][0]
    return pkl.loads(y)

def generate_missed_distributions(missed):
    shooters = []
    for x in missed.shooter.unique():
        shooters.append(('shooter',x))
    shooters = np.array(shooters)
    for shooter in shooters:
    # for i in range(647,721):
        # scorer = scorers[i]
        print(shooter[1])
        missed_p = missed[missed.shooter==int(shooter[1])]
        if missed_p.shape[0]<=10:
            continue
        else:
            m_cv = 10

        if 'missed_dist' not in db.players_year_20172018.find_one({'player_id':shooter[1]},{'missed_dist':1}).keys():
            density = make_shot_density(missed_p[['x','y']].values,m_cv)
            save_density_to_db('players_year_20172018',shooter[1],density,'missed_dist',shooter[0])
        else:
            print(shooter[1], ' shooter missed dist exists')

def generate_all_distributions(shots,goals,missed):
    goalies = set()
    for x in goals.goalie.unique():
        goalies.add(('goalie',x))
    # goalies = np.array(goalies)
    scorers = set()
    for x in goals.scorer.unique():
        scorers.add(('scorer',x))
    # scorers = np.array(scorers)
    for goalie in goalies:
        # print(goalie[1])
        # shots_g, goals_g = goalie_shots_goals(int(goalie[1]),shots,goals)
        g = goals[goals.goalie==goalie[1]]
        g.x = g.x.abs()
        for period in range(1,4):
            for pp in ['even','pp']:
                print(goalie[1],' ',period,' ',pp)
                goals_g = g[(g['period']==period)&(g['pp_status']==pp)]
                if goals_g.shape[0]<=20:
                    if goals_g.shape[0]<=1:
                        continue
                    g_cv = goals_g.shape[0]
                else:
                    g_cv = 20
                if 'save_dist' not in db.players_year_20172018.find_one({'player_id':str(goalie[1])},\
                                      {'period.'+str(period)+'.'+pp+'.save_dist':1}).keys():
                    density = make_shot_density(goals_g[['x','y']].values,g_cv)
                    save_density_to_db('players_year_20172018',goalie[1],density,'save_dist',goalie[0],period,pp)
                else:
                    print(goalie[1], ' goalie save dist exists')
        # if shots_g.shape[0]<=10:
        #     continue
        # else:
        #     s_cv = 10
        # if 'shot_dist' not in db.players_year_20172018.find_one({'player_id':goalie[1]},{'shot_dist':1}).keys():
        #     density = make_shot_density(shots_g[['x','y']].values,s_cv)
        #     save_density_to_db('players_year_20172018',goalie[1],density,'shot_dist',goalie[0])
        # else:
        #     print(goalie[1], ' goalie shot dist exists')

    for scorer in scorers:
        print(scorer[1])
        # shots_p, goals_p = player_shots_goals(int(scorer[1]),shots,goals)
        p_g = goals[goals.scorer==scorer[1]]
        p_s = shots[shots.shooter==scorer[1]]
        p_m = missed[missed.shooter==scorer[1]]
        p_g.x = p_g.x.abs()
        p_s.x = p_s.x.abs()
        p_m.x = p_m.x.abs()
        for period in range(1,4):
            for pp in ['even','pp']:
                goals_p = p_g[(p_g['period']==period)&(p_g['pp_status']==pp)]
                shots_p = p_s[(p_s['period']==period)&(p_s['pp_status']==pp)]
                missed_p = p_m[(p_m['period']==period)&(p_m['pp_status']==pp)]
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
                if missed_p.shape[0]<=10:
                    continue
                else:
                    m_cv = 10
                if 'goal_dist' not in db.players_year_20172018.find_one({'player_id':str(scorer[1])},\
                                      {'period.'+str(period)+'.'+pp+'.goal_dist':1}).keys():
                    density = make_shot_density(goals_p[['x','y']].values,g_cv)
                    save_density_to_db('players_year_20172018',scorer[1],density,'goal_dist',scorer[0],period,pp)
                else:
                    print(scorer[1], ' scorer goal dist exists')

                if 'shot_dist' not in db.players_year_20172018.find_one({'player_id':str(scorer[1])},\
                                      {'period.'+str(period)+'.'+pp+'.shot_dist':1}).keys():
                    density = make_shot_density(shots_p[['x','y']].values,s_cv)
                    save_density_to_db('players_year_20172018',scorer[1],density,'shot_dist',scorer[0],period,pp)
                else:
                    print(scorer[1], ' scorer shot dist exists')

                if 'missed_dist' not in db.players_year_20172018.find_one({'player_id':str(scorer[1])},\
                                      {'period.'+str(period)+'.'+pp+'.missed_dist':1}).keys():
                    density = make_shot_density(missed_p[['x','y']].values,m_cv)
                    save_density_to_db('players_year_20172018',shooter[1],density,'missed_dist',shooter[0],period,pp)
                else:
                    print(shooter[1], ' shooter missed dist exists')


def single_row(db,row,p_type):
    goalie = str(int(row['goalie']))
    x = int(row['x'])
    scorer = str(int(row[p_type]))
    y = int(row['y'])+42
    g_density = 1-retrieve_density(db,goalie,'goalie','save_dist').reshape(85,100)
    p_density = retrieve_density(db,scorer,'player','goal_dist').reshape(85,100)
    g_g_den = g_density[y][x]
    g_p_den = p_density[y][x]
    return np.append(row,[g_p_den,g_g_den])

def make_data(db,shots,goals,missed):
    coll = db['players_year_20172018']
    goal_data = []

    for row in goals.iterrows():
        if coll.find_one({'player_id':str(int(row[1]['scorer']))}) and coll.find_one({'player_id':str(int(row[1]['goalie']))}):
            shooter_goal = bool('goal_dist' in coll.find_one({'player_id':str(int(row[1]['scorer']))}).keys())
            goalie_save = bool('save_dist' in coll.find_one({'player_id':str(int(row[1]['goalie']))}).keys())
            if not(shooter_goal and goalie_save):
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
            if not(shooter_goal and goalie_save):
                continue
            row_data = single_row(db,row_d,'shooter')
            row_data = np.append(row_data,0)
            shot_data.append(row_data)
    shot_data = np.array(shot_data)
    sd = shot_data[:,2:]
    td = np.concatenate((gd,sd),axis=0)

    return td

def gen_block_dist(team,blocked):
    blk = blocked[blocked['d_team']==team][['x','y']].values
    blk_dist = make_shot_density(blk,20).reshape(100,85)
    # blk_dist = distribution
    out_blk_dist = np.zeros(blk_dist.shape)
    for x in range(100):
        print('Starting: ',x)
        for y in range(85):
            point = np.array([[x],[y]])
            # print(point)
            coords = t.in_triangle_coords(point)
            if coords is None:
                out_blk_dist[x][y] = 0.0
            else:
                block_val=0
                for coord in coords:
                    block_val += blk_dist[coord[0]][coord[1]]
                out_blk_dist[x][y]=block_val
        print('Completed: ',x)
    return out_blk_dist.T

def gen_all_blocks(year,blocks,db):
    for team in blocks['d_team'].unique():
        out_block = gen_block_dist(team,blocks)
        save_block_to_db(str(year),str(team),out_block,db)

def load_data(year):
    goals = pd.read_csv('data/'+str(year)+'_goals.csv')
    shots = pd.read_csv('data/'+str(year)+'_shots.csv')
    missed = pd.read_csv('data/'+str(year)+'_missed.csv')
    return goals,shots,missed

def get_5_player_data(rw,c,lw,d1,d2,goals,shots,missed):
    list_of_players = [rw,c,lw,d1,d2]
    goals_5 = goals[goals['scorer'].isin(list_of_players)]
    shots_5 = shots[shots['shooter'].isin(list_of_players)]
    missed_5 = missed[missed['shooter'].isin(list_of_players)]
    return goals_5,shots_5,missed_5

def get_player_id(player_name):
    db = _init_mongo()
    p_id = db['players'].find_one({'fullName':player_name},{'id':1})['id']
    return p_id

if __name__ == '__main__':
    db = _init_mongo()
    # blocks = pd.read_csv('data/2017_blocked.csv')
    goals,shots,missed = load_data(2017)
    # gen_all_blocks(2017,blocks,db)
    # lw = get_player_id('Gabriel Landeskog')
    # c = get_player_id('Nathan MacKinnon')
    # rw = get_player_id('Mikko Rantanen')
    # d1 = get_player_id('Tyson Barrie')
    # d2 = get_player_id('Nikita Zadorov')

    # generate_missed_distributions(missed)
    generate_all_distributions(shots,goals,missed)

#     shots, goals = load_shots_goals(2017)
#     # goals = pd.read_csv('data/2017_goals.csv')
#     # shots = pd.read_csv('data/2017_shots.csv')
#     nm_shots, nm_goals = player_shots_goals(8477492,shots,goals)
#     # pr_shots, pr_goals = goalie_shots_goals(8471469,shots,goals)
#     #
#     # goal_d = make_shot_density(nm_goals[['x','y']].values)
#     # save_d = 1 - make_shot_density(pr_goals[['x','y']].values)
#
#     # feature = np.concatenate((goal_d.reshape(1,goal_d.shape[0]),save_d.reshape(1,save_d.shape[0])),axis=1)
#
#     # generate_all_distributions(shots,goals)
#
#     td = make_data(db,shots,goals)
#     #
#     # td_x = td[:,:-1]
#     # td_y = td[:,-1]
#     #
#     # sm = SMOTE(kind='regular')
#     #
#     # x_res, y_res = sm.fit_sample(td_x,td_y)
#
#     x_train,x_test,y_train,y_test = train_test_split(x_res,y_res,test_size=.2)
#     x_scaler = StandardScaler()
#
#     x_std = x_scaler.fit_transform(x_train)
#     x_t_std = x_scaler.transform(x_test)
#
#     #
#
#
#     model.fit(x_std, y_train,
#               epochs=50,
#               batch_size=256)
#     score = model.evaluate(x_t_std, y_test, batch_size=256)
