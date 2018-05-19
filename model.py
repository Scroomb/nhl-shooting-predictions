from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pymongo
from make_graphs import plot_kde
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def define_model(input_size,nuerons_layer_1=100,neurons_layer_2=100,neurons_layer_3=100,
                init_mode='normal',dropout_rate=0.5,activation='relu',final_activation='sigmoid',
                optimizer='adagrad'):
    model = Sequential()
    model.add(Dense(nuerons_layer_1, input_dim=input_size, activation='relu', kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons_layer_2, activation=activation, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    # model.add(Dense(neurons_layer_3, activation='relu', kernel_initializer=init_mode))
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation=final_activation, kernel_initializer=init_mode))

    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    return model

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

def generate_prediction_data(shooter_id,goalie_id,d_team,scaler):
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
        row_d = single_row(db,row[1],'scorer',d_team)
        unseen_data.append(row_d)
    unseen_data = np.array(unseen_data)
    unseen_data_for_model = unseen_data[:,2:]
    return scaler.transform(unseen_data_for_model)

def single_row(db,row,p_type,d_team,year=2017):
    goalie = str(int(row['goalie']))
    scorer = str(int(row[p_type]))
    # team = str(int(row[d_team]))
    x = int(row['x'])
    y = int(row['y'])+42
    g_g_density = 1-retrieve_player_density(db,goalie,'goalie','save_dist').reshape(85,100) #goalie save
    p_g_density = retrieve_player_density(db,scorer,'player','goal_dist').reshape(85,100) #player goal
    # g_s_density = retrieve_player_density(db,goalie,'goalie','shot_dist').reshape(85,100) #goalie
    p_s_density = retrieve_player_density(db,scorer,'player','shot_dist').reshape(85,100) #player shot
    p_m_density = retrieve_player_density(db,scorer,'player','missed_dist').reshape(85,100)
    t_b_density = retrieve_team_density(db,d_team,year)
    g_g_den = g_g_density[y][x]
    p_g_den = p_g_density[y][x]
    p_s_den = p_s_density[y][x]
    p_m_den = p_m_density[y][x]
    t_b_den = t_b_density[y][x]
    return np.append(row,[p_g_den,g_g_den,p_s_den,p_m_den,t_b_den])

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

if __name__ == '__main__':
    db = _init_mongo()
    td = np.genfromtxt('data/2017_g_s_m_b.csv',delimiter=',')
    # td = np.genfromtxt('data/2017_total_data.csv',delimiter=',')
    # td = np.genfromtxt('data/2017_shots_goals_goalie.csv',delimiter=',')

    x_std,x_t_std,y_train,y_test,x_scaler = scale_transform_split(td)
    model = define_model(7)
    model.fit(x_std,y_train,epochs=5)
    model.save('working_model.h5')

    # model = load_model('trained_model.h5')

    # model = KerasClassifier(build_fn=define_model,verbose=1,input_size=4,epochs=epochs,batch_size=batch_size)
    # model = KerasClassifier(build_fn=define_model,verbose=1,input_size=7,epochs=100,batch_size=512)
    # # param_grid = dict(nuerons_layer_1=[25,50,100,500],neurons_layer_2=[25,50,100,500])
    # # param_grid = dict(epochs = [10,25,50,100], batch_size=[32,128,512]) # epochs = 100, batch_size = 32
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    # droput_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # hidden_activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    # final_activation = ['tanh','sigmoid','hard_sigmoid','softmax']
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # param_grid = dict(activation=hidden_activation,final_activation=final_activation,optimizer=optimizer,dropout_rate=droput_grid,init_mode=init_mode)

    # NM vs PR
    # pred_data = generate_prediction_data(8477492,8471469,18,x_scaler)

    # GL vs PR
    # pred_data = generate_prediction_data(8477492,8471469,x_scaler)

    # early_stop = EarlyStopping(monitor='acc',min_delta=0.001)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True)
    # model.fit(x_std,y_train,epochs=100,batch_size=32,verbose=1)
    # print(model.evaluate(x_t_std,y_test,batch_size=32))
    # plot_kde(model.predict(pred_data))

    # players = [8476887, 8475793, 8478042, 8474600, 8475176]
    # goalie = 8473541
    #
    # ff_data = generate_prediction_data(8476887,goalie,x_scaler)
    # rj_data = generate_prediction_data(8475793,goalie,x_scaler)
    # va_data = generate_prediction_data(8478042,goalie,x_scaler)
    # josi_data = generate_prediction_data(8474600,goalie,x_scaler)
    # re_data = generate_prediction_data(8475176,goalie,x_scaler)
    #
    # plot_kde(model.predict(ff_data),'Filip Forsberg vs Jonathan Bernier','Goals','ff_vs_jb')
    # plot_kde(model.predict(rj_data),'Ryan Johansen vs Jonathan Bernier','Goals','rj_vs_jb')
    # plot_kde(model.predict(va_data),'Viktor Arvidsson vs Jonathan Bernier','Goals','va_vs_jb')
    # plot_kde(model.predict(josi_data),'Roman Josi vs Jonathan Bernier','Goals','josi_vs_jb')
    # plot_kde(model.predict(re_data),'Ryan Ellis vs Jonathan Bernier','Goals','re_vs_jb')

    # gs = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,verbose=1)
    # gs.fit(x_std,y_train)
    # print(gs.best_params_,gs.best_score_)
    # print(gs.best_estimator_.score(x_t_std,y_test))
    # plot_kde(model.predict(
