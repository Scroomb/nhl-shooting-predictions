from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pymongo
from make_graphs import plot_kde
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def _init_mongo():
    client = pymongo.MongoClient()
    db = client.hockey
    return db

def define_nn_mlp_model(num_neurons_in_layer_1 = 20,num_neurons_in_layer_2 = 20,num_neurons_in_layer_3=20,lr=0.001):
    model = Sequential()
    num_inputs = X_train.shape[1]
    num_classes = y_train_ohe.shape[1]
    model.add(Dense(units=num_neurons_in_layer_1,
                    input_dim=num_inputs,
                    kernel_initializer='uniform',
                    activation='relu')) # is tanh the best activation to use here?
    # maybe add another dense layer here?  How about some deep learning?!?
    model.add(Dense(units=num_neurons_in_layer_2,
                    input_dim=num_inputs,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dense(units=num_classes,
                    input_dim=num_neurons_in_layer_3,
                    kernel_initializer='uniform',
                    activation='softmax')) # keep softmax as last layer
    sgd = SGD(lr=lr, decay=1e-7, momentum=.9) # using stochastic gradient descent (keep)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] ) # (keep)
    return model

def define_model(input_size,nuerons_layer_1=100,neurons_layer_2=100,neurons_layer_3):
    model = Sequential()
    model.add(Dense(nuerons_layer_1, input_dim=input_size, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(neurons_layer_2, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(neurons_layer_3, activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh', kernel_initializer='uniform'))

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
    # p_s_den = p_s_density[y][x]  # Use player shot density?
    return np.append(row,[p_g_den,g_g_den])

def retrieve_density(db,player,position,dist_type,year=2017):
    coll = db['players_year_'+str(year)+str(year+1)]
    if position == 'goalie':
        y = coll.find_one({'player_id':player})[dist_type][0]
    else:
        y = coll.find_one({'player_id':player})[dist_type][0]
    return pkl.loads(y)

if __name__ == '__main__':
    db = _init_mongo()
    td = np.genfromtxt('data/2017_total_data.csv',delimiter=',')
    # td = np.genfromtxt('data/2017_shots_goals_goalie.csv',delimiter=',')

    x_std,x_t_std,y_train,y_test,x_scaler = scale_transform_split(td)
    model = define_model(4,500,500,100)

    # model = KerasClassifier(build_fn=define_model,verbose=1,epochs=50,batch_size=512,input_size=4)
    # param_grid = dict(nuerons_layer_1=[25,50,100,500],neurons_layer_2=[25,50,100,500])

    # pred_data = generate_prediction_data(8477492,8471469,x_scaler)

    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True)
    model.fit(x_std,y_train,epochs=50,batch_size=512)
    print(model.evaluate(x_t_std,y_test,batch_size=512))

    # gs = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,verbose=1)
    # gs.fit(x_std,y_train)
    print(gs.best_params_,gs.best_score_)
    print(gs.best_estimator_.score(x_t_std,y_test))
