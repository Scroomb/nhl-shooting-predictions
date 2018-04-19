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
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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
                    kernel_initializer='glorot_uniform',
                    activation='relu')) # is tanh the best activation to use here?
    # maybe add another dense layer here?  How about some deep learning?!?
    model.add(Dense(units=num_neurons_in_layer_2,
                    input_dim=num_inputs,
                    kernel_initializer='glorot_uniform',
                    activation='relu'))
    model.add(Dense(units=num_classes,
                    input_dim=num_neurons_in_layer_3,
                    kernel_initializer='glorot_uniform',
                    activation='softmax')) # keep softmax as last layer
    sgd = SGD(lr=lr, decay=1e-7, momentum=.9) # using stochastic gradient descent (keep)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] ) # (keep)
    return model

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
                  metrics=[f1])
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
    model = define_model(4)
    # model = KerasClassifier(build_fn=define_model,verbose=1,input_size=4,epochs=100,batch_size=32)
    # param_grid = dict(nuerons_layer_1=[25,50,100,500],neurons_layer_2=[25,50,100,500])
    # param_grid = dict(epochs = [10,25,50,100], batch_size=[32,128,512]) # epochs = 100, batch_size = 32
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    # droput_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # hidden_activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    # final_activation = ['tanh','sigmoid','hard_sigmoid','softmax']
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # param_grid = dict(activation=hidden_activation,final_activation=final_activation,optimizer=optimizer)

    # NM vs PR
    # pred_data = generate_prediction_data(8477492,8471469,x_scaler)

    # GL vs PR
    # pred_data = generate_prediction_data(8476455,8471469,x_scaler)

    # early_stop = EarlyStopping(monitor='acc',min_delta=0.001)
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,write_graph=True)
    model.fit(x_std,y_train,epochs=100,batch_size=32,verbose=1)
    print(model.evaluate(x_t_std,y_test,batch_size=32))

    # gs = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,verbose=1,scoring=['acc','f1_score'])
    # gs.fit(x_std,y_train)
    # print(gs.best_params_,gs.best_score_)
    # print(gs.best_estimator_.score(x_t_std,y_test))
    # plot_kde(model.predict(
