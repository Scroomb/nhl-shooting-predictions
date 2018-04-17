from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

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

if __name__ == '__main__':
