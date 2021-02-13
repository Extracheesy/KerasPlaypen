# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# from numpy import loadtxt
import numpy as np
import tensorflow as tf

from keras.optimizers import SGD
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense

from dataloader import *
from train import *
from predictor import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def first_test():

    # load the dataset
    dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))

    # make probability predictions with the model
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]

    # make class predictions with the model
    predictions = model.predict_classes(X)

    # summarize the first 5 cases
    total_runs = 700
    count_ok = 0
    for i in range(total_runs):
        if predictions[i] == y[i] :
            count_ok = count_ok + 1
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

    print("% success: ", count_ok, " out of: ", total_runs, " %: ", round(100 * count_ok / total_runs, 2))


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':

    #first_test()

    # main_train()

    #test_yfinances()


    # main_predictor()

    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
