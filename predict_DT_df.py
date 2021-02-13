import sys
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from train_predict import *


def main_DT_train(df):


    y = df.pop('trend')
    remove_date = df.pop('Date')
    X = df

    print('DT - Splitting data to train and test dataset...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5  )
     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print(" ")

    Classifier = DecisionTreeRegressor()
    Classifier.fit(X_train, y_train)

    Result_predicted = Classifier.predict(X_test)

    Result_predicted = Result_predicted.reshape(-1,1)

    #print(Result_predicted, "     ", y_test )

    result = pd.DataFrame(Result_predicted)

    # reset index
    result.reset_index(drop=True, inplace=True)

    expected_y = pd.DataFrame(y_test)

    # reset index
    expected_y.reset_index(drop=True, inplace=True)

    result_ok = 0
    len_y_test = len(y_test)
    for i in range(0, len_y_test, 1):
        if result.loc[i][0] == expected_y.loc[i,"trend"]:
            result_ok = result_ok + 1

    return round(result_ok/len_y_test * 100, 2)
