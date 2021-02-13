
# from numpy import loadtxt
import numpy as np
import tensorflow as tf
import sys

from keras.optimizers import SGD
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense

from dataloader import *
from train_predict import *
from predictor_predict import *
from build_DT_df import *


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def clear_model_directory():

    listOfFilesToRemove = os.listdir('./yfinance_model/')
    pattern_pkl = "*.pkl"
    pattern_gz = "*.gz"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern_pkl):
            print("remove : ",entry)
            os.remove("./yfinance_model/" + entry)
        else:
            if fnmatch.fnmatch(entry, pattern_gz):
                print("remove : ", entry)
                os.remove("./yfinance_model/" + entry)


def clear_ydata_directory():

    listOfFilesToRemove = os.listdir('./yfinance_data/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            os.remove("./yfinance_data/" + entry)

def clear_output_directory():

    listOfFilesToRemove = os.listdir('./yfinance_output/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            os.remove("./yfinance_output/" + entry)

def clear_DT_data_directory():

    listOfFilesToRemove = os.listdir('./DT_finance_data/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            os.remove("./DT_finance_data/" + entry)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    DOWNLOAD_DATA = "DOWNLOAD"
    #DOWNLOAD_DATA = "DOWNLOAD"
    COMPUTE_MODEL = "COMPUTE_MODEL"
    #COMPUTE_MODEL = "COMPUTE_MODEL"
    #BUILD_DT_DATA = "NO_BUILD"
    BUILD_DT_DATA = "BUILD"

    # We can add and delete any ticker from the list to get desired ticker live data
    ticker_list = ["EOD", "LB", "EXPE", "PXD", "MCHP", "CRM", "AAPL" , "NRG", "HFC", "NOW"]

    # get finance data from yfinance
    if DOWNLOAD_DATA == "DOWNLOAD":
        clear_ydata_directory()
        data_file_list = get_yfinances_data(ticker_list)
    else:
        data_file_list = listOfFiles = os.listdir('./yfinance_data/')

    df_DT_performence = pd.DataFrame({"tick": [],
                                      "DT_%": []})

    if BUILD_DT_DATA == "BUILD":
        # DT: Decision Tree method
        clear_DT_data_directory()
        i = 0
        for entry in data_file_list:
            prefix = entry.split('_')
            print("prefix: ",prefix)
            print("tic: ", prefix[0])

            DT_perfo = main_decision_tree(prefix[0], entry)
            df_DT_performence.loc[i] = [prefix[0], DT_perfo]
            i = i + 1

        df_DT_performence.to_csv("./DT_finance_data/perfo_DT.csv")



    if (COMPUTE_MODEL == "COMPUTE_MODEL"):
        clear_model_directory()
        df_performence_model = pd.DataFrame({"tick": [],
                                             "RMSE": [],
                                             "MAPE": [],
                                             "TA":   []})  # TA: Trend Acuracy
    else:
        df_performence_model = pd.read_csv('./yfinance_output/perfo.csv')

    clear_output_directory()

    i = 0
    for entry in data_file_list:
        prefix = entry.split('_')
        print("prefix: ",prefix)
        print("tic: ", prefix[0])

        # Compute Model
        if (COMPUTE_MODEL == "COMPUTE_MODEL"):
             rmse, mape = main_train(prefix[0], entry)
             # Insert new row in dataframe
             df_performence_model.loc[i] = [prefix[0], rmse, mape, 0]
             i = i + 1

        TA = main_predictor(prefix[0], entry)
        # df_performence_model.loc[ [str(prefix[0])] , ["TA"] ] = TA
        df_performence_model.loc[ df_performence_model.tick == str(prefix[0]) , ["TA"]] = TA

    df_performence_model.to_csv("./yfinance_output/perfo.csv")

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/