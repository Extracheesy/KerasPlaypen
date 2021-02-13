import sys
import numpy as np
import pandas as pd
import pickle
import joblib

# import function defined earlier in train.py
from numpy.core._multiarray_umath import ndarray

from train import load_process_data

# slightly modified from get_pred_closing_price() from train.py
def pred_closing_price(df, scaler, model):
    """
    Predict stock price using past 60 stock prices

    INPUT:
    df - dataframe that has been preprocessed
    scaler - instantiated object for MixMaxScaler()
    model - fitted model

    OUTPUT:
    predicted_price - predicted closing price
    """
    inputs = df['Close'][len(df) - 278 - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test.append(inputs[-60:,0])

    X_test = np.array(X_test)

    X_test: ndarray = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    predicted_price = float(closing_price[-1])

    print("closing price: ", closing_price)

    return predicted_price

def main_predictor(tick, tick_data_file):

    model_filepath = "yfinance_model/model_" + tick + ".pkl"
    scaler_filepath = "yfinance_model/scaler_" + tick + ".gz"
    data_filepath = "yfinance_data/" + tick_data_file
    output_filepath = "yfinance_output/" + "output" + tick_data_file

    # load pkl model file
    print('Loading model...')
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)
    print(" ")

    print('Loading data...')
    df = load_process_data(data_filepath)
    print(" ")

    len_df = len(df)

    #df.insert(df.shape[1],"prediction",0)
    df.insert(7, "prediction", 0)
    df.insert(8, "trend", 0)
    df.insert(9, "predicted_trend", 0)
    df.insert(10, "trend_pred_vs_actual", 0)

    print('Loading scaler file...')
    my_scaler = joblib.load(scaler_filepath)
    print(" ")

    for i in range(0, 60, 1):
        # filter datas
        df_filtered = df[ (df.index < (len_df - 60 + i) )]

        print('Predicting closing stock price...')
        predicted_price = pred_closing_price(df_filtered, my_scaler, model)
        print(" ")

        print('Predicted price: '+'$ '+str("{:.2f}".format(predicted_price)))
        print(" ")

        df.loc[(len_df - 60 + i), "prediction"] = predicted_price
        df.loc[(len_df - 60 + i), "trend"] = df.loc[(len_df - 60 + i), "Close"] - df.loc[(len_df - 60 + i - 1), "Close"]
        df.loc[(len_df - 60 + i), "predicted_trend"] = predicted_price  - df.loc[(len_df - 60 + i - 1), "Close"]
        df.loc[(len_df - 60 + i), "trend_pred_vs_actual"] = df.loc[(len_df - 60 + i), "trend"] * df.loc[(len_df - 60 + i), "predicted_trend"]
        if df.loc[(len_df - 60 + i), "trend_pred_vs_actual"] < 0:
            df.loc[(len_df - 60 + i), "trend_pred_vs_actual"] = -1
        else:
            df.loc[(len_df - 60 + i), "trend_pred_vs_actual"] = 1


    df.to_csv(output_filepath)

    return (df.trend_pred_vs_actual.sum() + 30) * 100 / 60

# if __name__ == '__main__':
#    main()
