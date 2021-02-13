from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd

import os, fnmatch

def getData(ticker):

    # Tickers list

    today = date.today()
    # We can get data by our choice by giving days bracket
    start_date= str("2017") + "-" + str("01") + "-" + str("01")
    print(start_date)
    #end_date=”2019–11–30"
    files=[]

    print (ticker)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    dataname= ticker + "_" + str(today)
    files.append(dataname)
    SaveData(data, dataname)


# Create a data folder in your current dir.
def SaveData(df, filename):

    df.to_csv("./yfinance_data/" + filename + ".csv")


def get_yfinances_data(ticker_list):

    #This loop will iterate over ticker list, will pass one ticker to get data, and save that data as file.
    for tik in ticker_list:
        getData(tik)

    listOfFiles = os.listdir('./yfinance_data/')
    pattern = "*.csv"
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            print(entry)
            df1= pd.read_csv("./yfinance_data/" + entry)
            print (df1.head())

    return listOfFiles