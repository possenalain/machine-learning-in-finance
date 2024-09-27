from tools.helper_functions import *

import os
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cvx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import cvxpy as cp

DATA_DIR = os.getenv('FINANCE_DATA_DIR', '../../data')

if __name__ == '__main__':
    # get training and test data and save it in a csv file
    tickers = get_tickers(file_path=f'{DATA_DIR}/sp500tickers.txt',limit=-1)


    #training data
    training_data = pull_data(download=True, 
                                    tickers=tickers,
                                    start_date='2015-01-01', 
                                    end_date='2016-12-31',
                                    split = 'train',
                                    DATA_DIR=DATA_DIR)

    #testing data
    testing_data = pull_data(download=True,
                                tickers=tickers,
                                    start_date='2017-01-01', 
                                    end_date='2017-12-31',
                                    split = 'test',
                                    DATA_DIR=DATA_DIR)


    # clean tickers get new training and testing dfs

    tickers , training_data, testing_data = get_refined_data_and_tickers(
                                                                tickers=tickers,
                                                                training_data=training_data,
                                                                testing_data=testing_data)



    training_data = training_data[tickers].pct_change().dropna()
    testing_data = testing_data[tickers].pct_change().dropna()

    # check shapes 
    print(f"Training shape: {training_data.shape}\nTesting shape: {testing_data.shape} \nTickers len:{len(tickers)}")