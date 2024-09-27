"""

Mean-Variance portfolio optimization on S&P 500 index constituent stocks between 2015-2017. 
We will again be training the optimizer between 2015-2016, and testing it on between 2016 and 2017. 

We assume that stocks can only be long, and all portfolio should be allocated to stocks (We cannot hold cash). 
However, we will rebalance the portfolio every Monday in the test period rather than once in a sliding window-based approach.
"""

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

# Define the portfolio optimization function
def mean_variance_optimization(training_data, alpha):
    
    # Expected returns and covariance matrix
    expected_returns = training_data.mean()
    n = len(expected_returns)
    cov_matrix = training_data.cov()+ (np.finfo(np.float32).eps * np.identity(n))
        
    # weights = cp.Variable(n)  
    weights = cp.Variable(n, nonneg=True) 
    port_return = cp.matmul(weights.T, expected_returns.values)
    # port_return = weights @ expected_returns.values
    port_variance = cp.quad_form(weights, cov_matrix)

    objective = cp.Maximize(port_return - alpha * port_variance)
    constraint = [cp.sum(weights) == 1, weights >= 0]
    problem = cp.Problem(objective, constraint)
    problem.solve()

    # Extract optimal weights
    optimal_weights = weights.value

    return optimal_weights

from datetime import timedelta

def test_portfolio(weights, testing_data):
    port_returns = np.dot(testing_data, weights)
    return port_returns
    
def rolling_train_and_test(_all_times_data,
                           test_start_date, 
                           alpha,
                           rolling_window_days=5,
                           ):
    
    rolling_predictions =[]
    for train_end in _all_times_data[test_start_date:].index[::rolling_window_days]:
        train_on = _all_times_data[:train_end]
        # test_end = datetime.date.fromisoformat(train_end.strftime('%Y-%m-%d'))+timedelta(days=rolling_window_days)
        # test_on = _all_times_data[train_end:test_end.isoformat()]
        test_on = _all_times_data[train_end:]
        
        # Portfolio optimization for training period
        optimal_weights = mean_variance_optimization(train_on, alpha)
        
        # Calculate portfolio values for the testing period
        _port_performance = test_portfolio(optimal_weights, test_on)
    
        _port_returns = _port_performance[:rolling_window_days]        
        # append to rolling_predictions
        rolling_predictions.extend(_port_returns)
        # print(f"size: {_port_returns.shape},prf: {_port_performance.shape} , rolling: {len(rolling_predictions)}")
    rolling_predictions = np.array(rolling_predictions).flatten()
    
    formated_res = {
        "port_returns": rolling_predictions,
        "port_variance": np.var(rolling_predictions),
        "port_sharpe_ratio": rolling_predictions.mean() / np.var(rolling_predictions),
        "weights": optimal_weights
        }
    
    return formated_res

if __name__ == '__main__':
    
    # assuming that dat is already downloaded
    file_to_load = f"{DATA_DIR}/sp500_All.csv"
    data = pd.read_csv(file_to_load, parse_dates =True, index_col=0)
    
    # before preprocessing
    # print_df_information(data)

    # handle missing values
    data = handle_missing_values(data)

    # after preprocessing
    print_df_information(data)
    
    # since we will be working with returns instead
    returns_df = data.pct_change().dropna()

    # infer frequency
    returns_df.index.freq = pd.infer_freq(returns_df.index)
    returns_df.info()


    #train
    train_start_date='2015-01-01'
    train_end_date='2016-12-31'

    #test
    test_start_date='2017-01-01' 
    test_end_date='2017-12-31'


    training_data = returns_df.loc[:train_end_date]
    testing_data = returns_df.loc[test_start_date:]
    
    print(f"training_data: {training_data.shape} , testing_data: {testing_data.shape}")
    
    
    # Perform portfolio optimization with rolling windows
    # alpha_values = [.1]
    alpha_values = [0.1, 1.0, 5.0, 10.0]


    per_alpha_port_performance = {}
    for alpha in alpha_values:
        
        _port_returns = rolling_train_and_test(_all_times_data = returns_df,
                                                test_start_date = test_start_date, 
                                                alpha = alpha,
                                                rolling_window_days = 5
                                                )
        per_alpha_port_performance[f"{alpha}"] = _port_returns["port_returns"]
        
    # Convert portfolio values to DataFrame for easy plotting
    portfolio_df = pd.DataFrame(per_alpha_port_performance, index=testing_data.index)
    portfolio_df["EW_benchmark"] = testing_data.mean(axis=1)
    
    # Plot P&L curves
    portolio_cumprod = (portfolio_df + 1).cumprod()
    portolio_cumprod.plot(figsize=(10, 6))
    plt.title('Mean-Variance Portfolio- P&L curve')
    plt.xlabel('')
    plt.ylabel('Portfolio return')
    plt.show()
    
    ## compute infomation ratio and decide the best alpha
    alpha_cols = portfolio_df.columns.tolist()
    alpha_cols.remove("EW_benchmark")
    information_ratio = portfolio_df[alpha_cols].mean() / portfolio_df[alpha_cols].std()
    information_ratio = pd.DataFrame(information_ratio, columns=["information_ratio"], )
    information_ratio.sort_values(by="information_ratio", ascending=False, inplace=True)

    """

    looking at the information ratio we can say that alpha = 10 is the best, which implicates that
    it is has overal best return while minimizing the risk

    this also means that the stocks invested in are very risky therefore high

    """
    display(information_ratio)