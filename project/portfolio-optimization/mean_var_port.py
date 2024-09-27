"""
A simple Mean-Variance portfolio optimization on S&P 500 index constituent stocks between 2015-2017. 
We will simply be training the optimizer between 2015-2016, and testing it on between 2016 and 2017. 

We assume that stocks can only be long, and all portfolio should be allocated to stocks (We cannot hold cash).

portfolio optimization for range of alphas: [0.1, 1.0, 5.0, 10.0].

Information Ratio is also discussed. 
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

def mean_variance_optimization(training_data, alpha):
    
    # Expected returns and covariance matrix
    expected_returns = training_data.mean()
    n = len(expected_returns)
    cov_matrix = training_data.cov()
    # reg_term = 1e-5 * np.identity(n)
    # cov_matrix = cov_matrix + reg_term

    # weights = cp.Variable(n)  
    weights = cp.Variable(n, nonneg=True) 
    # port_return = (expected_returns.values) @ weights
    # port_variance = cp.quad_form(weights, cov_matrix)
    
    
    port_return = cp.matmul(weights.T, expected_returns.values)
    # port_return = weights @ expected_returns.values
    port_variance = cp.quad_form(weights, cov_matrix)

    # Objective function: Maximize the Sharpe ratio
    objective = cp.Maximize(port_return - alpha * port_variance)

    # Constraint: Sum of weights equals 1 (fully invested)
    constraint = [cp.sum(weights) == 1, weights >= 0]

    # Create and solve the problem
    problem = cp.Problem(objective, constraint)
    problem.solve()

    # Extract optimal weights
    optimal_weights = weights.value

    return optimal_weights


if __name__ == '__main__':
    
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
    
    # Perform portfolio optimization for different alpha values
    alpha_values = [0.1, 1.0, 5.0, 10.0]
    number_of_stocks = len(tickers)

    portfolio_values = []

    for alpha in alpha_values:
        # Portfolio optimization for training period
        optimal_weights = mean_variance_optimization(training_data, alpha)

        # Calculate portfolio values for the testing period 
        port_values = np.dot(testing_data, optimal_weights)
        portfolio_values.append(port_values)
        
        # print(f"alpha: {alpha}")
        # print(f"weights shape: { np.array(optimal_weights).shape}")
        # print(f"portfolio values shape: {np.array(port_values).shape}")

    # Convert portfolio values to DataFrame for easy plotting
    portfolio_df = pd.DataFrame(np.array(portfolio_values).T, 
                                columns=[f'a = {alpha}' for alpha in alpha_values], 
                                index=testing_data.index
                                )


    # Plot P&L curves
    portolio_cumprod = (portfolio_df + 1).cumprod()
    portolio_cumprod.plot(figsize=(10, 6))
    plt.title('Mean-Variance Portfolio- P&L curve')
    plt.xlabel('')
    plt.ylabel('Portfolio return')
    plt.show()


    # Calculate Information Ratios based on daily returns for the testing period
    # for simplicity we assume risk-free rate = 0
    risk_free_rate = 0
    portolio_cumprod = (portfolio_df + 1).cumprod()
    information_ratios = ((portolio_cumprod-1).mean()) / (portolio_cumprod-1).std()

    # Print Information Ratios
    print('\nInformation Ratios:')
    print(information_ratios)

    # Identify the alpha value that performs the best
    best_alpha = information_ratios.idxmax()
    print('\nBest Alpha Value:', best_alpha)