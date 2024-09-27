"""
Mean Variance Portfolio Optimization with leverage
leverages: [1, 2, 4].
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
def mean_variance_optimization_leverage(training_data,
                                        alpha, 
                                        leverage_limit):
    n = len(training_data.columns)
    expected_returns = training_data.mean()
    cov_matrix = training_data.cov()

    # Portfolio weights as variables
    weights = cp.Variable(n)

    # Objective function - Maximize the Sharpe ratio
    port_return = cp.matmul(weights.T, expected_returns.values)
    port_variance = cp.quad_form(weights, cov_matrix)
    
    # Objective function: Maximize the Sharpe ratio
    objective = cp.Maximize(port_return - alpha * port_variance)
    constraints = [cp.sum(weights) == 1, cp.norm(weights, 1) <= leverage_limit]

    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract optimal weights
    optimal_weights = weights.value

    return optimal_weights

if __name__ == '__main__':
    
    tickers = get_tickers(file_path='sp500tickers.txt',limit=-1)
    
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
    
    alpha_for_leverage = 10.0
    leverage_limits = [1,2,4]
    portfolio_values= []
    #
    for limit in leverage_limits:
        optimal_weights_leverage = mean_variance_optimization_leverage(training_data, alpha_for_leverage, limit)
        portfolio_values_leverage = np.dot(testing_data, optimal_weights_leverage)
        
        portfolio_values.append(portfolio_values_leverage)

    # Convert portfolio values to DataFrame for easy plotting
    portfolio_df = pd.DataFrame(np.array(portfolio_values).T, 
                                columns=[f'Leverage = {limit}' for limit in leverage_limits], 
                                index=testing_data.index
                                )

    portolio_cumprod = (portfolio_df + 1).cumprod()
    portolio_cumprod.plot(figsize=(10, 6))
    plt.title('Portfolio Values - with leverage')
    plt.xlabel('')
    plt.ylabel('Portfolio Return')
    plt.legend()
    plt.show()
    
    """
    information_ration
    """
    
    # Calculate Information Ratios based on daily returns for the testing period
    # for simplicity we assume risk-free rate = 0
    risk_free_rate = 0
    information_ratios = ((portolio_cumprod - 1).mean() - risk_free_rate) / (portolio_cumprod - 1).std()

    # Print Information Ratios
    print('\nInformation Ratios for leverages:')
    print(information_ratios)

    # Identify the alpha value that performs the best
    best_limit = information_ratios.idxmax()
    print('\nBest leverage limit:', best_limit)


    print("""
        Based on the information ratios, the best leverage limit 1 with information ratio of 1.58 compaired to the others
        
        The information ratio is a measure of the excess is calculated with respect to the risk measured as a standard deviation of the returns.
        The higher the information ratio the better the portfolio performance.
        
        Intuitively, the information ration should tend to decrease as the leverage increases.
        This should indicate that the more leverage implicate more risk.
        """
        )