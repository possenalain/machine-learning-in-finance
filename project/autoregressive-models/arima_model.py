"""
fitting ARIMA type model to Turkish monthly unemployment rate data. 
The data is obtained from Central Bank.

ARIMA from statsmodels implementation is used.

is using SARIMA (Seasonal ARIMA) better ?.

Integration orders: 0, 1 or 2
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cvx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf,adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time

from tools.helper_functions import *
pd.options.mode.chained_assignment = None 

import warnings
warnings.filterwarnings("ignore")



from itertools import product

""" 
Helper functions
"""
def fit_evaluate_arima(train_data,
                       test_data,
                       order=(1,0,1)):
    # fit arima model 
    assert order is not None and len(order) == 3, ""
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
    predictions = pd.Series(predictions, index=test_data.index)
    
    res_df = pd.DataFrame({'true':test_data.iloc[:,0],
                            "pred":predictions
                            })

    residuals =res_df['pred'] - res_df['true']
    
    information_ratio =  residuals.mean() / residuals.std()
    rmse = np.sqrt(np.mean(residuals**2))
    
    metrics = {
        'order': order,
        'rmse': rmse,
        'information_ratio': information_ratio,
        'aic': model_fit.aic,
        'bic': model_fit.bic
    }
    
    return model_fit, metrics , res_df


def fit_arima_multiple_orders(train_data,
                              testing_data,
                              orders=None):
    
    p = d = q = range(0, 3)
    pdq = list(product(p, d, q))
    
    orders = orders if orders is not None else pdq
    # for each order 
    metrics = []
    for order in orders:
        try:
            _,  per_order_metrics , res_df = fit_evaluate_arima(train_data, testing_data, order)
            metrics.append(per_order_metrics)
            
        except Exception as e:
            print(f" order : {order} failed")
            # print(e)
            continue
    return metrics



def fit_evaluate_SARIMA(train_data,
                       test_data,
                       seasonal_order=(1,0,1)):
    # fit arima model 
    arima_order = (1,0,1)
    assert seasonal_order is not None and len(seasonal_order) == 4, ""
    model = SARIMAX(train_data, order=arima_order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
    predictions = pd.Series(predictions, index=test_data.index)
    
    res_df = pd.DataFrame({'true':test_data.iloc[:,0],
                            "pred":predictions
                            })

    residuals =res_df['pred'] - res_df['true']
    
    information_ratio =  residuals.mean() / residuals.std()
    rmse = np.sqrt(np.mean(residuals**2))
    
    metrics = {
        'fixated_order': arima_order,
        'order': seasonal_order,
        'rmse': rmse,
        'information_ratio': information_ratio,
        'aic': model_fit.aic,
        'bic': model_fit.bic
    }
    
    return model_fit, metrics , res_df


def fit_SARIMA_multiple_orders(train_data,
                              testing_data,
                              orders=None):
    
    p = d = q = range(0, 3)
    pdq = list(product(p, d, q))
    
    orders = orders if orders is not None else pdq
    # for each order 
    metrics = []
    for order in orders:
        try:
            _,  per_order_metrics , res_df = fit_evaluate_SARIMA(train_data, 
                                                                testing_data, 
                                                                seasonal_order=(*order,12))
            metrics.append(per_order_metrics)
            
        except Exception as e:
            print(f"seasonal order : {order} failed")
            print(e)
            continue
    return metrics


DATA_DIR = os.getenv('FINANCE_DATA_DIR', '../../data')


if __name__ == "__main__":
        
    # load the data

    #1. load the data
    path_to_data = Path(f'{DATA_DIR}/unemployment.xlsx')

    """ read only the first 120 rows """
    data = pd.read_excel(path_to_data, nrows=116 ,parse_dates=[0],index_col=0)


    # infer the frequency of the data
    data.index.freq = pd.infer_freq(data.index)

    # get visual 
    data.plot(figsize=(12, 5))
    plt.axhline(data.mean()[0], color='r', alpha=0.2, linestyle='--')
    plt.title('Unemployment Rate') 

    """   
    # the red line is the mean of the data
    # althought the data seems to be centered around the mean,
    # once we zoom in, it's obvious that the data has varying means

    # we remove trend with differencing
    # next we check for stationarity with acf and pacf plots


    """;

    # first_diff = data.diff()[1:]
    first_diff = data.diff()[1:]
    first_diff.plot(figsize=(12, 5))
    plt.axhline(first_diff.mean()[0], color='r', alpha=0.2, linestyle='--')
    plt.title('Unemployment Rate First Difference (no trend)')

    """
    Now data is centered around a contant mean
    """;

    ad_test_results = adfuller(first_diff)
    print(f"ADF Statistic: {ad_test_results[0]}")
    print(f"p-value: {ad_test_results[1]}")

    """ given that p-value is high we can state that the data is not perfectly stationary """

    # fit acf and pacf plots
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    axes = ax.ravel()
    plot_acf(first_diff, lags=20,  ax=axes[0])
    plot_pacf(first_diff, lags=20,   ax=axes[1])
    plt.show()


    """

    from the acf, pacf plots, we has significant lags which might implicate seasonality

    """;    

    train_end = "2022-01-01"

    train_data = first_diff[:train_end]
    test_data = first_diff[train_end:]

    plt.figure(figsize=(12, 5))
    plt.plot(train_data, color='b',label='train')
    plt.plot(test_data, color='r', label='test')
    plt.legend(loc='best')
    #use data from 2022 as test data

    ## fit ARIMA model

    # uncomments orders to test None default to prduct(p,d,q) for p,d,q in range(0,3)
    # orders = [(1,0,1),(12,0,2),(3,0,3)] 
    # orders = [(1,0,1),(12,0,2),(3,0,3),(1,0,2),(2,0,1),(2,0,2),(3,0,1),(1,0,3),(2,0,3)] 
    orders = None

    arima_orders_results = fit_arima_multiple_orders(train_data, 
                                                    test_data, 
                                                    orders= orders
                                                    )
    results_df = pd.DataFrame(arima_orders_results)

    results_df.sort_values(by='information_ratio', ascending=False, inplace=True)
    print(f"best order based on information ratio : {results_df.iloc[0]['order']}")
    print(f"best integration order : {results_df.iloc[0]['order'][1]}")
    display(results_df.head(3))

    results_df.sort_values(by='rmse', ascending=True, inplace=True)
    print(f"best order based on rmse : {results_df.iloc[0]['order']}")
    arima_best_order = results_df.iloc[0]['order']
    display(results_df.head(3))

    results_df.sort_values(by='aic', ascending=True, inplace=True)
    print(f"best order based on aic : {results_df.iloc[0]['order']}")
    display(results_df.head(3))


    ## SARIMA
    # uncomments orders to test None default to prduct(p,d,q) for p,d,q in range(0,3)
    # seasonal_orders = [(1,0,1),(12,0,2),(3,0,3)] 
    # seasonal_orders = [(1,0,1),(12,0,2),(3,0,3),(1,0,2),(2,0,1),(2,0,2),(3,0,1),(1,0,3),(2,0,3)] 
    seasonal_orders = None

    sarima_orders_results = fit_SARIMA_multiple_orders(train_data, 
                                                    test_data, 
                                                    orders= seasonal_orders
                                                    )

    results_df = pd.DataFrame(sarima_orders_results)

    results_df.sort_values(by='information_ratio', ascending=False, inplace=True)
    print(f"best order based on information ratio : {results_df.iloc[0]['order']}")
    print(f"best integration order : {results_df.iloc[0]['order'][1]}")
    sarima_best_order = results_df.iloc[0]['order']
    display(results_df.head(3))

    results_df.sort_values(by='rmse', ascending=True, inplace=True)
    print(f"best order based on rmse : {results_df.iloc[0]['order']}")
    display(results_df.head(3))

    results_df.sort_values(by='aic', ascending=True, inplace=True)
    print(f"best order based on aic : {results_df.iloc[0]['order']}")
    display(results_df.head(3))


    # arima_results with best order 
    _,  per_order_metrics , arima_best_res_df =  fit_evaluate_arima(train_data, test_data, arima_best_order)
    _,  per_order_metrics , sarima_best_res_df = fit_evaluate_SARIMA(train_data, test_data, sarima_best_order)

    """

        ARIMA Vs SARIMA
        
        both qualitatively and quantitatively, SARIMA performs better than ARIMA.
        this is because SARIMA takes into account seasonality.
        
        for both models we attemp to choose best orders based on metrics such as rmse, aic, bic, information ratio on test data.
        we also plot the actual vs predicted values for both models.
        
        arima performs okayish for closer timesteps but appears to be constant later on.
        sarima  performs relatively well for close and further timesteps.

    """

    train_data["2020-01-01":].plot(figsize=(12, 5))
    arima_best_res_df['pred'].plot(color='r', label='arima')
    sarima_best_res_df['pred'].plot(color='g', label='sarima')
    sarima_best_res_df['true'].plot(color='b', label='actual')
    plt.legend(loc='best')
    plt.title('ARIMA vs SARIMA comparison on test data')
    plt.show()