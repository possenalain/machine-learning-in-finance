"""
price dynamics of multiple crytocurrencies jointly via multivariate ARIMA. 
 BTC, ETH, LTC, SOL, AVAX, KDA will be used.

predict these cryptocurrency prices by multivariate ARIMA.
Our train period 2021-2022 and test period will be 2022-2023. 

Once predicted after training, our portfolio will be allocated to the top two cryptocurrencies with highest return estimation.

p,q,r in ranges [0.1, 0.5].
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

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from pmdarima import auto_arima

from statsmodels.tsa.stattools import acf, pacf,adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from tools.helper_functions import *
pd.options.mode.chained_assignment = None 

import warnings
warnings.filterwarnings("ignore")



from itertools import product

DATA_DIR = os.getenv('FINANCE_DATA_DIR', '../../data')

"""
    Helper functions
"""

def handle_missing_values(data):
    """
    given a pandas dataframe, this function will:
    1. fill numerical values with the median
    2. fill categorical values with the mode
    
    args:
        data: pandas dataframe
    return:
        data: pandas dataframe
    """
    
    #1. handle missing values
    #1.1 fill numerical values with the median
    numerical_cols = data.select_dtypes(include=['number']).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())#,inplace=True

    return data

def download_crypto_data(symbols : list, 
                         start_date : str, 
                         end_date : str,
                         save_as = "data/crypto_data.csv"):
    """
         download crypto data from yahoo finance for the given symbols and date range
         and save it to a csv file to avoid downloading it again
         args:
                symbols: list of str
                start_date: str
                end_date: str             
    """    
    print(f"Downloading data...")
    # get data
    data = yf.download(symbols, 
                    start=start_date, 
                    end=end_date, 
                    progress=False)
    sp_prices_df = pd.DataFrame(data["Close"])
    #save data
    print(f"Saving data ... ")
    Path(save_as).parent.mkdir(parents=True, exist_ok=True)
    sp_prices_df.to_csv(save_as)

def fit_varima_model(data : pd.DataFrame, 
                     order : tuple = (1,1), 
                     prediction_horizon = 252
                     ):
    """ 
        fit varma model with given data and order
        args:
            data: pd.DataFrame 
            order: tuple (p,q)
            horizon_for_testing: int (how many days to predict)
        return:
            predicted_values: pd.DataFrame
            varma_fit_model incase we want to use it later
    """
    
    model = VARMAX(data, 
                   order=order, 
                   trend='c'
                   )
    try:
        fit_model = model.fit(disp=False)
    except Exception as e:
        print("cant fit model")
        return None, None
    #get predictions
    predictions = fit_model.get_forecast(steps = prediction_horizon)
    predictions = predictions.predicted_mean
    
    return predictions , fit_model

def information_ratio(actual : pd.DataFrame, 
                      predicted : pd.DataFrame,
                      aggregate : bool = False
                      ):
    """
        Information ratio adjusted when std is zero
        args:
            actual: pd.DataFrame
            predicted: pd.DataFrame
            aggregate: if true return a single value instead of each assests information ratio
            
    """
    predicted.index = actual.index
    excess_returns = actual - predicted 
    information_ratio = excess_returns.mean() / (excess_returns.std() + np.finfo(np.float32).eps)
        
    return information_ratio if not aggregate else information_ratio.mean()


if __name__=="__main__":
    
    symbols = ['BTC-USD', 
            'ETH-USD', 
            'LTC-USD', 
            'AVAX-USD', 
            'KDA-USD', 
            'SOL-USD',
            'DOGE-USD'
            ] 
    train_start_date = '2021-01-01'
    train_end_date = '2022-12-31'
    test_start_date = '2022-12-31'
    test_end_date = '2023-12-31'


    # load data (download if not already downloaded)
    file_to_load = f"{DATA_DIR}/crypto_data.csv"
    if not Path(file_to_load).exists():
        download_crypto_data(symbols, 
                            start_date = train_start_date, 
                            end_date = test_end_date,
                            save_as = file_to_load
                            )
    data = pd.read_csv(file_to_load, parse_dates =True, index_col=0)

    # preprocess data

    # # before preprocessing
    print_df_information(data)

    # handle missing values
    data = handle_missing_values(data)

    # # after preprocessing
    print_df_information(data)

    # since we will be working with returns instead
    # convert to returns
    data = data.pct_change().dropna()
    # infer frequency
    data.index.freq = pd.infer_freq(data.index)

    # we might need to check the stationarity of the data 
    stationality_df = {}
    for symbol in list(data.columns.tolist()):
        ad_test_results = adfuller(data[symbol])
        stationality_df.update({symbol: {
                                "ADF Statistic": ad_test_results[0],
                                "p-value": ad_test_results[1]}
                                })
    stationality_df = pd.DataFrame(stationality_df).T
    display(stationality_df)

    """

        looking at the p-values, we can say that values are small enough to consider stationality.
        this is also supported by the plot of the data. although there might still be some seasonality.
        indicated by the spikes at some points they don't seem to be periodic.
        
    """

    data.plot(figsize=(12, 5), title="Crypto returns")
    plt.show()


    # split data into train and test
    train_data = data[:train_end_date]
    test_data = data[test_start_date:]

    print(f"train_data: {train_data.shape} , test_data: {test_data.shape}")


    #handle p, and q orders
    from itertools import product

    """ 
    results in so many tuples and time consuming.
    istead we will use auto_arima to reduce search space
    """
    # # p = q = range(0, 5)
    # orders = list(product(p, q))
    # orders
    from pmdarima import auto_arima

    # get best orders for each symbol
    # retain only p and q and remove redundants

    best_orders = {}
    for symbol in list(train_data.columns.tolist()):
        stepwise_fit = auto_arima(train_data[symbol], 
                                start_p=0, 
                                start_q=0, 
                                max_p=12, 
                                max_q=12, 
                                suppress_warnings=True, 
                                stepwise=True
                                )
        order = stepwise_fit.order
        best_orders.update({symbol: order})
        print(f"symbol: {symbol}, order: {order}")

    #just retain p and q and remove redundants
    best_orders_varma = list(set([(order[0],order[2]) for order in best_orders.values()]))
    best_orders_varma = list(filter(lambda x: x!=(0,0), best_orders_varma))
    print(f"orders adjusted for varma: {best_orders_varma}")


    # fit the model and get predictions
    information_ratios_results = []
    predictions_dict = {}
    for order in best_orders_varma:
        print(f"fitting model for order: {order}")
        predicted_prices , fit_model = fit_varima_model(train_data,
                                                        order=order,
                                                        prediction_horizon = len(test_data)) 
        assert predicted_prices is not None, "model fit failed"
        assert fit_model is not None, "model fit failed"

        # Calculate Information Ratio
        test_data_for_horizon = test_data.iloc[:len(predicted_prices)]
        info_ratio = information_ratio(test_data_for_horizon, 
                                    predicted_prices, 
                                    aggregate=False
                                    )
        predictions_dict[order] = {"prices": predicted_prices, "model": fit_model}
        information_ratios_results.append({"order": order, **info_ratio.to_dict()})

    # examine the best model based on information ratio
    #per symbol information ratios

    information_ratios_df = pd.DataFrame(information_ratios_results)
    display(information_ratios_df)



    # we will used aggregated information ratio to get the best model
    # instead of looking at each symbol's information ratio

    information_ratios_df.set_index("order", inplace=True)
    aggregated_info_ratio = information_ratios_df.sum(axis=1).sort_values(ascending=False)
    aggregated_info_ratio = pd.DataFrame(aggregated_info_ratio, columns=["information_ratio"])
    # get the best order based on agregate information ratio
    best_model_order = aggregated_info_ratio.index[0]
    print(f"best order: {best_model_order}")

    display(aggregated_info_ratio)


    _days_to_plot = -1
    _symbols_to_plot = [str(c) for c in test_data.columns]
    # _symbols_to_plot = ['BTC-USD']

    print(_symbols_to_plot)
    # # get and plot predictions of best model
    best_model_predictions = predictions_dict[best_model_order]["prices"]
    best_model_fit = predictions_dict[best_model_order]["model"]

    cumulative_returns = (best_model_predictions + 1).cumprod()
    cumulative_test_returns = (test_data_for_horizon + 1).cumprod()

    cumulative_returns.columns = [f"pred_{col}" for col in cumulative_returns.columns]
    cumulative_test_returns.columns = [f"actual_{col}" for col in cumulative_test_returns.columns] 

    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_test_returns.iloc[:_days_to_plot][[f"actual_{col}" for col in _symbols_to_plot]].plot(ax=plt.gca(), color=['r','g','b','y','k'])
    cumulative_returns.iloc[:_days_to_plot][[f"pred_{col}" for col in _symbols_to_plot]].plot(linestyle='--', ax=plt.gca(), color=['r','g','b','y','k'])
    # plt.legend(['Actual', 'Predicted'])
    plt.show()

    """ 
    although it is not obvious at first the varma model is able to predict meaningful values in the 
    first few days but then later on looses the ability to predict.(as seen from later days)
    
    let zoom in to the first few days and with few symbols
    """


    _days_to_plot = 5
    # _symbols_to_plot = [str(c) for c in test_data.columns]
    _symbols_to_plot = ['BTC-USD', 'ETH-USD']

    print(_symbols_to_plot)
    # # get and plot predictions of best model
    best_model_predictions = predictions_dict[best_model_order]["prices"]
    best_model_fit = predictions_dict[best_model_order]["model"]

    cumulative_returns = (best_model_predictions + 1).cumprod()
    cumulative_test_returns = (test_data_for_horizon + 1).cumprod()

    cumulative_returns.columns = [f"pred_{col}" for col in cumulative_returns.columns]
    cumulative_test_returns.columns = [f"actual_{col}" for col in cumulative_test_returns.columns] 

    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_test_returns.iloc[:_days_to_plot][[f"actual_{col}" for col in _symbols_to_plot]].plot(ax=plt.gca(), color=['r','g','b','y','k'])
    cumulative_returns.iloc[:_days_to_plot][[f"pred_{col}" for col in _symbols_to_plot]].plot(linestyle='--', ax=plt.gca(), color=['r','g','b','y','k'])
    # plt.legend(['Actual', 'Predicted'])
    plt.show()

    """    
        Get the latest return percentage  to choose assets
    """
    best_predictions = (best_model_predictions + 1).cumprod()
    last_return = best_predictions.iloc[-1]
    assets_sorted_by_return = last_return.sort_values(ascending=False)

    """

    Based on highest return percentage

    DOGE-USD    29.376705%
    KDA-USD      7.562899%

    We would be out top 2 assets to allocate our portfolio

    """

    """

    Overall Varma model not able to catch the time and multivariate dependencies in the data.
    the models could be improved by instead using more sophisticated models, or more data preprocessing to ensure stationarity.
    also, we could use models that take seasonality and integration into account such as sarima to capture seasonality.

    """
    display(assets_sorted_by_return)
