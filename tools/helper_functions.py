
"""
    Helper functions
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cvx
from pathlib import Path
import seaborn as sns
import datetime 

#get the ticker 
def get_tickers(file_path : Path = "sp500tickers.txt",
                limit:int = -1
                ) -> list:
    """
    Args: 
        file path
    return:
        list of tickers
    """
    #get the tickers
    tickers = list(np.array(pd.read_csv(f'{file_path}', sep= "\t" , header=None).values[:limit]).flatten())
    # tickers = tickers[0].tolist()
    # tickers = sorted(tickers)
    # tickers = tickers[:limit]
    
    return tickers

def pull_data(download : bool = False,
              tickers : list = None, 
              start_date : str = None, 
              end_date : str = None, 
              split: str = "train",
              DATA_DIR : str = '../../data'):
    """
        given time range pull sp500 data
    """
    
    #if data is not downloaded download it
    
    file_to_load = f"{DATA_DIR}/sp500_{split}.csv"
    
    if download or not Path(file_to_load).is_file():
        print(f"Downloading data for {split} set")
        # get data
        data = yf.download(tickers, 
                        start=start_date, 
                        end=end_date, 
                        progress=False)
        sp_prices_df = pd.DataFrame(data["Adj Close"])
        
        print(f"Saving data for {split} set")
        Path(file_to_load).parent.mkdir(parents=True, exist_ok=True)
        #save data
        
        sp_prices_df.to_csv(file_to_load)
        print(f"Data for {split} set saved")
    
    else:
        print(f"Loading data for {split} set")
        sp_prices_df = pd.read_csv(file_to_load, parse_dates=True, index_col=0)
        
    print(f"Data for {split} set loaded")
        
    return sp_prices_df

def get_refined_data_and_tickers( tickers, 
                                 training_data,
                                 testing_data):
    
    #force data types
    training_data = training_data.infer_objects()   
    testing_data  = testing_data.infer_objects()

    # #reset index
    # training_data.reset_index(drop=True)
    # testing_data.reset_index(drop=True)
    
    #from the data get columns that don't contan null values
    training_data = training_data.dropna(axis=1)
    testing_data = testing_data.dropna(axis=1)

    # only use intersecting columns
    intersecting_columns = list(set(training_data.columns).intersection(testing_data.columns))
    tickers = list(intersecting_columns)
    try:
        tickers.remove("Date")
    except:
        pass
    training_data = training_data[tickers]
    testing_data = testing_data[tickers]
    
    return tickers, training_data, testing_data

def get_right_format(_data : pd.DataFrame, 
                     missing_values_correction_method : str = 'median'):
    
    # infer data types
    _data = _data.infer_objects()
    # only retain numerical columns
    # only retain numerical 
    _data = _data.select_dtypes(include=[np.number])

    assert missing_values_correction_method in ['median', 'dropna'], 'invalid missing values correction method'
    ## drop columns with missing values
    if missing_values_correction_method == 'dropna':
        #drop columns with missing values
        _data = _data.dropna(axis=1)
    else:
        # replace missing values with median
        _data = _data.fillna(_data.median())

    return _data

def print_df_information(data):
    """
    
    give useful information about the data
    
    """
    
    print(f"="*50)
    
    #1. print the shape of the data
    print(f"Data shape (rows, columns) : {data.shape}")
    print(f"Categorical columns : {len(list(data.select_dtypes(include=['object']).columns))}")
    print(f"Numerical columns : {len(list(data.select_dtypes(include=['number']).columns))}")
    
    # # data info 
    # print(f"*"*50)
    # display(data.info())
    
    # memory usage
    df_memory_usage = data.memory_usage(deep=True).sum()
    df_memory_usage_mb = df_memory_usage / (1024 * 1024)
    print(f"memory usage: {df_memory_usage_mb:.2f} MB")
    
    # unique values
    print(f"*"*50)
    print(f"Unique values for categorical columns")
    category_cols =data.select_dtypes(include=['object'], exclude=["number"]).columns
    unique_values = data[category_cols].nunique()
    display(unique_values)
        
    
    # missing values
    print(f"*"*50)
    print(f"Columns with missing values")
    columns_with_missing_values = data.columns[data.isnull().any()]
    display(data[columns_with_missing_values].isnull().sum())
    
    print(f"="*50)

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

    #1.2 fill categorical values with the mode
    category_cols = data.select_dtypes(include=['object']).columns
    if len(category_cols) > 0:
        data[category_cols] = data[category_cols].fillna(data[category_cols].mode().iloc[0])#, inplace=True  
    
    return data