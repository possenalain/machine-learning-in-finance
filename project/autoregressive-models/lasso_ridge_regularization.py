"""
Regularized linear regression by using sklearn to predict house sale price.
L1 norm and L2 norm regularized linear regression by using sklearn library.
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
import cvxpy as cp

DATA_DIR = os.getenv('FINANCE_DATA_DIR', '../../data')

"""
Helper functions for data cleaning 

The purpose of helper functions is to be able to use them for subsequent problems

"""

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
    data[category_cols] = data[category_cols].fillna(data[category_cols].mode().iloc[0])#, inplace=True  
    
    return data

def normalize_data(data, target_col = None):
    """ normalize numerical data using standard scaler exluding the target column
    args:
        data: pandas dataframe
        target_col: str, default None
    return:
        data: pandas dataframe
    """
    numerical_cols = data.select_dtypes(include=['number']).columns
    numerical_cols = numerical_cols[numerical_cols!=target_col]
    
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data
        
def get_numerical_categorical_cols(data):
    """ 
    return numerical and categorical columns from a pandas dataframe
    """
    numerical_cols = data.select_dtypes(include=['number']).columns
    category_cols = data.select_dtypes(include=['object']).columns
    
    return numerical_cols, category_cols
 
def categorical_to_numerical(data, 
                             threshold=8,
                             target_categorical = None):
    """ given a pandas dataframe, this function will:
    1. get columns and their unique values count
    2. encode columns with unique values count <= threshold with one hot encoding
    3. encode columns with unique values count > threshold with label encoding

    args:
        data: pandas dataframe
        threshold: int, default 8
    return:
        data: pandas dataframe
        
    Note:
        if target_categorical is not None, then the target column will be encoded with label encoding
    """
    category_cols = data.select_dtypes(include=['object'], exclude=["number"]).columns
    category_cols = category_cols[category_cols!=target_categorical]
    unique_values = data[category_cols].nunique()
    ## 2.1.1 cats for dummies
    cats_for_dummies = unique_values[unique_values <= threshold].index.tolist()
    print(f"categorical cols encoded with one hot encoding : {len(cats_for_dummies)}")
    # display(cats_for_dummies)#unique_values[cats_for_dummies]
    data = pd.get_dummies(data, columns=cats_for_dummies, drop_first=True)
    
    # create a labels map
    # create a label mapping for the target column
    labels_map = {}
    if target_categorical is not None:
        labeL_mapping = { }
        for i, val in enumerate(data[target_categorical].unique()):
            labeL_mapping[val] = i
        labels_map ={ v:k for k,v in labeL_mapping.items()}

    ## 2.1.2 cats for label encoding
    cat_for_labelE = list(unique_values[ unique_values > threshold].index.tolist())
    cat_for_labelE = list(set(cat_for_labelE if not target_categorical else [target_categorical, *cat_for_labelE]))
    print(f"categorical cols for label encoding: {len(cat_for_labelE)}")
    # display(cat_for_labelE) #unique_values[cat_for_labelE]
    data[cat_for_labelE] = data[cat_for_labelE].apply(lambda x: x.astype('category').cat.codes)
    
    
    
    return data , labels_map


if __name__=="__main__":
    
    # load the data and do cleaning 

    #1. load the data
    path_to_data = Path(f'{DATA_DIR}/kaggle_house.csv')

    data = pd.read_csv(path_to_data, infer_datetime_format=True)
    data.set_index('Id', inplace=True)
    
    # get data summary 
    print("before cleaning")
    print_df_information(data)
    

    #1. handle missing values
    #2. normalize the numerical data
    #3. handle categorical values
    #4. normalize the data

    #1. handle missing values
    data = handle_missing_values(data)

    #2. normalize the numerical data
    data = normalize_data(data, target_col='SalePrice')

    #3. handle categorical values
    data, _labels_map = categorical_to_numerical(data, 
                                    threshold=100, 
                                    target_categorical = None)

    # print data shape 
    print(f"Data shape after cleaning : {data.shape}")
    
    
    # implements the regularized regression model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    # separate the features from the target
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    # we will use the scikit-learn train_test_split function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape} X_test: {X_test.shape}, y_test: {y_test.shape}")

    # implement lasso  and ridge with cross validation
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import RidgeCV

    # get scores R2 and RMSE
    def get_prediction_scores(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, rmse


    # create lasso and ridge models
    lasso = LassoCV(cv=5, 
                    random_state=42, 
                    max_iter=100000,  
                    alphas=[0.001, 0.01, 0.1, 0.25, 0.5, 1.0]
                    )
    ridge = RidgeCV(cv=5,
                    alphas=[0.001, 0.01, 0.1, 0.25, 0.5, 1.0],
                    scoring='neg_mean_squared_error')

    # fit the models
    lasso.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    # get predictions
    y_pred_lasso = lasso.predict(X_test)
    y_pred_ridge = ridge.predict(X_test)

    # get scores
    r2_lasso, rmse_lasso = get_prediction_scores(y_test, y_pred_lasso)
    r2_ridge, rmse_ridge = get_prediction_scores(y_test, y_pred_ridge)

    # print the scores
    print(f"Lasso: R2 score: {r2_lasso}, RMSE: {rmse_lasso}")
    print(f"Ridge: R2 score: {r2_ridge}, RMSE: {rmse_ridge}")

    print(f"""
        We need to have a small RMSE and a high R2 score.
        a good R_2 score is close to 1 
        a good RMSE should be close to 0
        
        given these two metrics, we can say Ridge regularization performs better than Lasso regularization
        """)