# Outlier Detection with Isolation Forest

In this project, we intend to use outlier detection techniques on different datasets to find anomalies in the data.
Specifically, we will be using **Isolation Forest**. As a benchmark, we will use various methods mainly from pyod library.


## Outlier Detection Techniques

There is a considerable amount of research on outlier detection techniques. 

1. Isolation Forest (https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
2. LOF: Local Outlier Factor
3. LoOP: Local Outlier probablity
4. ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions
5. Rapid distance-based outlier detection via sampling
6. LOCI: Fast outlier detection using the local correlation integral 
...


## Financial Datasets for Outlier Detection

The scarcity of financial datasets for outlier detection tasks is a significant challenge in the field. While there are a few notable datasets available, such as the creditcard dataset containing transactions made by European cardholders, the number of publicly available financial datasets specifically designed for outlier detection is limited or not intepretable if public at all. Researcher often have to rely on a small number of datasets or resort to synthetic data generation techniques to address this scarcity in the field of outlier detection in finance. In total we have used 7 datasets hand picked from different sources.

**Datasets details table**

| dataset     | #instances | #features | #outliers  | #outliers (%)  |from           |#instances | #outliers|
|:--------    |-----------:|----------:|----------: |--------------: |--------------:|---------------:|---------------------------:|
| creditcard  | 284,807     | 30        | 492       | 0.17           | Kaggle        |All             |All                         |
| SFD         | 6,362,620   | 10        | 8213      | 10             | HuggingFace   |500000          |233                         |
| 2018-FinData| 3,167       | 224       | 121       | 3.82           | Kaggle        |All             |All                         |
| CrCFraud    | 1,500,000   | 22        | 6006      | 0.05           | HuggingFace   |200000          |1645                        |
| Defaulter   | 10000       | 3         | 1000      | 10             | Kaggle        |9,841           |174                         |
| campaign    | 41,188      | 62        | 4640      | 11.27          | ADBench       |All             |All                         |
| fraud       | 284,807     | 29        | 492       | 0.17           | ADBench       |All             |All                         |


**creditcard**
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
PCA transformation was applied to the dataset to protect sensitive information. 

**synthetic-fraud-detection (SFD)** 
synthetic-fraud-detection in banking activities including Payment, Withdrawal, Debit, Transfer, and Cash-out, etc. the dataset as opposed to creditcard dataset has attributes that are not PCA transformed and are more interpretable. With more than 6 million instances. 

**2018_financial_data (2018-FinData)**
 collects 200+ financial indicators for all the stocks of the US stock market. The financial indicators have been scraped from Financial Modeling Prep API, and are those found in the 10-K filings that publicly traded companies release yearly. The dataset is originally for stock direction prediction, but we use it for outlier detection with stocks that go up as normal and outliers otherwise. we use the 2018 data and sample the stocks that went down to represent the outliers.

**dazzle-nu/CIS435-CreditCardFraudDetection (CrCFraud)** 
The dataset contains transactions made by credit cards with detailed attributes including card number, time, amount, etc. The dataset is originally for credit card fraud detection. The dataset has 1+M transactions. 

**Defaulter**
The dataset contains information about the customers of a bank. The dataset is orginally intended to decide whether a customer is a defaulter or not ( meaning whether the customer will pay back the loan or not). We sample defaulters and take them to be outliers. 
The dataset is does not have many attrbutes but important ones are: balance, income, student. 

**campaign**
Campain is originally a dataset from ADBench ( Anomally Detection Benchmark) and is used for outlier detection. The dataset has 62 attributes and 41188 instances. with 4640 outliers. although not much information is provided as to what kind of data it is, we consider it a good example of outlier detection dataset as it is hosted in ADBench a known benchmark for outlier detection.

**fraud**
Similar to campaign, fraud is originally a dataset from ADBench ( Anomally Detection Benchmark) and is used for outlier detection. The dataset has 29 attributes and 284807 instances. with 492 outliers. 


## methods shorts

- Iforest: Isolation Forest 
- LOF: Local Outlier Factor
- ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions
- COPOD: Copula-Based Outlier Detection
- ABOD: Angle-Based Outlier Detection
- QMCD: Quasi-Monte Carlo Discrepancy outlier detection
- Rapid distance-based outlier detection via sampling
