# Singapore_Resale_Flat_Price_Prediction
A Regression Machine Learning model is built to predict the Resale Flat Price in Singapore Resale market. User can provide several input details such as the town the flat is located in, floor size, flat type, flat model and lease commencement year and get the predicted resale price of a flat in Singapore | Python | Machine Learning | Sklearn | Pandas | Streamlit |

**Introduction**

  The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can
affect resale prices, such as location, flat type, floor area, and lease duration. A predictive model can help to overcome these challenges by providing users with an
estimated resale price based on these factors. 

**User Guide**

Enter the town in which the flat is located, the flat type and model, storey range, lease commencement date, the flat sale date and floor area in square meters to get the resale price prediction by clicking the predict resale price button on the Streamlit app.

**Developer Guide**

**1.Tools required**

  • Python

  • Visual Studio Code

  **2.Python libraries to install**
  **a.For dashboard creation**

    • Streamlit

  **b.For Analysis**

    • matplotlib.pyplot

    • Seaborn

    • scipy.stats

    • Numpy

    • Pandas

  **c.For Machine Learning models**

    • Scikit-learn

    • xgboost

  **3. Modules to import**

  a. File handling Libraries

    • import pickle
    
    • import os

  b. Pandas Library

    • import pandas

  c. Numerical calculatoins Library

    • import numpy as np

    • from scipy.stats import skew

  d. Visualization Libraries

    • import matplotlib.pyplot as plt

    • import seaborn as sns

  e. Dashboard Libraries

    • import streamlit as st

  f. Machine Learning Libraries

    • from sklearn.preprocessing import OneHotEncoder

    • from sklearn.model_selection import train_test_split

    •  from sklearn.preprocessing import StandardScaler
    
    • from sklearn.metrics import mean_squared_error, r2_score
    
    • from sklearn.linear_model import LinearRegression

    • from sklearn.tree import DecisionTreeRegressor
    
    • from sklearn.ensemble import RandomForestRegressor
    
    • from sklearn.ensemble import AdaBoostRegressor

    • from sklearn.ensemble import ExtraTreesRegressor

    • from sklearn.ensemble import GradientBoostingRegressor

    • from xgboost import XGBRegressor

  g. Others

    • import warnings

  **4.Process**

    • Data is extracted from the csv files and stored as a pandas Dataframe

    • Duplicates are dropped and a date column is type casted to pandas datetime
    
    • Skewness of continuous variables are analysed

    • Bi variate analysis is performed on resale price and floor area columns and a heatmap is visualized

    • Categorical columns are one hot encoded

    • Regression model is chosen after training several models and getting their accuracy

    • The Random Forest Regressor model which gives 97.5% accuracy is chosen

    • The selected models are pickled which can be used in the Streamlit app file

    • A simple Streamlit UI is built where user enters the required values and the predictions are printed

        
**NOTE:**

  • The EDA.ipynb file contains the code for data extraction, pre processing, EDA and Machine learning models
    
  • The Sg_dahboard.py file consists of the code for Streamlit dashboard creation

  • Data is downloaded as csv files from the Singapore Government webiste linkk provided in the Problem Statement file

  • The pickled models are also uploaded which can be used in the Streamlit file
