# -*- coding: utf-8 -*-

# Importing
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import os

def clean_db():

    # Defining the dataset...!
    df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

    # Removing unwanted columns for feature purposes...!
    # Sum of These 4 columes are represented as Global_Sales...!
    del df['NA_Sales']
    del df['EU_Sales']
    del df['JP_Sales']
    del df['Other_Sales']

    # Total No of rows were 16719...!
    # After removing null values 16416...!
    # Difference is 303. It won't harm to the end result...!
    df = df[df["Year_of_Release"].notnull()]
    df = df[df["Genre"].notnull()]
    df = df[df["Publisher"].notnull()]
    df['Year_of_Release'] = df['Year_of_Release'].astype('int64')
    df['User_Score'] = df['User_Score'].replace('tbd', 0).astype('float64')

    # Replacing null numeric values to 0...!
    df['Critic_Score'] = df['Critic_Score'].replace(np.nan, 0)
    df['Critic_Count'] = df['Critic_Count'].replace(np.nan, 0)
    df['User_Score'] = df['User_Score'].replace(np.nan, 0)
    df['User_Count'] = df['User_Count'].replace(np.nan, 0)

    # Replacing null characteristic values to applicable values...!
    df["Developer"].fillna("Open Source", inplace=True)
    df["Rating"].fillna("M", inplace=True)

    # Assigning corresponding mean value to prior replaced 0 numerical values...!
    df['Critic_Score'] = df['Critic_Score'].replace(0.0, round((np.mean(df['Critic_Score'])), 2))
    df['Critic_Count'] = df['Critic_Count'].replace(0.0, round((np.mean(df['Critic_Count'])), 2))
    df['User_Score'] = df['User_Score'].replace(0.0, round((np.mean(df['User_Score'])), 2))
    df['User_Count'] = df['User_Count'].replace(0.0, round((np.mean(df['User_Count'])), 2))

    return df