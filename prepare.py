# importing dependencies
import os
import pandas as pd
import numpy as np

from skimpy import clean_columns
import random

# plotting libraries/modules
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


'''Function to randomly apply county based on the employee's distance from home'''
def get_county(x, lst_a, lst_b, lst_c, lst_d, lst_e):
        random.seed(548)
        '''where x = employees' work distance from home in miles. 
        function will iterate through all records and randomly assign a county based on distance from work.'''
        lst = []

        if x <= 5:
                county = random.choice(lst_a)
                lst.append(county)

        elif x > 5 and x <= 10:
                county = random.choice(lst_b)
                lst.append(county)

        elif x > 10 and x <= 21:
                county = random.choice(lst_c)
                lst.append(county)
        
        elif x > 27 and x <= 30:
                county = random.choice(lst_e)
                lst.append(county)

        else:
                county = lst_d[0]
                lst.append(county)

        # returning the list of counties
        return lst

'''Function that initial cleans the ibm/opportunity atlas merged dataframes'''
def clean_employee_df(df):
    # let's normalize the column names
    df = clean_columns(df)

    # setting employee number as the index for future referencing and attrition modeling/predictions
    df = df.set_index("employee_number").sort_index().rename_axis(None)
    
    # pulling needed variables for exploration/analysis
    df = df[[
            'attrition',
            'age',
            'monthly_income',
            'percent_salary_hike',
            'total_working_years',
            'training_times_last_year',
            'years_at_company',
            'household_income_at_35',
            'high_school_graduation_rate',
            'percentage_married_by_35',
            'incarceration_rate',
            'women_teenage_birthrate',
            'poverty_rate',
            'employment_rates_at_35yrs',
            'single_parent_frac',
            'years_since_last_promotion',
            'county_name',
            'department',
            'education',
            'education_field',
            'environment_satisfaction',
            'gender',
            'job_involvement',
            'job_level',
            'job_role',
            'job_satisfaction',
            'marital_status',
            'performance_rating',
            'relationship_satisfaction',
            'state',
            'stock_option_level',
            'work_life_balance',
            'years_in_current_role',
            'years_with_curr_manager']]

    # renaming necessary columns for clarity
    df = df.rename(columns = {"age": "employee_age"})

    # renaming/classifying attrition as boolean value types
    df["attrition"] = df["attrition"].replace({"Yes": True, "No": False})

    # converting continuous variables to discrete type
    disc_lst = [
    'stock_option_level',
    'work_life_balance',
    'education',
    'job_involvement',
    'job_level',
    'job_satisfaction',
    'performance_rating',
    'relationship_satisfaction',
    'county_name',
    'state',
    'department',
    'education_field',
    'gender',
    'job_role',
    'marital_status',
    'environment_satisfaction']

    # setting the data types
    df[disc_lst] = df[disc_lst].astype(object)

    # printing the new df shape
    print(f'shape after cleaning: {df.shape}')

    # lastly, return the dataframe
    return df


'''Function to clean outliers at upperbounds'''
def df_outliers(df):

    # monthly income / leadership or seniority
    df = df[df["monthly_income"] <= 16581.00]
    
    # length of working tenure
    df = df[df["total_working_years"] <= 28.00]

    # length of tenure at current company
    df = df[df["years_at_company"] <= 18.00]

    # number of years since last promotion
    df = df[df["years_since_last_promotion"] <= 7.50]

    # number of years in current role 
    df = df[df["years_in_current_role"] <= 14.50]

    # number of year with current manager
    df = df[df["years_with_curr_manager"] <= 14.50]

    # returning the cleaned dataset
    print(f'shape after outliers: {df.shape}')

    return df


'''Function created to split the initial dataset into train, validate, and test datsets'''
def train_validate_test_split(df):
    train_and_validate, test = train_test_split(
    df, test_size = 0.2, random_state = 548)
    
    train, validate = train_test_split(
        train_and_validate,
        test_size = 0.3,
        random_state = 548)

    print(f'train shape: {train.shape}')
    print(f'validate shape: {validate.shape}')
    print(f'test shape: {test.shape}')

    return train, validate, test

'''Function created to scale selected columns/continuous data for modeling'''
def scaled_data(df):
    # selecting features to scale
    scale_lst = df.select_dtypes(exclude = ["object", "uint8", "bool"]).columns.tolist()

    # creating a copy of the original zillow/dataframe
    df_scaled = df.copy()

    # created the standard scaler
    scaler = StandardScaler()

    # fit/learn from the selected columns
    scaler.fit(df_scaled[scale_lst])

    # apply/transform the data
    df_scaled[scale_lst] = scaler.transform(df_scaled[scale_lst])
    
    print(f'scaled df shape: {df_scaled.shape}')

    # returning newly created dataframe with scaled data
    return df_scaled

'''Function to create dummy variables for discrete variables/feature'''
def get_dummy_dataframes(train_df, val_df, test_df):

    # train dataset
    train_dummy = pd.get_dummies(
        data = train_df, 
        columns = [
                'job_level', 
                'job_role', 
                'marital_status', 
                'stock_option_level'],
        drop_first = False, 
        dtype = bool)

    # validate dataset
    validate_dummy = pd.get_dummies(
        data = val_df, 
        columns = [
                'job_level', 
                'job_role', 
                'marital_status', 
                'stock_option_level'],
        drop_first = False, 
        dtype = bool)

    # test dataset
    test_dummy = pd.get_dummies(
        data = test_df, 
        columns = [
                'job_level', 
                'job_role', 
                'marital_status', 
                'stock_option_level'],
        drop_first = False, 
        dtype = bool)

    # cleaning column names after dummy transformation
    train_dummy = clean_columns(train_dummy)
    validate_dummy = clean_columns(validate_dummy)
    test_dummy = clean_columns(test_dummy)

    # returning dummy datasets
    return train_dummy, validate_dummy, test_dummy


'''Function to create a single dummy variable dataframe'''
def get_dummy_df(df):
    dummy_df = pd.get_dummies(
                    data = df, 
                    columns = [
                            'job_level', 
                            'job_role', 
                            'marital_status', 
                            'stock_option_level'],
                            drop_first = False, 
                            dtype = bool)

    print(f'dummy df shape: {dummy_df.shape}')

    return dummy_df