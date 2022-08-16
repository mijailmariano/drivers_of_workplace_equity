# notebook dependencies

# diasbling warning messages
import warnings
warnings.filterwarnings("ignore")

# importing dependencies
import os
import pandas as pd
import numpy as np

from skimpy import clean_columns
import random

# plotting libraries/modules
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")

from sklearn.model_selection import train_test_split


# creating a function to randomly apply county based on the employee's distance from home
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


def get_employee_df():
    random.seed(548)
    # check for a cached dataset
    filename = "emp_df.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)

    else:
        # importing the two (2) main dataframes (ibm attrition/opportunity atlas data)
        df1 = pd.read_csv("/Users/mijailmariano/Desktop/IBM_HR-Employee-Attrition.csv")
        df2 = pd.read_csv("/Users/mijailmariano/Desktop/equity_table.csv")

        df2["distance"] = df2["distance"].str.replace("miles", "").astype(int)
        
        # creating the county bins
        area_one = df2[df2["distance"] <= 5].county_name.tolist()
        area_two = df2[(df2["distance"] > 5) & (df2["distance"] <= 10)].county_name.tolist()
        area_three = df2[(df2["distance"] > 10) & (df2["distance"] <= 21)].county_name.tolist()
        area_four = df2[(df2["distance"] > 21) & (df2["distance"] <= 27)].county_name.tolist()
        area_five = df2[(df2["distance"] > 27) & (df2["distance"] <= 30)].county_name.tolist()

        # applying the 'get_county()' function to ibm employee df
        county_lst = df1["DistanceFromHome"].apply(get_county, args = (area_one, area_two, area_three, area_four, area_five))
        
        # let's flatten the county list
        county_lst = [val for sublist in county_lst for val in sublist]
        county_lst = pd.Series(county_lst)

        # assing the series to the ibm dataframe
        df1["county_name"] = county_lst

        # merging the two dataframes and dropping unneeded columns
        df = df1.merge(
                        df2,
                        how = "left",
                        left_on = "county_name",
                        right_on = "county_name"
                    ).drop(columns = "distance")

        # creating a cached file for future referencing
        df.to_csv("emp_df.csv")

    # let's print the shape too
    print(f'df shape: {df.shape}')

    return df


def clean_employee_df(df):
    # let's normalize the column names
    df = clean_columns(df)
    
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

    # removing the following features/columns as they appear to be redundant
    df = df.drop(columns = ["over_18", "employee_count"])
    
    # setting employee number as the index for future referencing and attrition modeling/predictions
    df = df.set_index("employee_number").sort_index().rename_axis(None)

    # printing the new df shape
    print(f'df shape: {df.shape}')

    # lastly, return the dataframe
    return df


'''creating a function to clean outliers at upperbounds'''
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
    print(f'dataframe shape: {df.shape}')

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
