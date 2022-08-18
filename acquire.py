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


'''Initial function to pull the ibm and opportunity atlas data.
If a merged .csv file is not already cached, then this function merges the two datasets and caches a .csv file to local directory.'''
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
    print(f'initial df shape: {df.shape}')

    return df