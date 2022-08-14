# notebook dependencies

# diasbling warning messages
import warnings
warnings.filterwarnings("ignore")

# importing dependencies
import os
import pandas as pd
import numpy as np

# plotting libraries/modules
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")



def get_equity_df():

    # check for a cached dataset
    filename = "equity_df.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)

    else:
        # importing the two (2) main dataframes (ibm attrition/opportunity atlas data)
        ibm_df = pd.read_csv("/Users/mijailmariano/Desktop/IBM_HR-Employee-Attrition.csv")
        equity_df = pd.read_csv("/Users/mijailmariano/Desktop/equity_table.csv")

        '''let's use a random sampler to create 1470 geographical location records of of the 
        - using Pandas' '.sample()' method wtih parameters 'replace' set to True to allow for duplicate records
        - resetting the index number
        - setting a random state for reproducibility'''

        sample_df = equity_df.sample(n = 1470, replace = True, ignore_index = True, random_state = 528)

        '''let's also reshuffle the ibm df for random assignment & suffling of the dataframe
        - resetting the index number (can use unique employee id for predictions/future indexing)
        - setting a random state for reproducibility'''

        ibm_shuffled = ibm_df.sample(n = 1470, replace = False, ignore_index = True, random_state = 528)
        
        # concatinating the two (2) dataframes
        df = pd.concat([ibm_shuffled, sample_df], axis = 1)

        # creating a cached file for future referencing
        df.to_csv("equity_df.csv")
        
    # let's print the shape too
    print(f'df shape: {df.shape}')

    return df



