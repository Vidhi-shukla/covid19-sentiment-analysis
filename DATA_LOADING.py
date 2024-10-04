# Importing Necessary Libraries
import pandas as pd
from tabulate import tabulate
import re

# List of files in chronological order
file_paths = [
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-03-29 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-03-30 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-03-31 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-01 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-02 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-04 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-05 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-06 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-07 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-08 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-09 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-10 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-11 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-12 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-13 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-14 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-15 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-16 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-17 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-18 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-19 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-20 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-21 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-22 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-23 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-24 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-25 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-26 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-27 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-28 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-29 Coronavirus Tweets.CSV',
    'C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/raw/2020-04-30 Coronavirus Tweets.CSV',
]

# Combine the datasets
combined_data = pd.concat((pd.read_csv(file, encoding='ISO-8859-1') for file in file_paths), ignore_index=True)

# Saving the combined data to a new CSV file
combined_data.to_csv('C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/processed_data/combined_data.csv', index=False, encoding='utf-8')
combined_data.to_feather('C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/processed_data/combined_data.feather')

# Display the shape of the combined dataset
combined_data_shape = combined_data.shape
combined_data_head = combined_data.head()
shape_table = [["Number of Rows", combined_data_shape[0]], ["Number of Columns", combined_data_shape[1]]]
shape_table_str = tabulate(shape_table, headers=['Dimension', 'Count'], tablefmt='fancy_grid', showindex='True')
print(shape_table_str)

# Display the first few rows of dataset
df_table = tabulate(combined_data.head(), headers='keys', tablefmt='fancy_grid', showindex='False')
print(df_table)


