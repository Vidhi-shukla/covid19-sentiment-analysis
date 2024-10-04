# Importing Necessary Libraries
import pandas as pd
from tabulate import tabulate
import re
import dask.dataframe as dd

# Load the combined data from the CSV file
data = pd.read_feather('C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/processed_data/combined_data.feather')

# Dropping columns
tweet = data.copy()
tweet.drop(['status_id','user_id','screen_name','source','reply_to_status_id','reply_to_user_id',
            'is_retweet','place_full_name','place_type','reply_to_screen_name','is_quote',
            'followers_count','friends_count','account_lang','account_created_at','verified'],axis=1, inplace = True)

# Filtering data with 'country_code = IN' and 'language = en'
tweet = tweet[(tweet.country_code == "IN") & (tweet.lang == "en")].reset_index(drop = True)
tweet.drop(['country_code','lang'],axis=1,inplace=True)

# Ensure all entries in the created_at column are strings and convert the time
tweet["created_at"] = tweet["created_at"].astype(str)
tweet["created_at"] = tweet["created_at"].apply(lambda i: (int(i.split("T")[1].split(":")[0]) + 
                                                           int(i.split("T")[1].split(":")[1])/60) if "T" in i else i)

#  Checking for missing values
missing_values = tweet.isna().sum()

# Data preprocessing
def preprocess_text(text):
    cleaned_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(#[A-Za-z0-9]+)", " ", text).split()).lower()
    return cleaned_text

# Applying the preprocessing to the 'text' column
tweet['text'] = tweet['text'].apply(preprocess_text)

# Saving the combined DataFrame to a CSV file
tweet.to_csv('C:/Users/Vidhi/Documents/PROJECTS/SENTIMENT ANALYSIS/data/processed_data/preprocessed_data.csv', index=False)


