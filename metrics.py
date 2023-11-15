# Importing packages
import numpy as np
import pandas as pd

def word_grouping(df):
    is_alnum = df['text_change'].str.contains('q')
    word_count = [0] * len(is_alnum)  # Initialize with zeros
    j = 0

    for i in range(len(is_alnum)):
        word_count[i] = j
        if i < len(is_alnum) - 1:
            if not is_alnum.iloc[i] and is_alnum.iloc[i + 1]:
                j += 1

    word_count[-1] = j  # Last element
    
    return pd.Series(word_count)  # Return as a Pandas Series


def calculate_features(df):
    """
    """
    # Create a DataFrame to store the features with a single column of IDs
    features = pd.DataFrame({'id': df['id'].unique()})

    
    # ----------- LONG PAUSE CALCULATIONS ------------
    iki = df.groupby('id')['down_time'].diff().fillna(0) #interkeystroke interval



    
    # Compute aggregated statistics
    mean_iki = iki.groupby(df['id']).mean().reset_index()
    mean_iki.columns = ['id', 'mean_iki']

    median_iki = iki.groupby(df['id']).median().reset_index()
    median_iki.columns = ['id', 'median_iki']

    std_iki = iki.groupby(df['id']).std().reset_index()
    std_iki.columns = ['id', 'std_iki']

    max_iki = iki.groupby(df['id']).max().reset_index()
    max_iki.columns = ['id', 'max_iki']

    # Merge with features DataFrame
    features = features.merge(mean_iki, on='id', how='left')
    features = features.merge(median_iki, on='id', how='left')
    features = features.merge(std_iki, on='id', how='left')
    features = features.merge(max_iki, on='id', how='left')


    df['word_count'] = word_grouping(df)

    # Calculate the difference in down_time within groups defined by both 'id' and 'word_count'
    df['down_time_diff'] = df.groupby(['id', 'word_count'])['down_time'].diff()
    # Filter out the rows where activity is 'Backspace' or any other non-letter activity
    df_filtered = df[df['activity'] == 'Input']
    # Calculate the mean difference in down_time within each word for each id
    mean_intra_word_iki = df_filtered.groupby(['id', 'word_count'])['down_time_diff'].mean().reset_index()
    # Aggregate this feature at the 'id' level to match the granularity of your features DataFrame
    mean_intra_word_iki = mean_intra_word_iki.groupby('id')['down_time_diff'].mean().reset_index(name='mean_intra_word_iki')
    
    # Calculate the standard deviation of down_time_diff within each word for each id
    std_intra_word_iki = df_filtered.groupby(['id', 'word_count'])['down_time_diff'].std().reset_index()
    # Aggregate this feature at the 'id' level to match the granularity of your features DataFrame
    std_intra_word_iki = std_intra_word_iki.groupby('id')['down_time_diff'].std().reset_index(name='std_intra_word_iki')
    
    #features = features.merge(mean_intra_word_iki, on='id', how='left')
    #features = features.merge(std_intra_word_iki, on='id', how='left')
    
    #mean_inter_word_iki
    #std_inter_word_iki

    #mean_time_between_words
    #std_time_between_words

    #mean_time_between_sentences
    #std_time_between_sentences
    
    #n_iki_1
    #n_iki_2
    #n_iki_3
    #n_iki_4
    #n_iki_5


    # Revision calcuations

    # Fluency calculations

    # Verbosity calculations

    # Non-typing event calculations


    return features
