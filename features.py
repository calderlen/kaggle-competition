# Importing Packages
import numpy as np
import pandas as pd


def word_grouping(df):
    """
    Group words in a DataFrame based on the 'activity' column.

    Parameters:
    df (DataFrame): The input DataFrame containing the 'activity' column.

    Returns:
    DataFrame: The modified DataFrame with additional columns 'word_begin' and 'word_end' indicating the boundaries of words.
    """
    
    # Initialize columns for word beginnings and endings
    df['word_begin'] = 0
    df['word_end'] = 0
    
    # Shifting the activity columns up and down one for subsequent calculations
    shifted_activity_prev = df['activity'].shift(1)
    shifted_activity_next = df['activity'].shift(-1)
    
    # Identification of word boundaries
    df['word_begin'] = ((df['activity'] == 'Input') & (shifted_activity_prev != 'Input')).astype(int)
    df['word_end'] = ((df['activity'] == 'Input') & (shifted_activity_next != 'Input')).astype(int)
    
    # Handling edge cases: adressing first and last column of datafraem
    df.at[0, 'word_begin'] = int(df.iloc[0]['activity'] == 'Input')
    df.at[df.index[-1], 'word_end'] = int(df.iloc[-1]['activity'] == 'Input')
    
    return df


def iki_core(df):
    # Calculate IKI for all events
    df['iki'] = df['down_time'].diff().fillna(0)

    # Initialize columns for intra-word IKI and inter-word IKI with NaN
    df['intra_word_iki'] = np.nan
    df['inter_word_iki'] = np.nan

    # Identify the start and end of words
    word_starts = df['word_begin'] == 1
    word_ends = df['word_end'] == 1

    # Compute intra-word and inter-word IKI
    df.loc[word_starts, 'inter_word_iki'] = df.loc[word_starts, 'iki']
    df.loc[~word_starts & ~word_ends, 'intra_word_iki'] = df.loc[~word_starts & ~word_ends, 'iki']

    return df



def iki_features(df):
    """
    Compute various interkeystroke interval (IKI) features from a given DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing keystroke data.

    Returns:
    features (DataFrame): DataFrame containing computed IKI features.
    """
    
    df = word_grouping(df)
    df = iki_core(df)
    
    # BASIC FEATURES
    # Create a DataFrame to store the features with a single column of IDs
    features = pd.DataFrame({'id': df['id'].unique()})
    
    # Computing median, standard deviation, and maximum IKI, intra-word IKI, and inter-word IKI

    agg_functions = ['median', 'std', 'max']
    iki_basics = df.groupby('id')['iki'].agg(agg_functions).reset_index()
    intra_word_iki_basics = df.groupby('id')['intra_word_iki'].agg(agg_functions).reset_index()
    inter_word_iki_basics = df.groupby('id')['inter_word_iki'].agg(agg_functions).reset_index()

    # Renaming the columns
    iki_basics_columns = ['id'] + [f'iki_{f}' for f in agg_functions]
    intra_word_iki_basics_columns = ['id'] + [f'intra_word_iki_{f}' for f in agg_functions]
    inter_word_iki_basics_columns = ['id'] + [f'inter_word_iki_{f}' for f in agg_functions]

    # Merging features
    features = features.merge(iki_basics, on='id')
    features = features.merge(intra_word_iki_basics, on='id')
    features = features.merge(inter_word_iki_basics, on='id')

    return features