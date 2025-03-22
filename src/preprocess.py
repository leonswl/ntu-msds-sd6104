from rich.console import Console

import pandas as pd
from thefuzz import process, fuzz
from collections import defaultdict

console = Console()

def fuzzy_normalize_column(df, column_name, threshold=80):
    """
    Normalize text values in a DataFrame column using fuzzy matching.
    
    Args:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Column name to normalize.
    - threshold (int): Similarity threshold for fuzzy matching (default is 80).
    
    Returns:
    - pd.DataFrame: DataFrame with a new normalized column.
    """
    df[column_name] = df[column_name].astype(str).fillna('')  # Convert to string

    unique_values = list(set(df[column_name].str.lower()))  # Unique values in lowercase

    # Reference mapping for normalization
    reference_mapping = {}
    groups = defaultdict(list)  # To store word clusters

    for value in unique_values:
        # Check if it's already in a group
        if value in reference_mapping:
            continue
        
        # Find similar words
        matches = process.extract(value, unique_values, limit=10, scorer=fuzz.ratio)
        matches = [(match, score) for match, score in matches if score >= threshold]
        
        if matches:
            best_match = max(matches, key=lambda x: x[1])[0]  # Pick the best-scoring match
        else:
            best_match = value  # Keep original if no good match found

        # Assign all similar words to the best match
        for match, score in matches:
            reference_mapping[match] = best_match
            groups[best_match].append(match)

    # Apply normalization mapping
    df[f'{column_name}'] = df[column_name].str.lower().map(reference_mapping)
    
    return df


def preprocess(df):
    # renaming column names to snake_case
    COLUMN_NAMES = [
        'inspection_id',
        'dba_name',
        'aka_name',
        'license_',
        'facility_type',
        'risk',
        'address',
        'city',
        'state',
        'zip',
        'inspection_date',
        'inspection_type',
        'results',
        'violations',
        'latitude',
        'longitude',
        'location'
    ]

    df.columns = COLUMN_NAMES

    # drop irrelevant columns
    df.drop(['inspection_id', 'aka_name', 'location'], axis=1, inplace=True)

    # drop missing values
    df.dropna(subset=['city', 'state', 'zip', 'latitude', 'longitude'], inplace=True)

    # fix data type
    df = df.astype({'zip':'Int64', 'license_':'Int64'})

    # consolidate redundant values using fuzzy matching
    df = fuzzy_normalize_column(df, 'inspection_type', threshold=80).drop(['inspection_type'],axis=1)

    # drop columns after post processing 
    df.drop(['state'], axis=1, inplace=True)

    console.print("[bold green]SUCCESS[/bold green] File preprocessing completed.")
    console.print(df.head())

    return df
