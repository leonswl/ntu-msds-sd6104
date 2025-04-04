# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:03:20 2025

@author: geneh
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os

pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Change directory
os.chdir('C:/Users/geneh/OneDrive - A STAR/MSDS/SD6104')

# -----------------------------------------------------------------------------
# STEP 1: LOAD THE DATA
# -----------------------------------------------------------------------------
df = pd.read_csv('Food_Inspections_Violations_Expanded_with_cleandata.csv')

unique_counts = df.nunique()

#remove AKA NAME, Latitude, Longitude, raw_violation and violation_comments

columns_to_remove = ['Inspection ID','AKA Name', 'Latitude', 'Longitude', 
                     'raw_violation', 'violation_comment', 'parse_error', 'error_reason', 
                     'Facility Type', 'City', 'Inspection Type', 'City Cleaned', 'State'] 

df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

df.to_csv('final_assiocation_file.csv', index=False)

unique_counts = df.nunique()

df_sample = df.sample(frac=0.4, random_state=42)
print("Sampled DataFrame shape:", df_sample.shape)

# -----------------------------------------------------------------------------
# STEP 2: PREPARE TRANSACTIONS FROM ALL COLUMNS (WITHOUT REMOVING DUPLICATES)
# -----------------------------------------------------------------------------
transactions = []
for _, row in df_sample.iterrows():
    row_items = []
    # Loop through every column in the sampled DataFrame
    for col in df_sample.columns:
        if pd.notna(row[col]):
            row_items.append(f"{col}={row[col]}")
    transactions.append(row_items)

# -----------------------------------------------------------------------------
# STEP 3: TRANSFORM TRANSACTIONS INTO ONE-HOT ENCODING
# -----------------------------------------------------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
transactions_df = pd.DataFrame(te_array, columns=te.columns_)

# -----------------------------------------------------------------------------
# STEP 4: RUN APRIORI TO FIND FREQUENT ITEMSETS
# -----------------------------------------------------------------------------
# Adjust min_support as needed; note that with a smaller sample, you may need to lower support
frequent_itemsets = apriori(transactions_df, min_support=0.05, use_colnames=True)

# -----------------------------------------------------------------------------
# STEP 5: EXTRACT ASSOCIATION RULES
# -----------------------------------------------------------------------------
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
rules = rules.sort_values(by='confidence', ascending=False)

# -----------------------------------------------------------------------------
# STEP 6: INSPECT & EXPORT THE RESULTS
# -----------------------------------------------------------------------------
print("Frequent Itemsets (Top 10):")
print(frequent_itemsets.head(10))

print("\nAssociation Rules (Top 10 by Confidence):")
print(rules.head(10))

# Optionally, export the results to CSV files
rules.to_csv('association_rules_output.csv', index=False)
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)


















