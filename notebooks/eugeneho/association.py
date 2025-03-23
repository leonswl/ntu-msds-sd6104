# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 09:20:23 2025

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
df = pd.read_csv('Food_Inspections_20250216_split.csv')

# Downsample the Data
#df = df.sample(frac=0.7, random_state=42)

# -----------------------------------------------------------------------------
# STEP 2: SELECT & PREPARE COLUMNS FOR ASSOCIATION MINING
# -----------------------------------------------------------------------------
transactions = []
for _, row in df.iterrows():
    row_items = []
    
    if pd.notna(row['DBA Name']):
        row_items.append(f"DBAName={row['DBA Name']}")
    if pd.notna(row['Risk']):
        row_items.append(f"Risk={row['Risk']}")
    if pd.notna(row['Zip']):
        row_items.append(f"Zip={row['Zip']}")
    if pd.notna(row['Inspection Type']):
        row_items.append(f"InspectionType={row['Inspection Type']}")
    if pd.notna(row['Results']):
        row_items.append(f"Results={row['Results']}")
    if pd.notna(row['Violations']):
        row_items.append(f"Results={row['Violations']}")

    # Remove duplicates and append
    transactions.append(list(set(row_items)))

# -----------------------------------------------------------------------------
# STEP 3: TRANSFORM TRANSACTIONS INTO ONE-HOT ENCODING
# -----------------------------------------------------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
transactions_df = pd.DataFrame(te_array, columns=te.columns_)
#transactions_df.to_csv('transactions_test_test.csv', index=False)

# -----------------------------------------------------------------------------
# STEP 4: RUN APRIORI TO FIND FREQUENT ITEMSETS
# -----------------------------------------------------------------------------
# Set min_support to threshold
frequent_itemsets = apriori(transactions_df, min_support=0.2, use_colnames=True)

# -----------------------------------------------------------------------------
# STEP 5: EXTRACT ASSOCIATION RULES
# -----------------------------------------------------------------------------
# Use confidence as the metric with a min_threshold
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)

# Sort rules by confidence (optional)
rules = rules.sort_values(by='confidence', ascending=False)

# -----------------------------------------------------------------------------
# STEP 6: INSPECT THE RESULTS
# -----------------------------------------------------------------------------
print("Frequent Itemsets (Top 10):")
print(frequent_itemsets.head(10))

print("\nAssociation Rules (Top 10 by Confidence):")
print(rules.head(10))

# Optionally export the rules to a CSV
rules.to_csv('association_rules_output.csv', index=False)
frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)
