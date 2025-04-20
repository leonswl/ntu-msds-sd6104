import os
import pandas as pd
from efficient_apriori import apriori

def clean_and_factorize_data(df, columns_to_remove):

    start_time = time.time()
    
    #df = pd.read_csv(input_csv)
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    mappings = {}
    for col in df.columns:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        mappings[col] = dict(enumerate(uniques))

    print(f"✅ Factorization complete")
    return df, mappings

def run_efficient_apriori_from_df(df, min_support=0.05, min_confidence=0.6):
    transactions = [
        tuple(f"{col}={row[col]}" for col in df.columns if pd.notna(row[col]))
        for _, row in df.iterrows()
    ]

    print("\n⏳ Running efficient-apriori...")
    start = time.time()
    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    print(f"✅ Apriori completed")

    # Frequent itemsets
    itemset_rows = []
    for k, itemset_dict in itemsets.items():
        for items, support in itemset_dict.items():
            itemset_rows.append({
                'itemsets': frozenset(items),
                'support': support
            })
    frequent_itemsets_df = pd.DataFrame(itemset_rows).sort_values(by='support', ascending=False)
    print("\nFrequent Itemsets (Top 30):")
    print(frequent_itemsets_df.head(30))

    # Association rules
    rules_rows = []
    for rule in rules:
        rules_rows.append({
            'antecedents': frozenset(rule.lhs),
            'consequents': frozenset(rule.rhs),
            'support': rule.support,
            'confidence': rule.confidence,
            'lift': rule.lift
        })
    rules_df = pd.DataFrame(rules_rows).sort_values(by=['confidence', 'support'], ascending=[False, False])
    print("\nAssociation Rules (Top 30 by Confidence (with Support as Tie-Breaker)):")
    print(rules_df.head(30))

    return frequent_itemsets_df, rules_df

if __name__ == "__main__":
    input_csv = '../data/Food_Inspections_Violations_Expanded_with_cleandata_address.csv'
    columns_to_remove = [
        'Inspection ID', 'AKA Name', 'Latitude', 'Longitude', 
        'raw_violation', 'violation_comment', 'parse_error', 'error_reason', 
        'Facility Type', 'City', 'Inspection Type', 'City Cleaned', 'State'
    ]
    
    print("🔍 Step 1: Cleaning and Factorizing Data")
    df_factorized, mappings = clean_and_factorize_data(input_csv, columns_to_remove)

    print("\n🔍 Step 2: Running Association Rule Mining")
    run_efficient_apriori_from_df(
        df=df_factorized,
        min_support=0.05,
        min_confidence=0.6
    )
