import pandas as pd
from efficient_apriori import apriori
import time

def run_efficient_apriori(factorized_csv, min_support=0.05, min_confidence=0.6,
                          association_rules_csv='association_rules_output.csv',
                          frequent_itemsets_csv='frequent_itemsets.csv'):
    """
    Loads a factorized CSV file, prepares transactions,
    runs efficient-apriori, and extracts association rules.
    """
    # Load the data
    df = pd.read_csv(factorized_csv)
    
    # STEP 1: PREPARE TRANSACTIONS FROM COLUMN=VALUE STRINGS (like your previous approach)
    transactions = [
        tuple(f"{col}={row[col]}" for col in df.columns if pd.notna(row[col]))
        for _, row in df.iterrows()
    ]
    
    # STEP 2: RUN EFFICIENT-APRIORI
    print("Running efficient-apriori...")
    start = time.time()
    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    end = time.time()
    print(f"Apriori completed in {end - start:.2f} seconds.")
    
    # STEP 3: FORMAT FREQUENT ITEMSETS INTO A DATAFRAME
    # itemsets is a dict of dicts: {1: {('a',): 0.5, ...}, 2: {...}, ...}
    itemset_rows = []
    for k, itemset_dict in itemsets.items():
        for items, support in itemset_dict.items():
            itemset_rows.append({
                'itemsets': frozenset(items),
                'support': support
            })
    frequent_itemsets_df = pd.DataFrame(itemset_rows)
    frequent_itemsets_df = frequent_itemsets_df.sort_values(by='support', ascending=False)

    print("Frequent Itemsets (Top 10):")
    print(frequent_itemsets_df.head(10))
    
    # STEP 4: FORMAT ASSOCIATION RULES INTO A DATAFRAME
    rules_rows = []
    for rule in rules:
        rules_rows.append({
            'antecedents': frozenset(rule.lhs),
            'consequents': frozenset(rule.rhs),
            'support': rule.support,
            'confidence': rule.confidence,
            'lift': rule.lift
        })
    rules_df = pd.DataFrame(rules_rows)
    rules_df = rules_df.sort_values(by='confidence', ascending=False)

    print("\nAssociation Rules (Top 10 by Confidence):")
    print(rules_df.head(10))
    
    # Export CSVs
    frequent_itemsets_df.to_csv(frequent_itemsets_csv, index=False)
    rules_df.to_csv(association_rules_csv, index=False)

    return frequent_itemsets_df, rules_df

# Example usage
if __name__ == "__main__":
    run_efficient_apriori(
        factorized_csv='notebooks/eugeneho/final_assiocation_file_int.csv',
        min_support=0.05,
        min_confidence=0.6
    )
