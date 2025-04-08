import time

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def run_association_rule_mining(factorized_csv, min_support=0.05, min_confidence=0.6,
                                association_rules_csv='association_rules_output.csv',
                                frequent_itemsets_csv='frequent_itemsets.csv'):
    """
    Loads a factorized CSV file, prepares transactions, one-hot encodes the data,
    runs the apriori algorithm, and extracts association rules.
    
    Parameters:
      - factorized_csv: str, path to the factorized CSV file.
      - min_support: float, minimum support threshold for the apriori algorithm.
      - min_confidence: float, minimum confidence threshold for association rules.
      - association_rules_csv: str, path to export association rules results.
      - frequent_itemsets_csv: str, path to export frequent itemsets results.
      
    Returns:
      - frequent_itemsets: pd.DataFrame containing the frequent itemsets.
      - rules: pd.DataFrame containing the association rules.
    """
    # Load the factorized data
    df = pd.read_csv(factorized_csv)
    
    # -----------------------------------------------------------------------------
    # STEP 2: PREPARE TRANSACTIONS FROM ALL COLUMNS (WITHOUT REMOVING DUPLICATES)
    # -----------------------------------------------------------------------------
    transactions = [
        [f"{col}={row[col]}" for col in df.columns if pd.notna(row[col])]
        for _, row in df.iterrows()
    ]
    
    # -----------------------------------------------------------------------------
    # STEP 3: TRANSFORM TRANSACTIONS INTO ONE-HOT ENCODING
    # -----------------------------------------------------------------------------
    start_time = time.time()
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions, sparse=True)

    transactions_df = pd.DataFrame.sparse.from_spmatrix(te_array, columns=te.columns_)
    elapsed_time = time.time() - start_time
    print(f"Successfully transformed transactions into one-hot encoding in {elapsed_time:.2f} seconds..")
    
    # -----------------------------------------------------------------------------
    # STEP 4: RUN APRIORI TO FIND FREQUENT ITEMSETS
    # -----------------------------------------------------------------------------
    frequent_itemsets = apriori(transactions_df, min_support=min_support, use_colnames=True)
    print("Successfully completed apriori rule mining")
    
    # -----------------------------------------------------------------------------
    # STEP 5: EXTRACT ASSOCIATION RULES
    # -----------------------------------------------------------------------------
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    rules = rules.sort_values(by='confidence', ascending=False)
    
    # -----------------------------------------------------------------------------
    # STEP 6: INSPECT & EXPORT THE RESULTS
    # -----------------------------------------------------------------------------
    print("Frequent Itemsets (Top 10):")
    print(frequent_itemsets.head(10))
    
    print("\nAssociation Rules (Top 10 by Confidence):")
    print(rules.head(10))
    
    # Export the results to CSV files
    rules.to_csv(association_rules_csv, index=False)
    frequent_itemsets.to_csv(frequent_itemsets_csv, index=False)
    
    return frequent_itemsets, rules

# Example usage:
if __name__ == "__main__":
    factorized_csv = 'notebooks/eugeneho/final_assiocation_file_int.csv'
    run_association_rule_mining(factorized_csv)