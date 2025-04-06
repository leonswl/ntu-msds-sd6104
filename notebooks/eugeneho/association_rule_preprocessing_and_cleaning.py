import os
import pandas as pd

def clean_and_factorize_data(input_csv, output_factorized_csv, columns_to_remove, working_dir=None):
    """
    Loads a CSV file, removes unnecessary columns, factorizes each column (storing mappings),
    and exports the factorized DataFrame.
    
    Parameters:
      - input_csv: str, path to the input CSV file.
      - output_factorized_csv: str, path to save the factorized CSV file.
      - columns_to_remove: list of str, names of columns to remove.
      - working_dir: str or None, directory to change into before processing (optional).
    
    Returns:
      - df: pd.DataFrame, the factorized DataFrame.
      - mappings: dict, mapping for each column (code -> original value).
    """
    # Change working directory if provided
    if working_dir:
        os.chdir(working_dir)
    
    # Load the data
    df = pd.read_csv(input_csv)
    
    # Remove unwanted columns; ignore errors if a column isn't found
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    # Factorize each column and store the mapping
    mappings = {}
    for col in df.columns:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        mappings[col] = dict(enumerate(uniques))
    
    # Print results for inspection
    print("Factorized DataFrame:")
    print(df)
    print("\nMapping for each column:")
    print(mappings)
    
    # Export the factorized DataFrame
    df.to_csv(output_factorized_csv, index=False)
    
    return df, mappings

# Example usage:
input_csv = 'Food_Inspections_Violations_Expanded_with_cleandata_address.csv'
output_factorized_csv = 'final_assiocation_file_int.csv'
columns_to_remove = [
    'Inspection ID', 'AKA Name', 'Latitude', 'Longitude', 
    'raw_violation', 'violation_comment', 'parse_error', 'error_reason', 
    'Facility Type', 'City', 'Inspection Type', 'City Cleaned', 'State'
]
working_dir = 'C:/Users/geneh/OneDrive - A STAR/MSDS/SD6104'

df_factorized, mappings = clean_and_factorize_data(input_csv, output_factorized_csv, columns_to_remove, working_dir)

df_factorized.to_csv('final_assiocation_file_int.csv', index=False)


