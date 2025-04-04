# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 20:17:39 2025

@author: geneh
"""

# Preprocessing violations and remove comments
def split_violations(input_csv, output_csv, column_name='Violations', delimiter='|'):
    """
    Reads `input_csv`, splits rows in the specified `column_name` by `delimiter`,
    then writes the transformed data to `output_csv`—with each piece in a new row.
    Also removes anything after " - Comments:" in that column.
    """
    # 1. Read the input CSV into a pandas DataFrame.
    df = pd.read_csv(input_csv)

    # 2. Split the chosen column on the specified delimiter into a list of violations.
    df[column_name] = df[column_name].astype(str).str.split(delimiter)

    # 3. Explode the list so each element becomes its own row.
    df = df.explode(column_name)

    # 4. Split on " - Comments:" and keep the part before it.
    #    This way, everything after " - Comments:" is discarded.
    df[column_name] = df[column_name].apply(lambda x: x.split(" - Comments:")[0])

    # 5. Trim whitespace after removal.
    df[column_name] = df[column_name].str.strip()

    # 6. Write the result out to a new CSV file.
    df.to_csv(output_csv, index=False)

# Just call the function below:
input_file = "Food_Inspections_20250216.csv"
output_file = "Food_Inspections_20250216_split.csv"

split_violations(input_file, output_file, column_name="Violations", delimiter='|')
print(f"Done! Wrote split violations to: {output_file}")