import os
import argparse

from rich.console import Console
import pandas as pd

from src.preprocess import preprocess
from src.fds import (
    convert_fd,
    convert_ind,
    find_fds,
    find_afds,
    find_inds,
    find_ainds,
    FunctionalDependencySet,
    InclusionDependencySet
)


console = Console()

print("hahaha")

def create_arg_parser():
    """
    Creates and returns an argument parser.
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Data preparation software for SD6104."
    )

    # Argument flag for preprocessing
    parser.add_argument(
        '-np',
        '--no-preprocess',
        action='store_true',
        required=False,
        help="Specify this flag to perform preprocessing on the dataset."
    )

    # Argument flag for profilling
    parser.add_argument(
        '-s',
        '--single-profile',
        action='store_true',
        required=False,
        help="Specify this flag to perform single-column profilling."
    )

    # Argument flag for association rule mining
    parser.add_argument(
        '-rm',
        '--rule-mining',
        action='store_true',
        required=False,
        help="Specify this flag to perform association rule mining."
    )

    # Argument for functional dependencies
    parser.add_argument(
        '-fd',
        '--func-dependencies',
        choices=['all', 'default', 'approximate'],
        nargs='?',
        default=None,
        required=False,
        type=str,
        help="Specify the method for functional dependencies: 'default' or 'approximate'."
    )

    # Argument for inclusion dependencies
    parser.add_argument(
        '-ind',
        '--ind-dependencies',
        choices=['all', 'default', 'approximate'],
        nargs='?',
        default=None,
        required=False,
        type=str,
        help="Specify the method for inclusion dependencies: 'default' or 'approximate'."
    )

    return parser


def load_data():
    file_path = "data/Food_Inspections_20250216.csv"
    df = pd.read_csv(file_path, header=[0])
    console.log(f"[bold green]SUCCESS[/bold green] File loaded from {file_path}")
    console.log(df.head())

    return df

def save_data(df):
    file_path = "data/Food_Inspections_20250216_preprocessed.parquet"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save as Parquet
    df.to_parquet(file_path, index=False)

    console.log(f"[bold green]SUCCESS[/bold green] File persisted to {file_path}")

def discover_fds(df):
    results = find_fds(df)

    console.log(f"There are {len(results)} functional dependencies using Default algorithm.")

    for fd in results:
        console.log(fd)

    fd_set = FunctionalDependencySet()
    for result in results:
        lhs, rhs =  convert_fd(fd=result)
        fd_set.add_dependency(lhs, rhs)

    # Validate all dependencies and store results
    fd_set.validate_fd(df)
    console.log(f"There are {len(fd_set)} fds in the dataset.")

    # Retrieve all validation results
    all_results = fd_set.get_all_validation_results()
    for (lhs, rhs), result in all_results.items():
        # Create a copy of result without the 'highlights' key
        filtered_result = {key: value for key, value in result.items() if key != "highlights"}
        
        console.log(f"FD: {lhs} -> {rhs}, Results: {filtered_result}")

    return all_results

def discover_afds(df, error):
    results = find_afds(df, error)

    console.log(f"There are {len(results)} functional dependencies using Default algorithm.")

    for fd in results:
        console.log(fd)

    fd_set = FunctionalDependencySet()
    for result in results:
        lhs, rhs =  convert_fd(fd=result)
        fd_set.add_dependency(lhs, rhs)

    # Validate all dependencies and store results
    fd_set.validate_afd(df)
    console.log(f"There are {len(fd_set)} fds in the dataset.")

    # Retrieve all validation results
    all_results = fd_set.get_all_validation_results()
    for (lhs, rhs), result in all_results.items():
        # Create a copy of result without the 'highlights' key
        filtered_result = {key: value for key, value in result.items() if key != "highlights"}
        
        console.log(f"FD: {lhs} -> {rhs}, Results: {filtered_result}")

    return all_results

def run_ind(df):
    results = find_inds([df, df])

    console.log(f"There are {len(results)} inclusion dependencies using Default algorithm.")

    for ind in results:
        console.log(ind)

    ind_set = InclusionDependencySet()
    for result in results:
        lhs, rhs =  convert_ind(result)
        ind_set.add_dependency(lhs, rhs)

    # Validate all dependencies
    ind_set.validate_ind(df)

def run_aind(df, error):
    results = find_ainds([df,df], error = error)

    console.log(f"There are {len(results)} inclusion dependencies using Default algorithm.")

    for ind in results:
        console.log(ind)

    ind_set = InclusionDependencySet()
    for result in results:
        lhs, rhs =  convert_ind(result)
        ind_set.add_dependency(lhs, rhs)

    # Validate all dependencies
    ind_set.validate_aind(df)

def expand_violations(df, save_path=None):
    """
    Parses and expands the 'Violations' column in the DataFrame.

    Parameters:
    - df: pandas DataFrame with a 'Violations' column
    - save_path: Optional string. If provided, saves the expanded DataFrame to this CSV path.

    Returns:
    - violations_expanded_df: A new DataFrame with extracted violation details and parsing metadata
    """

import pandas as pd
import re
def expand_violations(df, save_path=None):
    """
    Parses and expands the 'Violations' column in the DataFrame.

    Parameters:
    - df: pandas DataFrame with a 'Violations' column
    - save_path: Optional string. If provided, saves the expanded DataFrame to this CSV path.

    Returns:
    - violations_expanded_df: A new DataFrame with extracted violation details and parsing metadata
    """

    def extract_violations(row):
        violation_text = row.get('Violations', '')
        if pd.isna(violation_text):
            return []

        parts = [v.strip() for v in violation_text.split('|') if v.strip()]
        pattern = r"(?P<number>\d+)\.\s+(?P<text>.+?)\s+-\s+Comments:\s+(?P<comment>.+)"

        extracted = []
        for part in parts:
            match = re.match(pattern, part)
            combined = row.drop(labels=['Violations']).to_dict()
            combined['raw_violation'] = part

            if match:
                v = match.groupdict()
                combined['violation_number'] = v['number']
                combined['violation_text'] = v['text']
                combined['violation_comment'] = v['comment']
                combined['parse_error'] = False
                combined['error_reason'] = ""
            else:
                # Attempt to detect error reason
                if not re.search(r"\d+\.", part):
                    reason = "missing violation number"
                elif "Comments:" not in part:
                    reason = "missing 'Comments:'"
                else:
                    reason = "general format mismatch"
                
                combined['violation_number'] = None
                combined['violation_text'] = None
                combined['violation_comment'] = None
                combined['parse_error'] = True
                combined['error_reason'] = reason

            extracted.append(combined)
        return extracted

    # Apply extraction across all rows
    expanded_rows = []
    for _, row in df.iterrows():
        expanded_rows.extend(extract_violations(row))

    violations_expanded_df = pd.DataFrame(expanded_rows)

    if save_path:
        violations_expanded_df.to_csv(save_path, index=False)
        print(f"✅ Done! File saved as: {save_path}")

    return violations_expanded_df

# Example usage:
# df = pd.read_csv("Food_Inspections_20250216.csv")
# violations_df = expand_violations(df, save_path="Food_Inspections_Violations_Expanded.csv")

import pandas as pd
from fuzzywuzzy import process

def fuzzy_clean_columns(df, columns_to_clean, save_path=None, similarity_threshold=95):
    """
    Cleans categorical string columns in a DataFrame using fuzzy matching.

    Parameters:
    - df: pandas DataFrame
    - columns_to_clean: list of column names to clean
    - save_path: optional file path to save cleaned DataFrame as CSV
    - similarity_threshold: similarity score (0–100) for grouping values

    Returns:
    - cleaned DataFrame with additional *_Cleaned columns
    - dictionary with grouped original values under each canonical label
    """
    all_grouped_labels = {}

    def clean_column_fuzzy(df, column_name):
        print(f"\n🔍 Processing column: {column_name}")
        
        # Step 1: Standardize values
        df[column_name] = df[column_name].astype(str).str.strip().str.upper()

        # Step 2: Get unique values
        unique_values = df[column_name].dropna().unique().tolist()

        # Step 3: Fuzzy group
        grouped_map = {}     # raw -> cleaned
        grouped_labels = {}  # cleaned -> [originals]

        for value in unique_values:
            if value in grouped_map:
                continue

            matches = process.extract(value, unique_values, limit=None)
            close_matches = [m for m, score in matches if score >= similarity_threshold]

            canonical = close_matches[0]

            for match in close_matches:
                grouped_map[match] = canonical

            grouped_labels[canonical] = close_matches

        # Step 4: Add new column
        cleaned_col = f"{column_name} Cleaned"
        df[cleaned_col] = df[column_name].map(grouped_map)

        # Save grouping for display
        all_grouped_labels[column_name] = grouped_labels

        return df

    # Apply to all target columns
    for col in columns_to_clean:
        df = clean_column_fuzzy(df, col)

    # Optionally save result
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n✅ Cleaned data saved as: {save_path}")

    # Automatically print grouped labels
    print("\n🧾 All Grouped Labels:")
   # Show grouping results
    for col, group_map in all_grouped_labels.items():
        print(f"\n📦 Grouped values for '{col}':")
        for canonical, group in group_map.items():
            if len(group) > 1:
                print(f"\n  → {canonical}:")
                for g in group:
                    print(f"     - {g}")

    return df, all_grouped_labels

# Example usage:
# df = pd.read_csv("Food_Inspections_20250216.csv")
# columns = ['Facility Type', 'City', 'Inspection Type']
# cleaned_df, grouped_info = fuzzy_clean_columns(df, columns, save_path="Food_Inspections_Cleaned.csv")


def main(args):

    df = load_data()

    ##### PREPROCESSING ##### WANG YU
    if args.process_violations:
        console.log("Running preprocessing on violations column:")
        violations_df = expand_violations(df, save_path="Food_Inspections_Violations_Expanded.csv")
        #onsole.log("Persisting preprocessed file:")
        #save_data(df)

    if args.fuzzy_clean_columns:
        console.log("Running fuzzy cleaning:")
        columns = args.columns if args.columns else []  # fallback if None
        cleaned_df, grouped_info = fuzzy_clean_columns(df, columns, save_path="Food_Inspections_Cleaned.csv")
        #console.log("Persisting preprocessed file:")
        #save_data(df)

    #### SINGLE PROFILLING ##### SELENE
    if args.single_profile:
        # single_profilling()
        console.log("Running Single Profilling")


    #### RULE MINING ##### EUGENE
    if args.rule_mining:
        # rule_mining()
        console.log("Running Rule Mining")

    ##### FUNCTIONAL DEPENDENCIES #####
    if args.func_dependencies:
        if args.func_dependencies == 'default':
            console.log("Running both Default Functional Dependences:")
            fd_results = discover_fds(df)

        elif args.func_dependencies == 'approximate':
            console.log("Running both Approximate Functional Dependences:")
            afd_results = discover_afds(df=df, error=0.05)

        elif args.func_dependencies == 'all':
            console.log("Running both Default and Approximate Functional Dependences:")
            fd_results = discover_fds(df)
            afd_results = discover_afds(df=df, error=0.05)

    ##### INCLUSION DEPENDENCIES #####
    if args.ind_dependencies:

        if args.ind_dependencies == 'default':
            console.log("Running the Default Inclusion Dependences:")
            run_ind(df)

        elif args.ind_dependencies == 'approximate':
            console.log("Running Approximate Inclusion Dependences")
            run_aind(df=df, error=0.2)

        elif args.ind_dependencies == 'all':
            console.log("Running both Default and Approximate Inclusion Dependences:")
            run_ind(df)
            run_aind(df=df, error=0.2)
        

if __name__ == "__main__":

    parser = create_arg_parser()

    args = parser.parse_args()

    console.log(f"Args: {args}")

    main(args)