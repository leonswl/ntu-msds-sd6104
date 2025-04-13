import os
import argparse

from rich.console import Console
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  


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
from src.data_profiler import DataProfiler



console = Console()

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

def single_profiling(
    df: pd.DataFrame,
    out_dir: str = "output/profile",     
    numeric_bins: int = 10,
    top_n: int = 20,
    max_text_unique: int = 200,
) -> None:
    """
    • Prints a DataFrame with column‑level metrics (no CSV written)
    • Saves a first‑digit bar chart for every *numeric* column
    • Saves two bar charts for every *text* column:
        - full range  (only if #unique ≤ max_text_unique)
        - top-N + 'Others'
    All PNGs land in *out_dir* (created if missing).
    """
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    profiler = DataProfiler(df)

    # Profile table – print via Rich
    profile_df = profiler.profile_dataframe(numeric_bins=numeric_bins)
    console.rule("[bold green]Column‑level profile[/bold green]")
    console.print(profile_df)

    # Numeric columns – first‑digit plots
    for col, row in profile_df.iterrows():
        fd_dist = row.get("first_digit_distribution", {})
        if isinstance(fd_dist, dict) and fd_dist:
            plt.figure(figsize=(5, 3))
            profiler.plot_first_digit(col)
            plt.savefig(out_path / f"{col}_first_digit.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            console.log(f"   • First‑digit plot saved for '{col}'")

    # Text columns – frequency plots
    for col in df.columns:
        # Skip numeric columns already handled
        if col in profile_df and profile_df.loc[col].get("first_digit_distribution"):
            continue

        unique_vals = int(profile_df.loc[col, "distinct_count"])
        if unique_vals == 0:
            continue

        # 3a. full‑range plot (only if reasonable)
        if unique_vals <= max_text_unique:
            plt.figure(figsize=(max(6, unique_vals * 0.25), 4))
            profiler.plot_text_frequency(col, top_n=unique_vals, show_pct=False)
            plt.savefig(out_path / f"{col}_full.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            console.log(f"   • Full‑range plot saved for '{col}' "
                        f"({unique_vals} unique values)")
        else:
            console.log(
                f"   • Skipped full‑range plot for '{col}' "
                f"({unique_vals} unique > {max_text_unique})"
            )

        # 3b. top‑N (+ Others) plot – always
        plt.figure(figsize=(10, 4))
        profiler.plot_text_frequency(col, top_n=top_n, show_pct=False)
        plt.savefig(out_path / f"{col}_top{top_n}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        console.log(f"   • Top‑{top_n} plot saved for '{col}'")
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
from thefuzz import process, fuzz
from collections import defaultdict

def fuzzy_clean_columns(df, columns_to_clean, threshold=80, save_path=None):
    """
    Normalizes string columns in a DataFrame using fuzzy matching (via thefuzz).

    Parameters:
    - df: pandas DataFrame
    - columns_to_clean: list of column names to normalize
    - threshold: similarity score for fuzzy grouping (default is 80)
    - save_path: optional file path to save cleaned DataFrame as CSV

    Returns:
    - df: DataFrame with new *_normalised columns
    - all_grouped_labels: dictionary mapping canonical → [grouped originals]
    """
    all_grouped_labels = {}

    def fuzzy_normalize_column(df, column_name, threshold=80):
        print(f"\n🔍 Processing column: {column_name}")
        
        # Ensure strings and clean format
        df[column_name] = df[column_name].astype(str).fillna('').str.strip().str.lower()

        unique_values = list(set(df[column_name]))

        reference_mapping = {}
        groups = defaultdict(list)

        for value in unique_values:
            if value in reference_mapping:
                continue

            matches = process.extract(value, unique_values, limit=10, scorer=fuzz.ratio)
            matches = [(match, score) for match, score in matches if score >= threshold]

            best_match = max(matches, key=lambda x: x[1])[0] if matches else value

            for match, score in matches:
                reference_mapping[match] = best_match
                groups[best_match].append(match)

        # Apply mapping
        normalised_col = f"{column_name}_normalised"
        df[normalised_col] = df[column_name].map(reference_mapping)

        # Store groups for printing
        all_grouped_labels[column_name] = dict(groups)

        return df

    # Process each column
    for col in columns_to_clean:
        df = fuzzy_normalize_column(df, col, threshold)

    # Save to CSV if needed
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n✅ Cleaned data saved as: {save_path}")

    # Print grouped results
    print("\n🧾 All Grouped Labels:")
    for col, group_map in all_grouped_labels.items():
        print(f"\n📦 Grouped values for '{col}':")
        for canonical, originals in group_map.items():
            if len(originals) > 1:
                print(f"\n  → {canonical}:")
                for original in originals:
                    print(f"     - {original}")

    return df, all_grouped_labels

# Example usage:
# df = pd.read_csv("Food_Inspections_20250216.csv")
# columns = ['Facility Type', 'City', 'Inspection Type']
# cleaned_df, grouped_info = fuzzy_clean_columns(df, columns, save_path="Food_Inspections_Cleaned.csv")

import pandas as pd
from thefuzz import process, fuzz
from collections import defaultdict
import re

def fuzzy_clean_address(df, columns_to_clean, threshold=80, save_path=None):
    all_grouped_labels = {}

    def contains_numeric_range(value):
        # Checks if the string contains patterns like 3443-3445
        return bool(re.search(r'\b\d{1,5}-\d{1,5}\b', value))

    def tokenize_address(value):
        tokens = value.strip().lower().split()
        numeric_positions = {}
        text_tokens = []

        for i, token in enumerate(tokens):
            if re.fullmatch(r"[#\-]?\d+([a-z]*)?", token):  # Matches 123, -02, 11a, etc.
                numeric_positions[i] = token
            else:
                text_tokens.append(token)
        
        return text_tokens, numeric_positions, tokens

    def detokenize_address(text_tokens, numeric_positions):
        output_tokens = []
        text_iter = iter(text_tokens)
        for i in range(max(numeric_positions.keys(), default=-1) + len(text_tokens) + 1):
            if i in numeric_positions:
                output_tokens.append(numeric_positions[i])
            else:
                try:
                    output_tokens.append(next(text_iter))
                except StopIteration:
                    break
        return " ".join(output_tokens)

    def fuzzy_normalize_column(df, column_name, threshold=80):
        print(f"\n🔍 Processing column: {column_name}")

        df[column_name] = df[column_name].astype(str).fillna('').str.strip().str.lower()

        # Identify rows to exclude from fuzzy matching
        df[f"{column_name}_skip_fuzzy"] = df[column_name].apply(contains_numeric_range)

        tokenized = df[column_name].apply(tokenize_address)
        df[f"{column_name}_text_tokens"] = tokenized.apply(lambda x: x[0])
        df[f"{column_name}_num_positions"] = tokenized.apply(lambda x: x[1])
        df[f"{column_name}_orig_tokens"] = tokenized.apply(lambda x: x[2])
        df[f"{column_name}_text"] = df[f"{column_name}_text_tokens"].apply(lambda x: " ".join(x))

        # Perform fuzzy matching only on rows without numeric ranges
        mask_fuzzy = ~df[f"{column_name}_skip_fuzzy"]
        values_to_match = df.loc[mask_fuzzy, f"{column_name}_text"].unique().tolist()

        reference_mapping = {}
        groups = defaultdict(list)

        for value in values_to_match:
            if value in reference_mapping:
                continue

            matches = process.extract(value, values_to_match, limit=10, scorer=fuzz.ratio)
            matches = [(match, score) for match, score in matches if score >= threshold]

            best_match = max(matches, key=lambda x: x[1])[0] if matches else value

            for match, score in matches:
                reference_mapping[match] = best_match
                groups[best_match].append(match)

        all_grouped_labels[column_name] = dict(groups)

        # Map fuzzy results
        df[f"{column_name}_text_normalised"] = df.apply(
            lambda row: row[f"{column_name}_text"] if row[f"{column_name}_skip_fuzzy"]
            else reference_mapping.get(row[f"{column_name}_text"], row[f"{column_name}_text"]),
            axis=1
        )

        df[f"{column_name}_text_tokens_normalised"] = df[f"{column_name}_text_normalised"].apply(lambda x: x.split())
        df[f"{column_name}_normalised"] = df.apply(
            lambda row: detokenize_address(row[f"{column_name}_text_tokens_normalised"], row[f"{column_name}_num_positions"]),
            axis=1
        )

        df.drop(columns=[
            f"{column_name}_text_tokens", f"{column_name}_num_positions",
            f"{column_name}_orig_tokens", f"{column_name}_text",
            f"{column_name}_text_normalised", f"{column_name}_text_tokens_normalised",
            f"{column_name}_skip_fuzzy"
        ], inplace=True)

        return df

    for col in columns_to_clean:
        df = fuzzy_normalize_column(df, col, threshold)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n✅ Cleaned data saved as: {save_path}")

    print("\n🧾 All Grouped Labels:")
    for col, group_map in all_grouped_labels.items():
        print(f"\n📦 Grouped values for '{col}':")
        for canonical, originals in group_map.items():
            if len(originals) > 1:
                print(f"\n  → {canonical}:")
                for original in originals:
                    print(f"     - {original}")

    return df, all_grouped_labels
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

    if args.fuzzy_clean_address:
        console.log("Running fuzzy cleaning for address:")
        columns = args.columns if args.columns else []  # fallback if None
        cleaned_df, grouped_info = fuzzy_clean_columns(df, columns, save_path="Food_Inspections_Cleaned_address.csv")
        #console.log("Persisting preprocessed file:")
        #save_data(df)

    #### SINGLE PROFILLING ##### 
    if args.single_profile:
        single_profiling(df)
        console.log("Running Single Profilling")


    #### RULE MINING ##### EUGENE
    if args.rule_mining:
        console.log("Running Rule Mining")

        # Safely parse stringified list of column names
        columns_to_remove = ast.literal_eval(args.columns_to_remove)

        df_factorized, mappings = clean_and_factorize_data(
            input_csv=args.input_csv,
            columns_to_remove=columns_to_remove
        )

        run_efficient_apriori_from_df(
            df=df_factorized,
            min_support=args.min_support,
            min_confidence=args.min_confidence
        )

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
    
    #Association Mining
    parser.add_argument("--rule_mining", action="store_true", help="Run rule mining")
    parser.add_argument("--input_csv", type=str,
                        default='data/Food_Inspections_Violations_Expanded_with_cleandata_address.csv',
                        help="Path to input CSV file for rule mining")
    parser.add_argument("--columns_to_remove", type=str,
                        default="['Inspection ID', 'AKA Name', 'Latitude', 'Longitude', 'raw_violation', 'violation_comment', 'parse_error', 'error_reason', 'Facility Type', 'City', 'Inspection Type', 'City Cleaned', 'State']",
                        help="Stringified list of column names to remove")
    parser.add_argument("--min_support", type=float, default=0.05,
                        help="Minimum support threshold for rule mining")
    parser.add_argument("--min_confidence", type=float, default=0.6,
                        help="Minimum confidence threshold for rule mining")
    
    args = parser.parse_args()

    console.log(f"Args: {args}")

    main(args)