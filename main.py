import os
import argparse
from pathlib import Path
import re
import ast

from rich.console import Console
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  

from thefuzz import process, fuzz
from collections import defaultdict


from src.association_rule_mining import (clean_and_factorize_data, run_efficient_apriori_from_df)
from src.preprocess import preprocess
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

    # Association Mining
    parser.add_argument(
        "--rule_mining",
        action="store_true",
        help="Run rule mining"
    )

    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default="['Inspection ID', 'AKA Name', 'Latitude', 'Longitude', 'raw_violation', 'violation_comment', 'parse_error', 'error_reason', 'Facility Type', 'City', 'Inspection Type', 'City Cleaned', 'State']",
        help="Stringified list of column names to remove"
    )

    parser.add_argument(
        "--min_support",
        type=float,
        default=0.05,
        help="Minimum support threshold for rule mining"
        )
    
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for rule mining"
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
    • console.logs a DataFrame with column‑level metrics (no CSV written)
    • Saves a first‑digit bar chart for every *numeric* column
    • Saves two bar charts for every *text* column:
        - full range  (only if #unique ≤ max_text_unique)
        - top-N + 'Others'
    All PNGs land in *out_dir* (created if missing).
    """
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    profiler = DataProfiler(df)

    # Profile table – console.log via Rich
    profile_df = profiler.profile_dataframe(numeric_bins=numeric_bins)
    console.rule("[bold green]Column‑level profile[/bold green]")
    console.console.log(profile_df)

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
        console.log(f"✅ Done! File saved as: {save_path}")

    return violations_expanded_df


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
        console.log(f"\n🔍 Processing column: {column_name}")
        
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

        # Store groups for console.loging
        all_grouped_labels[column_name] = dict(groups)

        return df

    # Process each column
    for col in columns_to_clean:
        df = fuzzy_normalize_column(df, col, threshold)

    # Save to CSV if needed
    if save_path:
        df.to_csv(save_path, index=False)
        console.log(f"\n✅ Cleaned data saved as: {save_path}")

    # console.log grouped results
    console.log("\n🧾 All Grouped Labels:")
    for col, group_map in all_grouped_labels.items():
        console.log(f"\n📦 Grouped values for '{col}':")
        for canonical, originals in group_map.items():
            if len(originals) > 1:
                console.log(f"\n  → {canonical}:")
                for original in originals:
                    console.log(f"     - {original}")

    return df, all_grouped_labels

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
        console.log(f"\n🔍 Processing column: {column_name}")

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
        console.log(f"\n✅ Cleaned data saved as: {save_path}")

    console.log("\n🧾 All Grouped Labels:")
    for col, group_map in all_grouped_labels.items():
        console.log(f"\n📦 Grouped values for '{col}':")
        for canonical, originals in group_map.items():
            if len(originals) > 1:
                console.log(f"\n  → {canonical}:")
                for original in originals:
                    console.log(f"     - {original}")

    return df, all_grouped_labels


def find_fds(df, algorithm_name='Default'):
    """
    Finds functional dependencies in a given DataFrame using a specified algorithm.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        algorithm_name (str): The name of the FD algorithm to use. Defaults to 'Default'. Options are 
    
    Returns:
        list: A list of discovered functional dependencies.
    """
    try:
        # Get the algorithm class dynamically from desbordante.fd.algorithms
        algo_class = getattr(fd_algorithms, algorithm_name, fd_algorithms.Default)

        console.log(f"Algorthm: {algo_class.__name__}")
        
        algo = algo_class()
        algo.load_data(table=df)
        algo.execute()
        return algo.get_fds()
    except AttributeError:
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(fd_algorithms)}")
    

def find_afds(df:pd.DataFrame, error:float=0.1, algorithm_name:str='Default'):
    """
    Finds approximate functional dependencies in a given DataFrame using a specified algorithm.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        algorithm_name (str): The name of the FD algorithm to use. Defaults to 'Default'.
    
    Returns:
        list: A list of discovered approximate functional dependencies.
    """
    try:

        # Get the algorithm class dynamically from desbordante.fd.algorithms
        algo_class = getattr(afd_algorithms, algorithm_name, afd_algorithms.Default)

        console.log(f"Algorthm: {algo_class.__name__}")
        
        algo = algo_class()
        algo.load_data(table=df)
        algo.execute(error=error)
        return algo.get_fds()
    except AttributeError:
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(afd_algorithms)}")
    

def find_inds(df:list [pd.DataFrame] | pd.DataFrame, algorithm_name:str='Default'):
    """
    Finds inclusion dependencies in a given DataFrame using a specified algorithm.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        algorithm_name (str): The name of the FD algorithm to use. Defaults to 'Default'.
    
    Returns:
        list: A list of discovered approximate functional dependencies.
    """
    try:

        # Get the algorithm class dynamically from desbordante.fd.algorithms
        algo_class = getattr(ind_algorithms, algorithm_name, ind_algorithms.Default)

        console.log(f"Algorthm: {algo_class.__name__}")
        
        algo = algo_class()
        algo.load_data(tables=df)
        algo.execute(
            allow_duplicates=False,  # Ignore duplicate INDs
        )
        
        # Filter out self-dependencies
        return [
            ind for ind in algo.get_inds()
            if ind.get_lhs().column_indices != ind.get_rhs().column_indices
        ]
    except AttributeError:
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(ind_algorithms)}")

    
def find_ainds(df:list [pd.DataFrame] | pd.DataFrame, algorithm_name:str='Default', error:float=0.3):
    """
    Finds approximate inclusion dependencies in a given DataFrame using a specified algorithm.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        algorithm_name (str): The name of the FD algorithm to use. Defaults to 'Default'.
    
    Returns:
        list: A list of discovered approximate functional dependencies.
    """
    try:

        # Get the algorithm class dynamically from desbordante.fd.algorithms
        algo_class = getattr(ind_algorithms, algorithm_name, ind_algorithms.Default)

        console.log(f"Algorthm: {algo_class.__name__}")
        
        algo = algo_class()
        algo.load_data(tables=df)
        algo.execute(
            max_lhs_size=2,  # Look for multi-column INDs
            allow_approximate=True,  # Enable approximate matches
            error_threshold=error  # Allow 20% violations
        )
        # Filter out self-dependencies
        return [
            ind for ind in algo.get_inds()
            if ind.get_lhs().column_indices != ind.get_rhs().column_indices
        ]
    except AttributeError:
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(ind_algorithms)}")

@dataclass
class FunctionalDependency:
    lhs: List[str]  # Left-hand side attributes
    rhs: str        # Right-hand side attribute

    def __str__(self):
       lhs_count = len(self.lhs)
       base = f"LHS={self.lhs} ({lhs_count}), RHS={self.rhs}"
       return base
    
@dataclass
class FunctionalDependencySet:
    dependencies: List[FunctionalDependency] = field(default_factory=list)
    validation_results: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = field(default_factory=dict)

    def add_dependency(self, lhs: List[str], rhs: str):
        """Adds a new functional dependency to the set."""
        self.dependencies.append(FunctionalDependency(lhs, rhs))

    def __len__(self):
        """Returns the number of functional dependencies."""
        return len(self.dependencies)

    def __iter__(self):
        """Allows iteration over functional dependencies."""
        return iter(self.dependencies)
    
    def validate_fd(self, df):
        """Validates all functional dependencies in the dataset and stores the results."""
        

        verifier = desbordante.fd_verification.algorithms.Default()
    
        verifier.load_data(table=df)

        for fd in self.dependencies:
            lhs_idx = df.columns.get_indexer(fd.lhs)
            rhs_idx = df.columns.get_loc(fd.rhs)

            if lhs_idx[0] == -1:
                continue

            verifier.execute(lhs_indices=lhs_idx, rhs_indices=[rhs_idx])
            highlights = verifier.get_highlights()

            fd_key = (tuple(fd.lhs), fd.rhs)
            self.validation_results[fd_key] = {
                "holds": verifier.fd_holds(),
                "num_violations": verifier.get_num_error_clusters(),
                "highlights": highlights
            }

            if self.validation_results[fd_key]["holds"]:
                # console.log(GREEN_CODE, f"FD holds: {fd.lhs} -> {fd.rhs}", DEFAULT_COLOR_CODE)
                console.log(f"FD holds: {fd.lhs} -> {fd.rhs}", style="bold black on green")

            else:
                console.log(f"FD does not hold: {fd.lhs} -> {fd.rhs}", style="bold white on red")
                console.log(f"Number of clusters violating FD: {self.validation_results[fd_key]['num_violations']}")

    def validate_afd(self, df:pd.DataFrame, error:float=0.05):
        """Validates all functional dependencies in the dataset and stores the results."""

        verifier = desbordante.afd_verification.algorithms.Default()
            
        verifier.load_data(table=df)

        for fd in self.dependencies:
            lhs_idx = df.columns.get_indexer(fd.lhs)
            rhs_idx = df.columns.get_loc(fd.rhs)

            if lhs_idx[0] == -1:
                continue
            
            verifier.execute(lhs_indices=lhs_idx, rhs_indices=[rhs_idx])
            highlights = verifier.get_highlights()

            fd_holds = verifier.get_error() < error

            if fd_holds:
                console.log("AFD with this error threshold holds", style="bold black on green")
            else:
                console.log(f"AFD with this error threshold does not hold", style="bold white on red")
                console.log(f"But the same AFD with error threshold = {verifier.get_error()} holds.")


            fd_key = (tuple(fd.lhs), fd.rhs)
            self.validation_results[fd_key] = {
                "holds": fd_holds,
                "num_violations": verifier.get_num_error_clusters(),
                "highlights": highlights
            }

            if self.validation_results[fd_key]["holds"]:
                console.log(f"FD holds: {fd.lhs} -> {fd.rhs}", style="bold black on green")
            else:
                console.log(f"FD does not hold: {fd.lhs} -> {fd.rhs}", style="bold white on red")
                console.log(f"Number of clusters violating FD: {self.validation_results[fd_key]['num_violations']}")

    def get_validation_result(self, lhs: List[str], rhs: str) -> Dict[str, Any]:
        """Retrieves stored validation results for a specific FD."""
        fd_key = (tuple(lhs), rhs)
        return self.validation_results.get(fd_key, {})

    def get_all_validation_results(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Returns all stored validation results."""
        return self.validation_results
    

@dataclass
class InclusionDependency:
    lhs: List[str]  # Left-hand side attributes
    rhs: List[str]  # Right-hand side attributes

    def __str__(self):
       lhs_count = len(self.lhs)
       base = f"LHS={self.lhs} ({lhs_count}), RHS={self.rhs}"
       return base
    
@dataclass
class InclusionDependencySet:
    dependencies: List[InclusionDependency] = field(default_factory=list)
    validation_results: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = field(default_factory=dict)

    def add_dependency(self, lhs: List[str], rhs: List[str]):
        """Adds a new functional dependency to the set."""
        self.dependencies.append(InclusionDependency(lhs, rhs))

    def __len__(self):
        """Returns the number of functional dependencies."""
        return len(self.dependencies)

    def __iter__(self):
        """Allows iteration over functional dependencies."""
        return iter(self.dependencies)
    
    def validate_ind(self, df):
        """Validates all inclusion dependencies in the dataset and displays the results."""

        def ind_str(lhs, rhs):
            def cc_str(cc):
                (df, indices) = cc
                columns = [df.columns[idx] for idx in indices]
                return ", ".join(f"{col}" for col in columns)

            return f"[{cc_str(lhs)}] -> [{cc_str(rhs)}]"

        for fd in self.dependencies:
            lhs_idx = df.columns.get_indexer(fd.lhs)
            rhs_idx = df.columns.get_loc(fd.rhs)

            console.log(f"Checking the IND {ind_str((df, lhs_idx), (df, rhs_idx))}")

            if lhs_idx[0] == -1:
                continue

            algo = desbordante.ind_verification.algorithms.Default()
            algo.load_data(tables=[df, df])
            algo.execute(lhs_indices=lhs_idx, rhs_indices=rhs_idx)
            
            if algo.get_error() == 0:
                console.log("IND holds", style="bold black on green")
            else:
                console.log(f"IND holds with error = {algo.get_error():.2}", style="bold white on red")

    def validate_aind(self, df):
        """Validates all approximate inclusion dependencies in the dataset and displays the results."""

        def ind_str(lhs, rhs):
            def cc_str(cc):
                (df, indices) = cc
                columns = [df.columns[idx] for idx in indices]
                return ", ".join(f"{col}" for col in columns)

            return f"[{cc_str(lhs)}] -> [{cc_str(rhs)}]"

        for fd in self.dependencies:
            lhs_idx = df.columns.get_indexer(fd.lhs)
            rhs_idx = df.columns.get_loc(fd.rhs)

            console.log(f"Checking the IND {ind_str((df, lhs_idx), (df, rhs_idx))}")

            if lhs_idx[0] == -1:
                continue

            algo = desbordante.aind_verification.algorithms.Default()
            algo.load_data(tables=[df, df])
            algo.execute(lhs_indices=lhs_idx, rhs_indices=rhs_idx)
            
            if algo.get_error() == 0:
                console.log("IND holds", style="bold black on green")
            else:
                console.log(f"AIND holds with error = {algo.get_error():.2}", style="bold white on red")


def convert_fd(fd:desbordante.fd.FD) -> Tuple[list, str]:
    fd_str = str(fd) # convert fd to string
    fd_str_split = fd_str.split("->") # split fd to lhs and rhs
    lhs = fd_str_split[0].strip() 
    rhs = fd_str_split[-1].strip()

    lhs_list = lhs[1:-1].split(' ') # convert lhs to list of attributes

    return lhs_list, rhs

def convert_ind(ind:desbordante.ind.IND) -> Tuple[list, list]:
    ind_str = str(ind)
    ind_str_split = ind_str.split("->") # split fd to lhs and rhs
    lhs = ind_str_split[0].strip() 
    rhs = ind_str_split[-1].strip()

    # Regex to match content within square brackets
    pattern = r"\[([^\[\]]+)\]"

    # Find matches
    lhs_matches = re.findall(pattern, lhs)

    rhs_matches = re.findall(pattern, rhs)

    return lhs_matches, rhs_matches

import pandas as pd
import os
import matplotlib.pyplot as plt
from openpyxl.utils.exceptions import IllegalCharacterError

def detect_pattern(value):
    if pd.isna(value):
        return "Missing"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    value = str(value).strip()
    return "".join([
        "9" if c.isdigit() else
        "A" if c.isupper() else
        "a" if c.islower() else
        c
        for c in value
    ])

def clean_display_value(val):
    return int(val) if isinstance(val, float) and val.is_integer() else val

def analyze_column_patterns(df, column_name, top_n=20, show_pattern_index=None, show_distinct=False, output_dir="column_pattern_histograms"):
    if column_name not in df.columns:
        print(f"❌ Column '{column_name}' not found in DataFrame.")
        return

    if df[column_name].dropna().empty:
        print(f"⚠️ Column '{column_name}' is empty after dropping NA values.")
        return

    pattern_series = df[column_name].map(detect_pattern)
    pattern_counts = pattern_series.value_counts()
    print(f"✅ Column '{column_name}' has {len(pattern_counts)} unique pattern(s).")

    os.makedirs(output_dir, exist_ok=True)

    summary_data = {
        "Pattern": pattern_counts.index,
        "Count": pattern_counts.values
    }

    # Collect sample values for all patterns (up to 5 per pattern)
    if show_distinct:
        sample_values = []
        all_distinct = set()
        for pattern in pattern_counts.index:
            matches = df.loc[pattern_series == pattern, column_name]
            unique_vals = matches.dropna().unique()
            clean_vals = [clean_display_value(v) for v in unique_vals]
            all_distinct.update(clean_vals)
            sample_values.append(", ".join(map(str, clean_vals[:5])))
        summary_data["Sample Values"] = sample_values

        print("\n📌 Distinct values (all patterns):")
        for val in sorted(all_distinct):
            print(f"- {val}")

    # Save Excel with error handling
    summary_df = pd.DataFrame(summary_data)
    excel_path = os.path.join(output_dir, f"{column_name}_pattern_summary.xlsx")
    try:
        summary_df.to_excel(excel_path, index=False)
        print(f"📄 Pattern summary saved to: {excel_path}")
    except IllegalCharacterError:
        print("⚠️ Excel export failed due to illegal characters in the content. Summary not saved to Excel.")

    # Plot top N patterns
    top_patterns = pattern_counts.head(top_n)
    pattern_labels, pattern_freqs = top_patterns.index.tolist(), top_patterns.values.tolist()

    plt.figure(figsize=(12, 6))
    plt.barh(pattern_labels, pattern_freqs, color='skyblue')
    plt.xlabel("Frequency")
    plt.ylabel("Patterns (Aa9... & Special Characters)")
    plt.title(f"Top {top_n} Value Patterns in '{column_name}'")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"{column_name}_patterns_histogram.png")
    plt.savefig(file_path)
    plt.show()
    print(f"📊 Histogram saved to: {file_path}")

    # Print sample values from specific pattern indices
    if show_pattern_index is not None:
        if isinstance(show_pattern_index, int):
            show_pattern_index = [show_pattern_index]
        elif not isinstance(show_pattern_index, (list, tuple, set)):
            print(f"❌ Invalid type for show_pattern_index. Must be int or list of int.")
            return

        print("\n🔍 Sample values for selected pattern indices:")
        for idx in show_pattern_index:
            if idx < 0 or idx >= len(top_patterns):
                print(f"❌ Invalid pattern index {idx}. Valid range: 0 to {len(top_patterns)-1}")
                continue
            selected_pattern = pattern_labels[idx]
            matching_rows = df.loc[pattern_series == selected_pattern, column_name]
            cleaned_sample = matching_rows.head(10).map(clean_display_value)
            print(f"\n🔢 Pattern index [{idx}] - {len(matching_rows)} rows match")
            print(cleaned_sample.to_string(index=False))


def main(args):

    df = load_data()

    ##### PREPROCESSING ##### WANG YU
    if args.process_violations:
        console.log("Running preprocessing on violations column:")
        violations_df = expand_violations(df, save_path="Food_Inspections_Violations_Expanded.csv")


    if args.fuzzy_clean_columns:
        console.log("Running fuzzy cleaning:")
        columns = args.columns if args.columns else []  # fallback if None
        cleaned_df, grouped_info = fuzzy_clean_columns(df, columns, save_path="Food_Inspections_Cleaned.csv")

   
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV or Excel file.")
    parser.add_argument("--columns", nargs="+", required=True, help="List of columns to fuzzy clean.")
    parser.add_argument("--threshold", type=int, default=80, help="Similarity threshold (default: 80).")
    parser.add_argument("--save_path", type=str, help="Optional path to save the cleaned CSV.")
    parser.add_argument("--fuzzy_clean_columns", action="store_true", help="Run fuzzy address cleaning.")

    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(args.input)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(args.input)
    else:
        print(f"❌ Unsupported file format: {ext}")
        return

    if args.fuzzy_clean_address:
        print("🚀 Running fuzzy cleaning:")
        df, grouped_info = fuzzy_clean_address(
            df,
            columns_to_clean=args.columns,
            threshold=args.threshold,
            save_path=args.save_path
        )

    parser = argparse.ArgumentParser(description="Analyze value patterns in a DataFrame column.")
    parser.add_argument("--input", type=str, required=True, help="Path to the CSV or Excel file.")
    parser.add_argument("--column", type=str, required=True, help="Column name to analyze.")
    parser.add_argument("--analyze_column_patterns", action="store_true", help="Trigger column pattern analysis.")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top patterns to show.")
    parser.add_argument("--show_pattern_index", nargs='*', type=int, help="Indices of patterns to show sample values for.")
    parser.add_argument("--show_distinct", action="store_true", help="Show distinct values for each pattern.")

    args = parser.parse_args()

    # Load file
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(args.input)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(args.input)
    else:
        print(f"❌ Unsupported file type: {ext}")
        return

    # Trigger analysis
    if args.analyze_column_patterns:
        print("📊 Analyzing column patterns:")
        analyze_column_patterns(
            df=df,
            column_name=args.column,
            top_n=args.top_n,
            show_pattern_index=args.show_pattern_index,
            show_distinct=args.show_distinct
        )

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
            df,
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
    
    args = parser.parse_args()

    console.log(f"Args: {args}")

    main(args)