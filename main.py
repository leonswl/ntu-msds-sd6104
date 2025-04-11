import os
import argparse

from rich.console import Console
import pandas as pd
import numpy as np
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

def main(args):

    df = load_data()

    ##### PREPROCESSING ##### WANG YU
    if not args.no_preprocess:
        console.log("Running preprocessing on raw file:")
        df = preprocess(df)

        console.log("Persisting preprocessed file:")
        save_data(df)

    #### SINGLE PROFILLING ##### 
    if args.single_profile:
        single_profiling(df)
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