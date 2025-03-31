import os

from rich.console import Console
import pandas as pd

from src.preprocess import preprocess
from src.fds import (
    convert_fd,
    find_fds,
    find_afds,
    FunctionalDependencySet
)


console = Console()

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


def main():

    df = load_data()

    df = preprocess(df)

    save_data(df)

    fd_results = discover_fds(df)

    afd_results = discover_afds(df=df, error=0.05)

if __name__ == "__main__":
    main()