# functional discoveries

from typing import Tuple, List, Dict, Any
from dataclasses import dataclass, field

from rich.console import Console
import pandas as pd
import desbordante
import desbordante.fd.algorithms as fd_algorithms
import desbordante.afd.algorithms as afd_algorithms

console = Console()

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

        console.log(f"Algorthm: {algo_class}")
        
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

        print(f"Algorthm: {algo_class}")
        
        algo = algo_class()
        algo.load_data(table=df)
        algo.execute(error=error)
        return algo.get_fds()
    except AttributeError:
        raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(afd_algorithms)}")
    

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

def convert_fd(fd:desbordante.fd.FD) -> Tuple[list, str]:
    fd_str = str(fd) # convert fd to string
    fd_str_split = fd_str.split("->") # split fd to lhs and rhs
    lhs = fd_str_split[0].strip() 
    rhs = fd_str_split[-1].strip()

    lhs_list = lhs[1:-1].split(' ') # convert lhs to list of attributes

    return lhs_list, rhs