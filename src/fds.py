# functional discoveries

from typing import Tuple, List, Dict, Any
from dataclasses import dataclass, field
import re

from rich.console import Console
import pandas as pd
import desbordante
import desbordante.fd.algorithms as fd_algorithms
import desbordante.afd.algorithms as afd_algorithms
import desbordante.ind.algorithms as ind_algorithms

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

