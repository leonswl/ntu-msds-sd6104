import os
import argparse
from pathlib import Path
import re
import ast
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import string


from rich.console import Console
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from openpyxl.utils.exceptions import IllegalCharacterError
from thefuzz import process, fuzz
import desbordante
import desbordante.fd.algorithms as fd_algorithms
import desbordante.afd.algorithms as afd_algorithms
import desbordante.ind.algorithms as ind_algorithms
from efficient_apriori import apriori

matplotlib.use("Agg")
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_colwidth", None)

console = Console()



def create_arg_parser():
    """
    Creates and returns an argument parser.
    """

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Data preparation software for SD6104."
    )

    parser.add_argument(
        "-i",
        "--input", 
        type=str, 
        required=True, 
        help="Path to the CSV or Excel file."
    )

    parser.add_argument(
        "-p",
        "--preprocess",
        action="store_true"
    )

    # Argument flag for profilling
    parser.add_argument(
        "-s",
        "--single-profile",
        action="store_true",
        required=False,
        help="Specify this flag to perform single-column profilling.",
    )

    # Argument flag for association rule mining
    parser.add_argument(
        "-rm",
        "--rule-mining",
        action="store_true",
        required=False,
        help="Specify this flag to perform association rule mining.",
    )

    # Argument for functional dependencies
    parser.add_argument(
        "-fd",
        "--func-dependencies",
        choices=["all", "default", "approximate"],
        nargs="?",
        default=None,
        required=False,
        type=str,
        help="Specify the method for functional dependencies: 'default' or 'approximate'.",
    )

    # Argument for inclusion dependencies
    parser.add_argument(
        "-ind",
        "--ind-dependencies",
        choices=["all", "default", "approximate"],
        nargs="?",
        default=None,
        required=False,
        type=str,
        help="Specify the method for inclusion dependencies: 'default' or 'approximate'.",
    )

    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default="['Inspection ID', 'AKA Name', 'Facility Type', 'City', 'State', 'Inspection Type', 'Latitude', 'Longtitude', 'raw_violation', 'violation_comment', 'parse_error', 'error_reason', 'City_normalised', 'Address']",
        help="Stringified list of column names to remove",
    )

    parser.add_argument(
        "--min_support",
        type=float,
        default=0.05,
        help="Minimum support threshold for rule mining",
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for rule mining",
    )

    # ARguments for analyze_column_patterns
    parser.add_argument(
        "--analyze_column_patterns",
        action="store_true",
        help="Trigger column pattern analysis.",
    )

    parser.add_argument(
        "--column", type=str, required=False, help="Column name to analyze."
    )

    parser.add_argument(
        "--top_n", type=int, default=20, help="Number of top patterns to show."
    )
    parser.add_argument(
        "--show_pattern_index",
        nargs="*",
        default=[3,4,5,6],
        type=list,
        help="Indices of patterns to show sample values for.",
    )
    parser.add_argument(
        "--show_distinct",
        action="store_true",
        help="Show distinct values for each pattern.",
    )

    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=99,
        help="Threshold for fuzzy matching",
    )

    return parser

def clean_and_factorize_data(df, columns_to_remove):

    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    mappings = {}
    for col in df.columns:
        codes, uniques = pd.factorize(df[col])
        df[col] = codes
        mappings[col] = dict(enumerate(uniques))

    console.log(f"✅ Factorization complete")
    return df, mappings

def run_efficient_apriori_from_df(df, min_support=0.05, min_confidence=0.6):
    
    transactions = [
        tuple(f"{col}={row[col]}" for col in df.columns if pd.notna(row[col]))
        for _, row in df.iterrows()
    ]

    console.log("\n⏳ Running efficient-apriori...")
    itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    console.log(f"✅ Apriori completed")

    # Frequent itemsets
    itemset_rows = []
    for k, itemset_dict in itemsets.items():
        for items, support in itemset_dict.items():
            itemset_rows.append({
                'itemsets': frozenset(items),
                'support': support
            })
    frequent_itemsets_df = pd.DataFrame(itemset_rows).sort_values(by='support', ascending=False)
    console.log("\nFrequent Itemsets (Top 30):")
    console.log(frequent_itemsets_df.head(30))

    # Association rules
    rules_rows = []
    for rule in rules:
        rules_rows.append({
            'antecedents': frozenset(rule.lhs),
            'consequents': frozenset(rule.rhs),
            'support': rule.support,
            'confidence': rule.confidence,
            'lift': rule.lift
        })
    rules_df = pd.DataFrame(rules_rows).sort_values(by=['confidence', 'support'], ascending=[False, False])
    console.log("\nAssociation Rules (Top 30 by Confidence (with Support as Tie-Breaker)):")
    console.log(rules_df.head(30))

    return frequent_itemsets_df, rules_df

def is_none_or_empty(df):
    return df is None or df.empty

def discover_fds(df):
    results = find_fds(df)

    console.log(
        f"There are {len(results)} functional dependencies using Default algorithm."
    )

    for fd in results:
        console.log(fd)

    fd_set = FunctionalDependencySet()
    for result in results:
        lhs, rhs = convert_fd(fd=result)
        fd_set.add_dependency(lhs, rhs)

    # Validate all dependencies and store results
    fd_set.validate_fd(df)
    console.log(f"There are {len(fd_set)} fds in the dataset.")

    # Retrieve all validation results
    all_results = fd_set.get_all_validation_results()
    for (lhs, rhs), result in all_results.items():
        # Create a copy of result without the 'highlights' key
        filtered_result = {
            key: value for key, value in result.items() if key != "highlights"
        }

        console.log(f"FD: {lhs} -> {rhs}, Results: {filtered_result}")

    return all_results


def discover_afds(df, error):
    results = find_afds(df, error)

    console.log(
        f"There are {len(results)} functional dependencies using Default algorithm."
    )

    for fd in results:
        console.log(fd)

    fd_set = FunctionalDependencySet()
    for result in results:
        lhs, rhs = convert_fd(fd=result)
        fd_set.add_dependency(lhs, rhs)

    # Validate all dependencies and store results
    fd_set.validate_afd(df)
    console.log(f"There are {len(fd_set)} fds in the dataset.")

    # Retrieve all validation results
    all_results = fd_set.get_all_validation_results()
    for (lhs, rhs), result in all_results.items():
        # Create a copy of result without the 'highlights' key
        filtered_result = {
            key: value for key, value in result.items() if key != "highlights"
        }

        console.log(f"FD: {lhs} -> {rhs}, Results: {filtered_result}")

    return all_results


def run_ind(df):
    results = find_inds([df, df])

    console.log(
        f"There are {len(results)} inclusion dependencies using Default algorithm."
    )

    for ind in results:
        console.log(ind)

    ind_set = InclusionDependencySet()
    for result in results:
        lhs, rhs = convert_ind(result)
        ind_set.add_dependency(lhs, rhs)

    # Validate all dependencies
    ind_set.validate_ind(df)


def run_aind(df, error):
    results = find_ainds([df, df], error=error)

    console.log(
        f"There are {len(results)} inclusion dependencies using Default algorithm."
    )

    for ind in results:
        console.log(ind)

    ind_set = InclusionDependencySet()
    for result in results:
        lhs, rhs = convert_ind(result)
        ind_set.add_dependency(lhs, rhs)

    # Validate all dependencies
    ind_set.validate_aind(df)


# Data Classifier Functions
all_column_counts = defaultdict(lambda: defaultdict(int))
all_column_examples = defaultdict(lambda: defaultdict(list))
MAX_EXAMPLES = 5

def classify_data_class(value, column_name, id_value=None):
    if pd.isna(value):
        class_type = "Missing"
        value_str = "NaN"
    else:
        value_str = str(value).strip()

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value_str) or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", value_str):
            class_type = "Date/Time"
        elif re.fullmatch(r"\d+(\.\d+)?", value_str):
            class_type = "Quantity"
        elif re.fullmatch(r"[A-Za-z0-9]+", value_str) and re.search(r"[A-Za-z]", value_str) and re.search(r"\d", value_str):
            class_type = "Code/Identifier"
        # elif re.fullmatch(r"[A-Za-z0-9\s.,!?&@#%*'’\"();:\[\]_\-+/\\=<>$|{}\n\r]+", value_str): (old line from wang yu)
        elif re.fullmatch(r"[A-Za-z0-9\s.,!?&@#%*'\"();:\[\]_\-+/\\=<>$|{}\n\r]+", value_str): # selene changed this line
            class_type = "Text"
        elif len(value_str) > 0 and sum(c in string.printable for c in value_str) / len(value_str) > 0.6:
            class_type = "Text"
        else:
            class_type = "Other"

    all_column_counts[column_name][class_type] += 1

    if len(all_column_examples[column_name][class_type]) < MAX_EXAMPLES and id_value is not None:
        all_column_examples[column_name][class_type].append((id_value, value_str))

    return class_type

def analyze_all_columns(df, max_examples=5, id_column="Inspection ID"):
    all_column_counts.clear()
    all_column_examples.clear()

    for column in df.columns:
        for idx, val in df[column].items():
            id_val = df.at[idx, id_column] if id_column in df.columns else idx
            classify_data_class(val, column, id_val)

    console.log("Data Class Summary for All Columns:\n")
    for col, class_dict in all_column_counts.items():
        console.log(f" Column: '{col}'")
        for class_name, count in class_dict.items():
            console.log(f"  - {class_name} ({count} total):")
            for id_val, example in all_column_examples[col][class_name]:
                console.log(f"     • [Inspection ID {id_val}] {example}")
        console.log("")
    return all_column_counts


class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df.copy()
        self.column_profiles: Dict[str, Dict[str, Union[int, float, dict]]] = {}

    def profile_column(self, col: str, numeric_bins: int = 10) -> Dict[str, Union[int, float, dict]]:
        s = self.df[col]
        num_rows = len(s)
        profile: Dict[str, Union[int, float, dict]] = {
            "num_rows": num_rows,
            "null_count": s.isna().sum(),
        }
        profile["null_pct"] = profile["null_count"] / num_rows if num_rows else np.nan
        profile["distinct_count"] = s.nunique(dropna=False)
        profile["uniqueness"] = profile["distinct_count"] / num_rows if num_rows else np.nan
        most_freq = s.value_counts(dropna=False).iloc[0] if num_rows else 0
        profile["constancy"] = most_freq / num_rows if num_rows else np.nan

        #
        numeric = pd.to_numeric(s, errors="coerce")
        numeric_clean = numeric.dropna()
        is_numeric = (numeric.notna().sum() / num_rows) > 0.9 if num_rows else False

        if is_numeric:
            try:
                profile["quartiles"] = numeric_clean.quantile([0.25, 0.5, 0.75]).to_dict()
            except Exception as e:
                profile["quartiles"] = {"error": f"Could not compute quartiles: {e}"}
        
            profile.update(self._numeric_histograms(numeric_clean, bins=numeric_bins))
            profile["first_digit_distribution"] = self._first_digit_distribution(numeric_clean).to_dict()
        else:
            profile.update(self._length_metrics(s))
            profile["histogram"] = s.value_counts(dropna=False).to_dict()

        self.column_profiles[col] = profile
        return profile

    def profile_dataframe(self, numeric_bins: int = 10) -> pd.DataFrame:
        for col in self.df.columns:
            self.profile_column(col, numeric_bins=numeric_bins)
        return pd.DataFrame(self.column_profiles).T

    @staticmethod
    def _first_digit_distribution(series: pd.Series) -> pd.Series:
        s = series.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        first_digits: List[str] = []
        for val in s.astype(str):
            val = val.lstrip(" -+0")
            if val and val[0].isdigit():
                first_digits.append(val[0])
        if not first_digits:
            return pd.Series(dtype=float)
        counts = pd.Series(first_digits).value_counts(sort=False).sort_index()
        return counts / counts.sum()

    @staticmethod
    def _numeric_histograms(series: pd.Series, bins: int = 10) -> Dict[str, Union[dict, str]]:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return {}
        counts, edges = np.histogram(numeric, bins=bins)
        equi_width = {"bin_edges": edges.tolist(), "counts": counts.tolist()}
        try:
            labels = [f"bin_{i+1}" for i in range(bins)]
            q = pd.qcut(numeric, q=bins, labels=labels)
            equi_depth = q.value_counts().to_dict()
        except ValueError:
            equi_depth = "Insufficient numeric variety for qcut."
        return {"histogram_equi_width": equi_width, "histogram_equi_depth": equi_depth}

    @staticmethod
    def _length_metrics(series: pd.Series) -> Dict[str, float]:
        lengths = series.dropna().astype(str).apply(len)
        if lengths.empty:
            return {"length_min": np.nan, "length_max": np.nan, "length_median": np.nan, "length_mean": np.nan}
        return {
            "length_min": lengths.min(),
            "length_max": lengths.max(),
            "length_median": lengths.median(),
            "length_mean": lengths.mean(),
        }

    def plot_first_digit(self, col: str, ax: Optional[plt.Axes] = None) -> None:
        if col not in self.column_profiles:
            self.profile_column(col)
        dist = self.column_profiles[col].get("first_digit_distribution", {})

        if not dist:
            raise ValueError(f"Column '{col}' has no numeric first-digit distribution to plot.")
        
        series = pd.Series(dist).reindex([str(i) for i in range(1, 10)])
        ax = ax or plt.gca()
        series.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title(f"First-Digit Distribution – {col}")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Proportion")
        for x, y in enumerate(series.values):
            ax.text(x, y + 0.001, f"{y:.2f}", ha="center")
        ax.set_ylim(0, series.max() * 1.1)
        plt.tight_layout()

    def plot_text_frequency(self, col: str, top_n: int = 20, show_pct: bool = True, ax: Optional[plt.Axes] = None) -> None:
        s = self.df[col].astype(str).fillna("<NA>")
        counts = s.value_counts()
        total = counts.sum()
        if len(counts) > top_n:
            top = counts.iloc[:top_n]
            others_count = counts.iloc[top_n:].sum()
            counts = pd.concat([top, pd.Series({"Others": others_count})])
        counts = counts.sort_values(ascending=False)
        ax = ax or plt.gca()
        counts.plot(kind="bar", color="steelblue", ax=ax)
        ax.set_title(f"Top {top_n} categories – {col}")
        ax.set_xlabel(col)
        ylabel = "Percentage" if show_pct else "Count"
        if show_pct:
            pct = (counts / total * 100).round(2)
            ax.set_ylabel("Percentage")
            ax.bar_label(ax.containers[0], labels=[f"{p}%" for p in pct])
        else:
            ax.set_ylabel("Count")
        plt.xticks(rotation=80, ha="right")
        plt.tight_layout()

    def plot_time_counts(self, col: str, freq: str = "Y", ax: Optional[plt.Axes] = None) -> None:
        dates = pd.to_datetime(self.df[col], errors="coerce")
        if dates.isna().all():
            raise ValueError(f"Column '{col}' could not be parsed as datetime.")
        counts = dates.dt.to_period(freq).value_counts().sort_index()
        ax = ax or plt.gca()
        counts.plot(kind="line", marker="o", ax=ax)
        ax.set_title(f"Record count by {freq} – {col}")
        ax.set_xlabel("Period")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

def single_profiling(df: pd.DataFrame,
                     out_dir: str = "output/profile",
                     numeric_bins: int = 10,
                     top_n: int = 20,
                     max_text_unique: int = 200) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Calling all single-profiling tasks:
    analyze_all_columns(df)
    profiler = DataProfiler(df)
    profile_df = profiler.profile_dataframe(numeric_bins=numeric_bins)
    console.rule("[bold green]Column‑level profile[/bold green]")
    console.log(profile_df)

    for col, row in profile_df.iterrows():
        fd_dist = row.get("first_digit_distribution", {})
        if isinstance(fd_dist, dict) and fd_dist:
            plt.figure(figsize=(5, 3))
            profiler.plot_first_digit(col)
            plt.savefig(out_path / f"{col}_first_digit.png", dpi=150, bbox_inches="tight")
            plt.close()
            console.log(f"   • First‑digit plot saved for '{col}'")

    for col in df.columns:
        if col in profile_df and profile_df.loc[col].get("first_digit_distribution"):
            continue
        unique_vals = int(profile_df.loc[col, "distinct_count"])
        if unique_vals == 0:
            continue
        if unique_vals <= max_text_unique:
            plt.figure(figsize=(max(6, unique_vals * 0.25), 4))
            profiler.plot_text_frequency(col, top_n=unique_vals, show_pct=False)
            plt.savefig(out_path / f"{col}_full.png", dpi=150, bbox_inches="tight")
            plt.close()
            console.log(f"   • Full‑range plot saved for '{col}' ({unique_vals} unique values)")
        else:
            console.log(f"   • Skipped full‑range plot for '{col}' ({unique_vals} unique > {max_text_unique})")
        plt.figure(figsize=(10, 4))
        profiler.plot_text_frequency(col, top_n=top_n, show_pct=False)
        plt.savefig(out_path / f"{col}_top{top_n}.png", dpi=150, bbox_inches="tight")
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
        violation_text = row.get("Violations", "")
        if pd.isna(violation_text):
            return []

        parts = [v.strip() for v in violation_text.split("|") if v.strip()]
        pattern = r"(?P<number>\d+)\.\s+(?P<text>.+?)\s+-\s+Comments:\s+(?P<comment>.+)"

        extracted = []
        for part in parts:
            match = re.match(pattern, part)
            combined = row.drop(labels=["Violations"]).to_dict()
            combined["raw_violation"] = part

            if match:
                v = match.groupdict()
                combined["violation_number"] = v["number"]
                combined["violation_text"] = v["text"]
                combined["violation_comment"] = v["comment"]
                combined["parse_error"] = False
                combined["error_reason"] = ""
            else:
                # Attempt to detect error reason
                if not re.search(r"\d+\.", part):
                    reason = "missing violation number"
                elif "Comments:" not in part:
                    reason = "missing 'Comments:'"
                else:
                    reason = "general format mismatch"

                combined["violation_number"] = None
                combined["violation_text"] = None
                combined["violation_comment"] = None
                combined["parse_error"] = True
                combined["error_reason"] = reason

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
        df[column_name] = df[column_name].astype(str).fillna("").str.strip().str.lower()

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
        return bool(re.search(r"\b\d{1,5}-\d{1,5}\b", value))

    def tokenize_address(value):
        tokens = value.strip().lower().split()
        numeric_positions = {}
        text_tokens = []

        for i, token in enumerate(tokens):
            if re.fullmatch(
                r"[#\-]?\d+([a-z]*)?", token
            ):  # Matches 123, -02, 11a, etc.
                numeric_positions[i] = token
            else:
                text_tokens.append(token)

        return text_tokens, numeric_positions, tokens

    def detokenize_address(text_tokens, numeric_positions):
        output_tokens = []
        text_iter = iter(text_tokens)
        for i in range(
            max(numeric_positions.keys(), default=-1) + len(text_tokens) + 1
        ):
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

        df[column_name] = df[column_name].astype(str).fillna("").str.strip().str.lower()

        # Identify rows to exclude from fuzzy matching
        df[f"{column_name}_skip_fuzzy"] = df[column_name].apply(contains_numeric_range)

        tokenized = df[column_name].apply(tokenize_address)
        df[f"{column_name}_text_tokens"] = tokenized.apply(lambda x: x[0])
        df[f"{column_name}_num_positions"] = tokenized.apply(lambda x: x[1])
        df[f"{column_name}_orig_tokens"] = tokenized.apply(lambda x: x[2])
        df[f"{column_name}_text"] = df[f"{column_name}_text_tokens"].apply(
            lambda x: " ".join(x)
        )

        # Perform fuzzy matching only on rows without numeric ranges
        mask_fuzzy = ~df[f"{column_name}_skip_fuzzy"]
        values_to_match = df.loc[mask_fuzzy, f"{column_name}_text"].unique().tolist()

        reference_mapping = {}
        groups = defaultdict(list)

        for value in values_to_match:
            if value in reference_mapping:
                continue

            matches = process.extract(
                value, values_to_match, limit=10, scorer=fuzz.ratio
            )
            matches = [(match, score) for match, score in matches if score >= threshold]

            best_match = max(matches, key=lambda x: x[1])[0] if matches else value

            for match, score in matches:
                reference_mapping[match] = best_match
                groups[best_match].append(match)

        all_grouped_labels[column_name] = dict(groups)

        # Map fuzzy results
        df[f"{column_name}_text_normalised"] = df.apply(
            lambda row: (
                row[f"{column_name}_text"]
                if row[f"{column_name}_skip_fuzzy"]
                else reference_mapping.get(
                    row[f"{column_name}_text"], row[f"{column_name}_text"]
                )
            ),
            axis=1,
        )

        df[f"{column_name}_text_tokens_normalised"] = df[
            f"{column_name}_text_normalised"
        ].apply(lambda x: x.split())
        df[f"{column_name}_normalised"] = df.apply(
            lambda row: detokenize_address(
                row[f"{column_name}_text_tokens_normalised"],
                row[f"{column_name}_num_positions"],
            ),
            axis=1,
        )

        df.drop(
            columns=[
                f"{column_name}_text_tokens",
                f"{column_name}_num_positions",
                f"{column_name}_orig_tokens",
                f"{column_name}_text",
                f"{column_name}_text_normalised",
                f"{column_name}_text_tokens_normalised",
                f"{column_name}_skip_fuzzy",
            ],
            inplace=True,
        )

        return df

    for col in columns_to_clean:
        df = fuzzy_normalize_column(df, col, threshold)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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

def preprocess(df:pd.DataFrame)-> pd.DataFrame:

    df_trim = df.drop(['Inspection ID', 'AKA Name', 'Address', 'State','Facility Type', 'City', 'Inspection Type', 'Location', 'raw_violation','violation_number','violation_comment', 'parse_error','error_reason'], axis=1)

    #  renaming column names to snake_case
    COLUMN_NAMES_MAPPING = {
        'DBA Name':'dba_name',
        'License #':'license_',
        'Risk':'risk',
        'Zip':'zip',
        'Inspection Date':'inspection_date',
        'Results':'results',
        'Latitude':'latitude',
        'Longitude':'longitude',
        'violation_text':'violations',
        'Facility Type_normalised':'facility_type',
        'City_normalised':'city',
        'Inspection Type_normalised':'inspection_type',
        'Address_normalised':'address'
    }

    # renaming column names to snake_case
    df_trim = df_trim.rename(columns=COLUMN_NAMES_MAPPING)

    # fix data type
    df_trim = df_trim.astype({'zip':'Int64', 'license_':'Int64'})

    console.log("[bold green]SUCCESS[/bold green] File preprocessing completed.")

    return df_trim


def find_fds(df, algorithm_name="Default"):
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
        raise ValueError(
            f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(fd_algorithms)}"
        )


def find_afds(df: pd.DataFrame, error: float = 0.1, algorithm_name: str = "Default"):
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
        raise ValueError(
            f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(afd_algorithms)}"
        )


def find_inds(df: list[pd.DataFrame] | pd.DataFrame, algorithm_name: str = "Default"):
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
        algo.execute()

        # Filter out self-dependencies
        return [
            ind
            for ind in algo.get_inds()
            if ind.get_lhs().column_indices != ind.get_rhs().column_indices
        ]
    except AttributeError:
        raise ValueError(
            f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(ind_algorithms)}"
        )


def find_ainds(
    df: list[pd.DataFrame] | pd.DataFrame,
    algorithm_name: str = "Default",
    error: float = 0.3,
):
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
            error_threshold=error,
        )
        # Filter out self-dependencies
        return [
            ind
            for ind in algo.get_inds()
            if ind.get_lhs().column_indices != ind.get_rhs().column_indices
        ]
    except AttributeError:
        raise ValueError(
            f"Algorithm '{algorithm_name}' not found. Available algorithms: {dir(ind_algorithms)}"
        )


@dataclass
class FunctionalDependency:
    lhs: List[str]  # Left-hand side attributes
    rhs: str  # Right-hand side attribute

    def __str__(self):
        lhs_count = len(self.lhs)
        base = f"LHS={self.lhs} ({lhs_count}), RHS={self.rhs}"
        return base


@dataclass
class FunctionalDependencySet:
    dependencies: List[FunctionalDependency] = field(default_factory=list)
    validation_results: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = field(
        default_factory=dict
    )

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
                "highlights": highlights,
            }

            if self.validation_results[fd_key]["holds"]:
                # console.log(GREEN_CODE, f"FD holds: {fd.lhs} -> {fd.rhs}", DEFAULT_COLOR_CODE)
                console.log(
                    f"FD holds: {fd.lhs} -> {fd.rhs}", style="bold black on green"
                )

            else:
                console.log(
                    f"FD does not hold: {fd.lhs} -> {fd.rhs}", style="bold white on red"
                )
                console.log(
                    f"Number of clusters violating FD: {self.validation_results[fd_key]['num_violations']}"
                )

    def validate_afd(self, df: pd.DataFrame, error: float = 0.05):
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
                console.log(
                    "AFD with this error threshold holds", style="bold black on green"
                )
            else:
                console.log(
                    f"AFD with this error threshold does not hold",
                    style="bold white on red",
                )
                console.log(
                    f"But the same AFD with error threshold = {verifier.get_error()} holds."
                )

            fd_key = (tuple(fd.lhs), fd.rhs)
            self.validation_results[fd_key] = {
                "holds": fd_holds,
                "num_violations": verifier.get_num_error_clusters(),
                "highlights": highlights,
            }

            if self.validation_results[fd_key]["holds"]:
                console.log(
                    f"FD holds: {fd.lhs} -> {fd.rhs}", style="bold black on green"
                )
            else:
                console.log(
                    f"FD does not hold: {fd.lhs} -> {fd.rhs}", style="bold white on red"
                )
                console.log(
                    f"Number of clusters violating FD: {self.validation_results[fd_key]['num_violations']}"
                )

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
    validation_results: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = field(
        default_factory=dict
    )

    def add_dependency(self, lhs: List[str], rhs: List[str]):
        """Adds a new functional dependency to the set."""

        ind = InclusionDependency(lhs, rhs)

        if ind not in self.dependencies:
            self.dependencies.append(ind)

    def __len__(self):
        """Returns the number of functional dependencies."""
        return len(self.dependencies)

    def __iter__(self):
        """Allows iteration over functional dependencies."""
        return iter(self.dependencies)

    def validate_ind(self, df):
        """Validates all inclusion dependencies in the dataset and displays the results."""

        for ind in self.dependencies:
            lhs_idx = df.columns.get_indexer(ind.lhs)
            rhs_idx = df.columns.get_indexer(ind.rhs)

            console.log(f"Checking the IND: {df.columns[lhs_idx].to_list()} -> {df.columns[rhs_idx].to_list()}")

            if lhs_idx[0] == -1:
                continue

            algo = desbordante.ind_verification.algorithms.Default()
            algo.load_data(tables=[df, df])
            algo.execute(lhs_indices=lhs_idx, rhs_indices=rhs_idx)

            if algo.get_error() == 0:
                console.log("IND holds", style="bold black on green")
            else:
                console.log(
                    f"IND holds with error = {algo.get_error():.2}",
                    style="bold white on red",
                )

    def validate_aind(self, df):
        """Validates all approximate inclusion dependencies in the dataset and displays the results."""

        for ind in self.dependencies:
            lhs_idx = df.columns.get_indexer(ind.lhs)
            rhs_idx = df.columns.get_indexer(ind.rhs)

            console.log(f"Checking the IND: {df.columns[lhs_idx].to_list()} -> {df.columns[rhs_idx].to_list()}")

            if lhs_idx[0] == -1:
                continue

            algo = desbordante.aind_verification.algorithms.Default()
            algo.load_data(tables=[df, df])
            algo.execute(lhs_indices=lhs_idx, rhs_indices=rhs_idx)

            if algo.get_error() == 0:
                console.log("IND holds", style="bold black on green")
            else:
                console.log(
                    f"AIND holds with error = {algo.get_error():.2}",
                    style="bold white on red",
                )


def convert_fd(fd: desbordante.fd.FD) -> Tuple[list, str]:
    fd_str = str(fd)  # convert fd to string
    fd_str_split = fd_str.split("->")  # split fd to lhs and rhs
    lhs = fd_str_split[0].strip()
    rhs = fd_str_split[-1].strip()

    lhs_list = lhs[1:-1].split(" ")  # convert lhs to list of attributes

    return lhs_list, rhs


def convert_ind(ind: desbordante.ind.IND) -> Tuple[list, list]:
    ind_str = str(ind)
    ind_str_split = ind_str.split("->")  # split fd to lhs and rhs
    lhs = ind_str_split[0].strip()
    rhs = ind_str_split[-1].strip()

    # Regex to match content within square brackets
    pattern = r"\[([^\[\]]+)\]"

    # Find matches
    lhs_matches = re.findall(pattern, lhs)

    rhs_matches = re.findall(pattern, rhs)

    return lhs_matches, rhs_matches

def detect_pattern(value):
    if pd.isna(value):
        return "Missing"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    value = str(value).strip()
    return "".join(
        [
            "9" if c.isdigit() else "A" if c.isupper() else "a" if c.islower() else c
            for c in value
        ]
    )


def clean_display_value(val):
    return int(val) if isinstance(val, float) and val.is_integer() else val


def analyze_column_patterns(
    df,
    column_name,
    top_n=20,
    show_pattern_index=None,
    show_distinct=False,
    output_dir="column_pattern_histograms",
):
    if column_name not in df.columns:
        console.log(f"❌ Column '{column_name}' not found in DataFrame.")
        return

    if df[column_name].dropna().empty:
        console.log(f"⚠️ Column '{column_name}' is empty after dropping NA values.")
        return

    pattern_series = df[column_name].map(detect_pattern)
    pattern_counts = pattern_series.value_counts()
    console.log(
        f"✅ Column '{column_name}' has {len(pattern_counts)} unique pattern(s)."
    )

    os.makedirs(output_dir, exist_ok=True)

    summary_data = {"Pattern": pattern_counts.index, "Count": pattern_counts.values}

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

        console.log("\n📌 Distinct values (all patterns):")
        for val in sorted(all_distinct):
            console.log(f"- {val}")

    # Save Excel with error handling
    summary_df = pd.DataFrame(summary_data)
    excel_path = os.path.join(output_dir, f"{column_name}_pattern_summary.xlsx")
    try:
        summary_df.to_excel(excel_path, index=False)
        console.log(f"📄 Pattern summary saved to: {excel_path}")
    except IllegalCharacterError:
        console.log(
            "⚠️ Excel export failed due to illegal characters in the content. Summary not saved to Excel."
        )

    # Plot top N patterns
    top_patterns = pattern_counts.head(top_n)
    pattern_labels, pattern_freqs = (
        top_patterns.index.tolist(),
        top_patterns.values.tolist(),
    )

    plt.figure(figsize=(12, 6))
    plt.barh(pattern_labels, pattern_freqs, color="skyblue")
    plt.xlabel("Frequency")
    plt.ylabel("Patterns (Aa9... & Special Characters)")
    plt.title(f"Top {top_n} Value Patterns in '{column_name}'")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"{column_name}_patterns_histogram.png")
    plt.savefig(file_path)
    plt.show()
    console.log(f"📊 Histogram saved to: {file_path}")

    # console.log sample values from specific pattern indices
    if show_pattern_index is not None:
        if isinstance(show_pattern_index, int):
            show_pattern_index = [show_pattern_index]
        elif not isinstance(show_pattern_index, (list, tuple, set)):
            console.log(
                f"❌ Invalid type for show_pattern_index. Must be int or list of int."
            )
            return

        console.log("\n🔍 Sample values for selected pattern indices:")
        for idx in show_pattern_index:
            if idx < 0 or idx >= len(top_patterns):
                console.log(
                    f"❌ Invalid pattern index {idx}. Valid range: 0 to {len(top_patterns)-1}"
                )
                continue
            selected_pattern = pattern_labels[idx]
            matching_rows = df.loc[pattern_series == selected_pattern, column_name]
            cleaned_sample = matching_rows.head(10).map(clean_display_value)
            console.log(f"\n🔢 Pattern index [{idx}] - {len(matching_rows)} rows match")
            console.log(cleaned_sample.to_string(index=False))


def main(args):

    #  Read file
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(args.input)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(args.input)
    else:
        console.log(f"❌ Unsupported file format: {ext}")
        return
    
    ##### PREPROCESSING ##### WANG YU
    preprocessed_df = None
    if args.preprocess:
        columns = ['Facility Type','City' ,'Inspection Type']

        console.log("Running preprocessing on violations column:")
        violations_df = expand_violations(df)

        console.log("Running fuzzy cleaning:")
        cleaned_df, grouped_info = fuzzy_clean_columns(violations_df, columns, threshold=95)

        console.log("🚀 Running fuzzy cleaning:")
        preprocessed_df, grouped_info = fuzzy_clean_address(
            cleaned_df,
            columns_to_clean=['Address'],
            threshold=args.fuzzy_threshold,
            save_path="data/Food_Inspections_Cleaned.csv"
        )

    # Trigger analysis
    if args.analyze_column_patterns:
        console.log("📊 Analyzing column patterns:")
        analyze_column_patterns(
            df=df,
            column_name=args.column,
            top_n=args.top_n,
            show_pattern_index=args.show_pattern_index,
            show_distinct=args.show_distinct,
        )

    #### SINGLE PROFILLING #####
    if args.single_profile:
        if is_none_or_empty(preprocessed_df):
            console.log("Using the file input as the final dataframe for the specified tasks")
            preprocessed_df = df.copy()

        single_profiling(preprocessed_df)
        console.log("Running Single Profilling")

    #### RULE MINING ##### EUGENE
    if args.rule_mining:
        console.log("Running Rule Mining")

        if is_none_or_empty(preprocessed_df):
            console.log("Using the file input as the final dataframe for the specified tasks")
            preprocessed_df = df.copy()
        
        # Safely parse stringified list of column names
        columns_to_remove = ast.literal_eval(args.columns_to_remove)

        df_factorized, mappings = clean_and_factorize_data(
            preprocessed_df, columns_to_remove=columns_to_remove
        )

        run_efficient_apriori_from_df(
            df=df_factorized,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
        )

    ##### FUNCTIONAL DEPENDENCIES #####
    if args.func_dependencies:
        if is_none_or_empty(preprocessed_df):
            console.log("Using the file input as the final dataframe for the specified tasks")
            preprocessed_df = df.copy()

        final_preprocessed_df = preprocess(preprocessed_df)

        if args.func_dependencies == "default":
            console.log("Running both Default Functional Dependences:")

            fd_results = discover_fds(final_preprocessed_df)

        elif args.func_dependencies == "approximate":
            console.log("Running both Approximate Functional Dependences:")
            afd_results = discover_afds(df=final_preprocessed_df, error=0.05)

        elif args.func_dependencies == "all":
            console.log("Running both Default and Approximate Functional Dependences:")
            fd_results = discover_fds(final_preprocessed_df)
            afd_results = discover_afds(df=final_preprocessed_df, error=0.05)

    ##### INCLUSION DEPENDENCIES #####
    if args.ind_dependencies:
        if is_none_or_empty(preprocessed_df):
            console.log("Using the file input as the final dataframe for the specified tasks")
            preprocessed_df = df.copy()

        final_preprocessed_df = preprocess(preprocessed_df)

        if args.ind_dependencies == "default":
            console.log("Running the Default Inclusion Dependences:")
            run_ind(final_preprocessed_df)

        elif args.ind_dependencies == "approximate":
            console.log("Running Approximate Inclusion Dependences")
            run_aind(df=final_preprocessed_df, error=0.2)

        elif args.ind_dependencies == "all":
            console.log("Running both Default and Approximate Inclusion Dependences:")
            run_ind(final_preprocessed_df)
            run_aind(df=final_preprocessed_df, error=0.2)


if __name__ == "__main__":

    parser = create_arg_parser()

    args = parser.parse_args()

    console.log(f"Args: {args}")

    main(args)
