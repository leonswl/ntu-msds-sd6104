# coding: utf-8
"""
A lightweight, general‑purpose data‑profiling utility for tabular datasets.

Example
-------
>>> from data_profiler import DataProfiler
>>> profiler = DataProfiler(df)
>>> profiler.profile_dataframe()        # build column‑level metrics
>>> profiler.plot_first_digit("age")   # visualise Benford‑style first‑digit distribution
>>> profiler.plot_text_frequency("city", top_n=15)
>>> profiler.plot_time_counts("invoice_date")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "DataProfiler",
]


class DataProfiler:
    """Profile and visualise single columns of a :class:`pandas.DataFrame`."""

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df.copy()
        self.column_profiles: Dict[str, Dict[str, Union[int, float, dict]]] = {}

    # ---------------------------------------------------------------------
    # Core analysis helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _first_digit_distribution(series: pd.Series) -> pd.Series:
        """Return the distribution of the *first* significant digit (1‑9).

        Leading signs, spaces and zeros are ignored so that the digit reflects
        the true magnitude. Empty series returns an empty :class:`pd.Series`.
        """
        s = series.dropna()
        if s.empty:
            return pd.Series(dtype=float)

        first_digits: List[str] = []
        for val in s.astype(str):
            val = val.lstrip(" -+0")  # strip sign, spaces, and leading zeros
            if val and val[0].isdigit():
                first_digits.append(val[0])

        if not first_digits:
            return pd.Series(dtype=float)

        # pandas.value_counts(list) is deprecated; wrap in a Series first.
        counts = pd.Series(first_digits).value_counts(sort=False).sort_index()
        return counts / counts.sum()

    @staticmethod
    def _numeric_histograms(
        series: pd.Series, bins: int = 10
    ) -> Dict[str, Union[dict, str]]:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return {}

        # equi‑width
        counts, edges = np.histogram(numeric, bins=bins)
        equi_width = {"bin_edges": edges.tolist(), "counts": counts.tolist()}

        # equi‑depth (quantiles)
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

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def profile_column(self, col: str, numeric_bins: int = 10) -> Dict[str, Union[int, float, dict]]:
        """Profile *one* column and cache the result under :pyattr:`column_profiles`."""
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

        # Numeric?
        numeric = pd.to_numeric(s, errors="coerce")
        is_numeric = (numeric.notna().sum() / num_rows) > 0.9 if num_rows else False

        if is_numeric:
            profile["quartiles"] = numeric.quantile([0.25, 0.5, 0.75]).to_dict()
            profile.update(self._numeric_histograms(s, bins=numeric_bins))
            profile["first_digit_distribution"] = self._first_digit_distribution(numeric).to_dict()
        else:
            profile.update(self._length_metrics(s))
            profile["histogram"] = s.value_counts(dropna=False).to_dict()

        self.column_profiles[col] = profile
        return profile

    def profile_dataframe(self, numeric_bins: int = 10) -> pd.DataFrame:
        """Profile *all* columns and return a tidy :class:`pd.DataFrame`."""
        for col in self.df.columns:
            self.profile_column(col, numeric_bins=numeric_bins)
        return pd.DataFrame(self.column_profiles).T

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_first_digit(self, col: str, ax: Optional[plt.Axes] = None) -> None:
        """Bar‑plot of first‑digit distribution for *numeric* column *col*."""
        if col not in self.column_profiles:
            self.profile_column(col)
        dist = self.column_profiles[col].get("first_digit_distribution", {})
        if not dist:
            raise ValueError(f"Column '{col}' has no numeric first‑digit distribution to plot.")

        series = pd.Series(dist).reindex([str(i) for i in range(1, 10)])
        ax = ax or plt.gca()
        series.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title(f"First‑Digit Distribution – {col}")
        ax.set_xlabel("Digit")
        ax.set_ylabel("Proportion")
        for x, y in enumerate(series.values):
            ax.text(x, y + 0.001, f"{y:.2f}", ha="center")
        ax.set_ylim(0, series.max() * 1.1)
        plt.tight_layout()

    def plot_text_frequency(
        self,
        col: str,
        top_n: int = 20,
        show_pct: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """Plot frequency of text categories, collapsing low‑frequency values into *Others*."""
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

    def plot_time_counts(
        self,
        col: str,
        freq: str = "Y",
        ax: Optional[plt.Axes] = None,
    ) -> None:
        """Plot number of records aggregated by *freq* (default yearly) for datetime column."""
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

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def describe(self) -> None:
        """Pretty‑print a concise summary of stored column profiles."""
        if not self.column_profiles:
            self.profile_dataframe()
        summary = (
            pd.DataFrame(self.column_profiles)
            .T[["num_rows", "null_pct", "distinct_count", "constancy"]]
            .sort_index()
        )
        print(summary.to_string(float_format="{:.2%}".format))
