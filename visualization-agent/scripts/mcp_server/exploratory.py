"""visualization agent — `exploratory` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _save  # noqa: F401


@mcp.tool()
def plot_histogram(path: str, column: str, out_path: str, bins: int = 30) -> str:
    """Histogram (with KDE) of a numeric column."""
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    title = f"Distribution of {column}"
    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    saved = _save(fig, out_path, title)
    return json.dumps({**saved, "column": column, "n": int(df[column].notna().sum())})


@mcp.tool()
def plot_boxplot(path: str, column: str, out_path: str, by: str = "") -> str:
    """Boxplot of a numeric column, optionally grouped by a categorical column (e.g. the target)."""
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    title = f"{column} by {by}" if by else f"{column} distribution"
    if by:
        sns.boxplot(data=df, x=by, y=column, ax=ax)
    else:
        sns.boxplot(data=df, y=column, ax=ax)
    ax.set_title(title)
    saved = _save(fig, out_path, title)
    return json.dumps({**saved, "column": column, "by": by or None})


@mcp.tool()
def plot_scatter(path: str, x: str, y: str, out_path: str, hue: str = "") -> str:
    """Scatter plot of two numeric columns, optionally colored by a third (categorical) column."""
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    title = f"{y} vs {x}"
    sns.scatterplot(data=df, x=x, y=y, hue=(hue or None), ax=ax)
    ax.set_title(title)
    saved = _save(fig, out_path, title)
    return json.dumps({**saved, "x": x, "y": y, "hue": hue or None})


@mcp.tool()
def plot_correlation_heatmap(path: str, out_path: str) -> str:
    """Correlation heatmap across all numeric columns."""
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(
        figsize=(max(5, len(corr.columns) * 0.6), max(4, len(corr.columns) * 0.5))
    )
    title = "Correlation heatmap"
    sns.heatmap(
        corr, annot=len(corr.columns) <= 15, fmt=".2f", cmap="coolwarm", center=0, ax=ax
    )
    ax.set_title(title)
    saved = _save(fig, out_path, title)
    return json.dumps({**saved, "columns": list(corr.columns)})


@mcp.tool()
def plot_class_distribution(path: str, target: str, out_path: str) -> str:
    """Bar chart of class counts for the target column — the visual companion to check_imbalance."""
    df = pd.read_csv(path)
    counts = df[target].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    title = f"Class distribution: {target}"
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("count")
    saved = _save(fig, out_path, title)
    return json.dumps({**saved, "counts": counts.to_dict()})


@mcp.tool()
def plot_missingness(path: str, out_path: str) -> str:
    """Bar chart of missing-value percentage per column."""
    df = pd.read_csv(path)
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    title = "Missingness by column"
    fig, ax = plt.subplots(figsize=(max(5, len(missing) * 0.5), 4))
    if len(missing) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.barplot(x=missing.index, y=missing.values, ax=ax)
        ax.set_ylabel("% missing")
        plt.xticks(rotation=45, ha="right")
    ax.set_title(title)
    saved = _save(fig, out_path, title)
    return json.dumps({**saved, "missing_pct": missing.to_dict()})
