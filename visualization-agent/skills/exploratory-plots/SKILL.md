---
name: exploratory-plots
description: Charts for understanding a raw dataset before or instead of modeling — one column's shape, a relationship between two columns, correlations, class balance, missingness. Load this for any "plot/show/visualize" request that isn't about how a trained model performs.
---

# Exploratory Plots

Tools: `plot_histogram`, `plot_boxplot`, `plot_scatter`,
`plot_correlation_heatmap`, `plot_class_distribution`, `plot_missingness`.

## Picking the right chart

- **One numeric column's shape** → `plot_histogram`
- **One numeric column split by a category (e.g. outliers, spread by class)**
  → `plot_boxplot` (pass `by=` the categorical/target column)
- **Relationship between two numeric columns** → `plot_scatter` (pass `hue=`
  a categorical column to color by class, if there is one)
- **How every numeric column relates to every other** → `plot_correlation_heatmap`
- **Is the target imbalanced** → `plot_class_distribution` (visual companion
  to the classification agent's `check_imbalance`)
- **Where are the missing values** → `plot_missingness`

If a column doesn't exist or isn't numeric where a numeric one is required,
say so plainly instead of guessing a different column.

Each tool returns a `markdown` field — paste it verbatim on its own line so
the chart renders inline; see the base SKILL.md's Output format.
