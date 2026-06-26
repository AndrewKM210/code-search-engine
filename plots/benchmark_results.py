# Cells are delimited by `# %%` (VS Code Jupyter) and `# %% [markdown]`.
# To open as a notebook: install jupytext, then run `jupytext --to notebook plots/benchmark_results.py`
# To run as a script from the project root: `python plots/benchmark_results.py`

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils import (
    FIGSIZE,
    Palette,
    get_palette,
    save_fig,
    set_theme,
    show_fig,
)


def plot_benchmark_scores(
    df: pd.DataFrame,
    palette: Palette,
    theme: str = "dark",
    save_path: str = None,
):
    set_theme(theme)
    melted = df.melt(
        id_vars=["run"],
        value_vars=["correctness_pct", "faithfulness_pct"],
        var_name="metric",
        value_name="score",
    )
    melted["metric"] = (
        melted["metric"]
        .str.replace("_pct", "")
        .str.replace("_", " ")
        .str.capitalize()
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        data=melted,
        x="run",
        y="score",
        hue="metric",
        palette=[palette.success, palette.color1],
        ax=ax,
    )
    ax.set_ylim(0, 110)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Score (%)")
    ax.set_title("Benchmark: Correctness and Faithfulness by Agent")
    ax.tick_params(axis="x", rotation=20)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.0f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=9,
            )
    ax.legend(title="")
    plt.tight_layout()
    save_fig(save_path)
    show_fig()


def plot_benchmark_scatter(
    df: pd.DataFrame, theme: str = "dark", save_path: str = None
):
    set_theme(theme)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(
        data=df,
        x="correctness_pct",
        y="faithfulness_pct",
        hue="agent",
        style="agent",
        s=120,
        ax=ax,
    )

    xytexts = [(10, 0), (10, 5), (10, -5), (10, 0)]
    for (_, row), xytext in zip(df.iterrows(), xytexts, strict=False):
        ax.annotate(
            row["run"],
            (row["correctness_pct"], row["faithfulness_pct"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
        )
    ax.set_xlabel("Correctness (%)")
    ax.set_ylabel("Faithfulness (%)")
    ax.set_title("Correctness vs Faithfulness per Agent", fontsize=16)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(title="Agent")
    plt.tight_layout()
    save_fig(save_path)
    show_fig()


def plot_failure_modes(
    df_full: pd.DataFrame,
    palette: Palette,
    theme: str = "dark",
    save_path: str = None,
):
    set_theme(theme)
    STEP_LIMIT_MSG = "Could not produce an answer within the step limit"
    stats = defaultdict(
        lambda: {"Correct": 0, "Wrong answer": 0, "Step limit": 0}
    )
    for _, row in df_full.iterrows():
        a = row["agent"]
        if STEP_LIMIT_MSG in str(row["answer"]):
            stats[a]["Step limit"] += 1
        elif row["correctness"]:
            stats[a]["Correct"] += 1
        else:
            stats[a]["Wrong answer"] += 1

    rows = [
        {"Agent": a, "Outcome": outcome, "Count": n}
        for a, counts in stats.items()
        for outcome, n in counts.items()
    ]
    df_plot = pd.DataFrame(rows)
    agent_order = [
        "phi3-baseline",
        "llama3.2:3b-tool-loop",
        "llama3.2:3b-tool-loop-nudged",
        "qwen2.5-coder:3b-tool-loop",
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    outcome_colors = {
        "Correct": palette.success,
        "Wrong answer": palette.color1,
        "Step limit": palette.fail,
    }
    bottoms = {a: 0 for a in agent_order}
    bg = plt.rcParams["figure.facecolor"]

    for outcome, color in outcome_colors.items():
        subset = df_plot[df_plot["Outcome"] == outcome].set_index("Agent")
        counts = [
            subset.loc[a, "Count"] if a in subset.index else 0
            for a in agent_order
        ]
        bars = ax.bar(
            agent_order,
            counts,
            bottom=[bottoms[a] for a in agent_order],
            label=outcome,
            color=color,
        )
        for bar, count in zip(bars, counts, strict=False):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_y() + count / 2.0,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=bg,
                    fontweight="bold",
                )
        for a, count in zip(agent_order, counts, strict=False):
            bottoms[a] += count

    ax.set_title("Failure Mode Breakdown by Agent", fontsize=16)
    ax.set_ylabel("Questions (out of 24)")
    ax.set_xlabel("Agent")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    plt.tight_layout()
    save_fig(save_path)
    show_fig()


# %% [markdown]
# ## Score Summary

# %%
df = pd.read_csv("results/20260625-1903_benchmark.csv")
df_full = pd.read_csv("results/20260625-1903_benchmark_full.csv")

for theme in ("dark", "light"):
    p = get_palette(theme)
    plot_benchmark_scores(
        df,
        palette=p,
        theme=theme,
        save_path=f"assets/benchmark_scores_{theme}.png",
    )

# %% [markdown]
# ## Correctness vs Faithfulness

# %%
for theme in ("dark", "light"):
    plot_benchmark_scatter(
        df, theme=theme, save_path=f"assets/benchmark_scatter_{theme}.png"
    )

# %% [markdown]
# ## Failure Mode Breakdown

# %%
for theme in ("dark", "light"):
    p = get_palette(theme)
    plot_failure_modes(
        df_full,
        palette=p,
        theme=theme,
        save_path=f"assets/benchmark_failures_{theme}.png",
    )
