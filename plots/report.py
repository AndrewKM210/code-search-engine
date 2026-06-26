# Cells are delimited by `# %%` (VS Code Jupyter) and `# %% [markdown]`.
# To open as a notebook: install jupytext, then run `jupytext --to notebook plots/report.py`
# To run as a script from the project root: `python plots/report.py`

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_utils import (
    FIGSIZE,
    get_palette,
    save_fig,
    set_theme,
    show_fig,
)


def plot_loss(
    dfs: list[pd.DataFrame],
    colors: list[str],
    labels: list[str],
    theme: str = "dark",
    save_path: str = None,
):
    set_theme(theme)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for df, c, label in zip(dfs, colors, labels, strict=False):
        df["Smooth Loss"] = df["Loss"].ewm(span=20).mean()
        sns.lineplot(
            df, x="Step", y="Loss", c=c, alpha=0.25, label=label, ax=ax
        )
        sns.lineplot(df, x="Step", y="Smooth Loss", c=c, ax=ax)
    ax.set_title("Fine-tuning with Multiple Negatives Ranking Loss")
    ax.legend(title="")
    plt.tight_layout()
    save_fig(save_path)
    show_fig()


def plot_metrics_barplot(
    df: pd.DataFrame,
    colors: list[str],
    ylims: tuple = (0, 1),
    theme: str = "dark",
    save_path: str = None,
):
    set_theme(theme)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(df, x="Metric", y="Value", hue="Model", palette=colors, ax=ax)
    ax.set_ylim(ylims)
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".3f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )
    ax.set_title("Evaluation Metrics for Different Models on CoSQA Dataset")
    ax.legend(title="")
    plt.tight_layout()
    save_fig(save_path)
    show_fig()


def plot_metrics_improvement(
    df: pd.DataFrame, color: str, theme: str = "dark", save_path: str = None
):
    set_theme(theme)
    df_diff = df.pivot(index="Metric", columns="Model", values="Value")
    df_diff["Absolute_Gain"] = (
        df_diff["Fine-Tuned Model"] - df_diff["Base Model"]
    )
    df_diff["Percent_Gain"] = (
        df_diff["Absolute_Gain"] / df_diff["Base Model"]
    ) * 100
    df_plot = df_diff.reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x="Metric", y="Percent_Gain", data=df_plot, ax=ax)
    ax.set_title("Percentage Improvement over Base Model", fontsize=16)
    ax.set_ylabel("Percentage Gain (%)")
    plt.tight_layout()
    save_fig(save_path)
    show_fig()


# %% [markdown]
# # Fine-Tuning

# %%
df = pd.read_csv("results/losses.csv")
for theme in ("dark", "light"):
    p = get_palette(theme)
    plot_loss(
        [df],
        colors=[p.color1],
        labels=["Fine-Tuned Model"],
        theme=theme,
        save_path=f"assets/loss_ft_{theme}.png",
    )

# %% [markdown]
# # Evaluation

# %%
df = pd.read_csv("results/evaluation.csv").drop(
    columns=["Avg. Query Time (ms)"]
)
df = df.melt(
    id_vars="Model",
    value_vars=["MRR@10", "Recall@10", "nDCG@10"],
    var_name="Metric",
    value_name="Value",
)

ylims = (0.8, 1.05)
for theme in ("dark", "light"):
    p = get_palette(theme)
    plot_metrics_barplot(
        df,
        colors=[p.base, p.color1],
        ylims=ylims,
        theme=theme,
        save_path=f"assets/eval_ft_{theme}.png",
    )

# %%
df = pd.read_csv("results/evaluation.csv").drop(
    columns=["Avg. Query Time (ms)"]
)
df = df.melt(
    id_vars="Model",
    value_vars=["MRR@10", "Recall@10", "nDCG@10"],
    var_name="Metric",
    value_name="Value",
)
for theme in ("dark", "light"):
    p = get_palette(theme)
    plot_metrics_improvement(
        df,
        color=p.color1,
        theme=theme,
        save_path=f"assets/improvement_ft_{theme}.png",
    )

# %% [markdown]
# # Fine-Tuning with Function Names

# %%
dfs = [
    pd.read_csv("results/losses.csv"),
    pd.read_csv("results/losses_fn_names.csv"),
]
for theme in ("dark", "light"):
    p = get_palette(theme)
    plot_loss(
        dfs,
        colors=[p.color1, p.color2],
        labels=["Fine-Tuned Model", "Fine-Tuned Model (Function Names)"],
        theme=theme,
        save_path=f"assets/loss_fn_{theme}.png",
    )

# %%
df = pd.read_csv("results/evaluation_fn_names.csv")
df["Avg. Query Time (ms/10)"] = df["Avg. Query Time (ms)"] / 10
df = df.melt(
    id_vars="Model",
    value_vars=["MRR@10", "Recall@10", "nDCG@10", "Avg. Query Time (ms/10)"],
    var_name="Metric",
    value_name="Value",
)
for theme in ("dark", "light"):
    p = get_palette(theme)
    plot_metrics_barplot(
        df,
        colors=[p.color1, p.color2],
        ylims=(0.6, 1.1),
        theme=theme,
        save_path=f"assets/eval_fn_{theme}.png",
    )
