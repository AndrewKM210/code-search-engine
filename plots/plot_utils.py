"""Shared plot style and theme utilities for notebooks."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

# --- Figure size ---
FIGSIZE = (8, 5)

# --- Color constants ---
BASE_LIGHT = sns.color_palette("Set2")[-1]
BASE_DARK = sns.color_palette("Set2")[-1]
COLOR1_LIGHT = sns.color_palette()[0]
COLOR1_DARK = "#58A6FF"
COLOR2_LIGHT = sns.color_palette()[1]
COLOR2_DARK = "#F78166"
SUCCESS = "#3FB950"
FAIL = "#FF7B72"

# --- Dark theme (GitHub dark) ---
GITHUB_BG_DARK = "#0D1117"
GITHUB_TEXT_DARK = "#C9D1D9"

DARK_PARAMS = {
    "figure.facecolor": GITHUB_BG_DARK,
    "axes.facecolor": GITHUB_BG_DARK,
    "text.color": GITHUB_TEXT_DARK,
    "axes.labelcolor": GITHUB_TEXT_DARK,
    "xtick.color": GITHUB_TEXT_DARK,
    "ytick.color": GITHUB_TEXT_DARK,
    "axes.edgecolor": GITHUB_TEXT_DARK,
    "grid.color": "#21262D",
    "grid.alpha": 0.5,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}

# --- Light theme ---
GITHUB_BG_LIGHT = "#FFFFFF"
GITHUB_TEXT_LIGHT = "#24292F"

LIGHT_PARAMS = {
    "figure.facecolor": GITHUB_BG_LIGHT,
    "axes.facecolor": GITHUB_BG_LIGHT,
    "text.color": GITHUB_TEXT_LIGHT,
    "axes.labelcolor": GITHUB_TEXT_LIGHT,
    "xtick.color": GITHUB_TEXT_LIGHT,
    "ytick.color": GITHUB_TEXT_LIGHT,
    "axes.edgecolor": GITHUB_TEXT_LIGHT,
    "grid.color": "#D0D7DE",
    "grid.alpha": 0.8,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


@dataclass
class Palette:
    """Theme-resolved color palette."""

    base: str
    color1: str
    color2: str
    success: str = SUCCESS
    fail: str = FAIL


def get_palette(theme: str) -> "Palette":
    """Returns colors resolved for the given theme.

    Args:
        theme: Either 'dark' or 'light'.

    Returns:
        A Palette with all colors set for that theme.

    Raises:
        ValueError: If theme is not 'dark' or 'light'.
    """
    if theme == "dark":
        return Palette(base=BASE_DARK, color1=COLOR1_DARK, color2=COLOR2_DARK)
    if theme == "light":
        return Palette(
            base=BASE_LIGHT, color1=COLOR1_LIGHT, color2=COLOR2_LIGHT
        )
    raise ValueError("theme must be 'dark' or 'light'")


def set_theme(mode: str = "dark") -> None:
    """Switches the plot style between 'dark' (GitHub) and 'light' (standard).

    Args:
        mode: Theme mode, either 'dark' or 'light'.

    Raises:
        ValueError: If mode is not 'dark' or 'light'.
    """
    if mode == "dark":
        sns.set_theme(style="darkgrid")
        plt.rcParams.update(DARK_PARAMS)
    elif mode == "light":
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(LIGHT_PARAMS)
    else:
        raise ValueError("Mode must be 'dark' or 'light'")


def save_fig(save_path: str | None) -> None:
    """Saves the current figure to disk if save_path is given.

    Args:
        save_path: Destination path, or None to skip saving.
    """
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            transparent=False,
            facecolor=plt.rcParams["figure.facecolor"],
        )


def show_fig() -> None:
    """Calls plt.show() only when running inside a Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            plt.show()
    except ImportError:
        pass
