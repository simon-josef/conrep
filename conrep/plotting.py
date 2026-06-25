"""
plotting.py
-----------
Publication-ready figure functions for the three analytical demonstrations.

All functions return the figure object so the caller can save or further
modify. Default styling uses Charter at 7pt with dark spine colors, consistent
with the companion paper figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, pearsonr
from scipy import stats


def set_style():
    """Apply the default rcParams for the conrep figures."""
    rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Charter"],
        "font.size":         7,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def plot_partial_regression(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    control_col: str,
    weight_col: str,
    ylabel: str = None,
    xlabel: str = None,
    n_boot: int = 2000,
    n_bins: int = 100,
    figsize: tuple = (5.0, 3.5),
    save_path: str = None,
) -> plt.Figure:
    """Weighted regression plot with bootstrapped confidence band.

    If control_col is given, residualizes both x_col and y_col against it
    using weighted OLS, then fits the regression of the residuals (partial
    regression). If control_col is None, runs a simple weighted regression
    of y_col on x_col directly, with no partialling.

    Scatter shows equal-frequency binned means for visual clarity at large N.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain y_col, x_col, weight_col, and control_col if given.
    y_col : str
        Outcome variable (e.g., 'S_b').
    x_col : str
        Predictor variable (e.g., 'concreteness').
    control_col : str or None
        Control variable to partial out (e.g., 'synset_count').
        Set to None to run a simple regression without partialling.
    weight_col : str
        Observation weights (e.g., 'N').
    ylabel, xlabel : str or None
        Axis labels. Defaults to y_col and x_col (with "(residualized)"
        appended to xlabel if control_col is given).
    n_boot : int
        Bootstrap resamples for the confidence band.
    n_bins : int
        Number of equal-frequency bins for the scatter overlay.
    figsize : tuple
    save_path : str or None
        If provided, saves the figure as PDF to this path.

    Returns
    -------
    fig : plt.Figure
    stats_dict : dict with keys r, p, slope, r_ci_lo, r_ci_hi
    """
    required_cols = [y_col, x_col, weight_col] + ([control_col] if control_col else [])
    df = df.dropna(subset=required_cols).copy()
    x  = df[x_col].astype(float).values
    y  = df[y_col].astype(float).values
    w  = df[weight_col].astype(float).values

    if control_col is not None:
        z       = df[control_col].astype(float).values
        fit_y   = np.polyfit(z, y, 1, w=w)
        resid_y = y - (fit_y[0] * z + fit_y[1])
        fit_x   = np.polyfit(z, x, 1, w=w)
        resid_x = x - (fit_x[0] * z + fit_x[1])
    else:
        resid_y = y
        resid_x = x

    fit              = np.polyfit(resid_x, resid_y, 1, w=w)
    slope, intercept = fit

    w_norm = w / w.sum()
    mx     = np.sum(w_norm * resid_x)
    my     = np.sum(w_norm * resid_y)
    cov    = np.sum(w_norm * (resid_x - mx) * (resid_y - my))
    sx     = np.sqrt(np.sum(w_norm * (resid_x - mx) ** 2))
    sy     = np.sqrt(np.sum(w_norm * (resid_y - my) ** 2))
    r      = cov / (sx * sy)
    t_stat = r * np.sqrt((len(resid_x) - 2) / (1 - r ** 2))
    p      = 2 * stats.t.sf(np.abs(t_stat), df=len(resid_x) - 2)

    x_line = np.linspace(resid_x.min(), resid_x.max(), 300)
    y_line = slope * x_line + intercept

    rng        = np.random.default_rng(42)
    boot_lines = np.zeros((n_boot, len(x_line)))
    boot_r     = np.zeros(n_boot)
    for i in range(n_boot):
        idx       = rng.choice(len(resid_x), len(resid_x), replace=True)
        b         = np.polyfit(resid_x[idx], resid_y[idx], 1, w=w[idx])
        boot_lines[i] = b[0] * x_line + b[1]
        wb      = w[idx] / w[idx].sum()
        rx, ry  = resid_x[idx], resid_y[idx]
        mx2     = np.sum(wb * rx)
        my2     = np.sum(wb * ry)
        cov2    = np.sum(wb * (rx - mx2) * (ry - my2))
        sx2     = np.sqrt(np.sum(wb * (rx - mx2) ** 2))
        sy2     = np.sqrt(np.sum(wb * (ry - my2) ** 2))
        boot_r[i] = cov2 / (sx2 * sy2)

    ci_lo   = np.percentile(boot_lines, 2.5,  axis=0)
    ci_hi   = np.percentile(boot_lines, 97.5, axis=0)
    r_ci_lo = np.percentile(boot_r, 2.5)
    r_ci_hi = np.percentile(boot_r, 97.5)

    bin_idx = pd.qcut(resid_x, q=n_bins, labels=False, duplicates="drop")
    bin_x   = np.array([resid_x[bin_idx == b].mean() for b in np.unique(bin_idx)])
    bin_y   = np.array([resid_y[bin_idx == b].mean() for b in np.unique(bin_idx)])

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.16)

    ax.scatter(bin_x, bin_y, s=15, color="#3b6fd4", alpha=0.5,
               edgecolors="none", zorder=4)
    ax.fill_between(x_line, ci_lo, ci_hi, color="#c0392b", alpha=0.15,
                    linewidth=0, zorder=2)
    ax.plot(x_line, y_line, color="#c0392b", lw=1.4, zorder=3)

    default_xlabel = f"{x_col} (residualized)" if control_col is not None else x_col
    ax.set_xlabel(xlabel or default_xlabel, fontsize=8, labelpad=3)
    ax.set_ylabel(ylabel or y_col, fontsize=8, labelpad=3)
    ax.set_ylim(np.percentile(resid_y, 2), np.percentile(resid_y, 98))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#555555", labelsize=7, width=0.6, length=2.5)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight",
                    dpi=300, facecolor="white")

    result_stats = dict(r=r, p=p, slope=slope, r_ci_lo=r_ci_lo, r_ci_hi=r_ci_hi)
    return fig, result_stats


def plot_mantel_results(
    df_results: pd.DataFrame,
    panel_label: str = "",
    show_legend: bool = True,
    figsize: tuple = None,
    save_path: str = None,
) -> plt.Figure:
    """Lollipop chart of per-concept Mantel r values.

    Parameters
    ----------
    df_results : pd.DataFrame
        Output of mantel_test(). Must contain columns: cue, r, p_mantel.
    panel_label : str
        Panel letter for multi-panel figures (e.g., 'a').
    show_legend : bool
        Whether to draw the significance legend.
    figsize : tuple or None
        Auto-sized to number of concepts if None.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    df_plot  = df_results.sort_values("r", ascending=True).reset_index(drop=True)
    n        = len(df_plot)
    y        = range(n)
    sig_mask = df_plot["p_mantel"] < 0.05
    ns_idx   = [i for i, s in enumerate(sig_mask) if not s]
    sig_idx  = [i for i, s in enumerate(sig_mask) if s]

    if figsize is None:
        figsize = (5.5, n * 0.26 + 0.8)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hlines(y, 0, df_plot["r"], color="#aaaaaa", linewidth=0.6)
    ax.scatter(df_plot.loc[~sig_mask, "r"], ns_idx,
               color="#777777", s=30, zorder=3, linewidths=0)
    ax.scatter(df_plot.loc[sig_mask,  "r"], sig_idx,
               color="#111111", s=30, zorder=4, linewidths=0)
    ax.axvline(0, color="#333333", linewidth=1.0, zorder=2)

    for i, row in df_plot.iterrows():
        stars = _sig_stars(row["p_mantel"])
        if stars:
            x_pos = row["r"] + 0.003 if row["r"] >= 0 else row["r"] - 0.003
            ha    = "left" if row["r"] >= 0 else "right"
            ax.text(x_pos, i, stars, va="center", ha=ha,
                    fontsize=8, color="#222222")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df_plot["cue"].tolist(), fontsize=9,
                       color="#111111", style="italic")
    ax.set_xlabel("Mantel $r$", fontsize=9, labelpad=3, color="#222222")
    ax.tick_params(axis="x", labelsize=8, width=0.6, length=2.5, colors="#111111")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(left=False)

    if show_legend:
        ax.text(
            0.98, 0.02,
            "$^{*}p < .05$\n$^{**}p < .01$\n$^{***}p < .001$\n(Mantel test)",
            transform=ax.transAxes, fontsize=6.5, color="#222222",
            va="bottom", ha="right", linespacing=1.6,
            bbox=dict(boxstyle="square,pad=0.4", facecolor="white",
                      edgecolor="#aaaaaa", linewidth=0.5)
        )

    if panel_label:
        ax.text(-0.14, 1.02, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left",
                color="#111111")

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    return fig


def plot_mantel_joint(
    df_results_a: pd.DataFrame,
    df_results_b: pd.DataFrame,
    label_a: str = "a",
    label_b: str = "b",
    save_path: str = None,
) -> plt.Figure:
    """Two-panel lollipop chart with shared x-axis range.

    Parameters
    ----------
    df_results_a, df_results_b : pd.DataFrame
        Outputs of mantel_test() for two different predictors.
    label_a, label_b : str
        Panel labels.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    n_max = max(len(df_results_a), len(df_results_b))
    fig_h = n_max * 0.38 + 0.8

    fig, axes = plt.subplots(1, 2, figsize=(11, fig_h))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(wspace=0.4)

    def _panel(ax, df_res, label, legend):
        df_plot  = df_res.sort_values("r", ascending=True).reset_index(drop=True)
        n        = len(df_plot)
        y        = range(n)
        sig_mask = df_plot["p_mantel"] < 0.05
        ns_idx   = [i for i, s in enumerate(sig_mask) if not s]
        sig_idx  = [i for i, s in enumerate(sig_mask) if s]

        ax.hlines(y, 0, df_plot["r"], color="#aaaaaa", linewidth=0.6)
        ax.scatter(df_plot.loc[~sig_mask, "r"], ns_idx,
                   color="#777777", s=30, zorder=3, linewidths=0)
        ax.scatter(df_plot.loc[sig_mask,  "r"], sig_idx,
                   color="#111111", s=30, zorder=4, linewidths=0)
        ax.axvline(0, color="#333333", linewidth=1.0, zorder=2)

        for i, row in df_plot.iterrows():
            stars = _sig_stars(row["p_mantel"])
            if stars:
                x_pos = row["r"] + 0.003 if row["r"] >= 0 else row["r"] - 0.003
                ha    = "left" if row["r"] >= 0 else "right"
                ax.text(x_pos, i, stars, va="center", ha=ha,
                        fontsize=8, color="#222222")

        ax.set_yticks(list(y))
        ax.set_yticklabels(df_plot["cue"].tolist(), fontsize=11,
                           color="#111111", style="italic")
        ax.set_xlabel("Mantel $r$", fontsize=9, labelpad=3, color="#222222")
        ax.tick_params(axis="x", labelsize=8, width=0.6, length=2.5,
                       colors="#111111")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#888888")
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(left=False)

        if legend:
            ax.text(
                0.98, 0.02,
                "$^{*}p < .05$\n$^{**}p < .01$\n$^{***}p < .001$\n(Mantel test)",
                transform=ax.transAxes, fontsize=6.5, color="#222222",
                va="bottom", ha="right", linespacing=1.6,
                bbox=dict(boxstyle="square,pad=0.4", facecolor="white",
                          edgecolor="#aaaaaa", linewidth=0.5)
            )

        ax.text(-0.14, 1.02, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left",
                color="#111111")

        return df_plot["r"].min(), df_plot["r"].max()

    r_min_a, r_max_a = _panel(axes[0], df_results_a, label_a, legend=False)
    r_min_b, r_max_b = _panel(axes[1], df_results_b, label_b, legend=True)

    x_min = min(r_min_a, r_min_b) - 0.01
    x_max = max(r_max_a, r_max_b) + 0.01
    axes[0].set_xlim(x_min, x_max)
    axes[1].set_xlim(x_min, x_max)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    return fig


def plot_distribution_comparison(
    dist_data: dict,
    group_a_label: str = "Reference\u2013reference",
    group_b_label: str = "Target\u2013reference",
    figsize: tuple = (8, None),
    save_path: str = None,
) -> plt.Figure:
    """Ridge-style KDE plot comparing two pairwise dissimilarity distributions per concept.

    Generic comparison plot used by both Avenue 3 subsections:
    - Subsection A (subpopulation within dataset): group_a = reference,
      group_b = cross (target-to-reference) distances.
    - Subsection B (external data, e.g. LLM): group_a = human-human,
      group_b = LLM-human distances.

    Each panel shows normalized KDE curves for both groups. Tick marks at
    the panel baseline indicate the group means. Permutation p-values are
    annotated on the right margin.

    Parameters
    ----------
    dist_data : dict
        Output of run_deviation_test(..., return_distributions=True) or
        compare_llm_human(). Keys are concept strings; values contain
        'hh', 'lh' (np.ndarray) and 'p' (float).
    group_a_label : str
        Legend label for the 'hh' distribution (e.g. "Reference" or "Human\u2013human").
    group_b_label : str
        Legend label for the 'lh' distribution (e.g. "Target" or "LLM\u2013human").
    figsize : tuple
        Height is auto-computed from concept count if the second element is None.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    GROUP_B_COLOR = "#4a6fa5"
    GROUP_A_FACE  = "#b0b0b0"
    GROUP_A_EDGE  = "#888888"
    GROUP_B_EDGE  = GROUP_B_COLOR

    n_concepts  = len(dist_data)
    fig_height  = figsize[1] if figsize[1] is not None else round(n_concepts * 0.35 + 1.1, 1)

    fig, axes = plt.subplots(n_concepts, 1, figsize=(figsize[0], fig_height))
    if n_concepts == 1:
        axes = [axes]
    # Bottom margin (xlabel + legend) needs a roughly constant absolute height
    # regardless of how many panels there are, so it's computed as an absolute
    # inch budget converted to a fraction of the total figure height, rather
    # than a fixed fraction (which would leave a huge gap for many panels and
    # be cramped for very few).
    bottom_budget_in = 0.85
    bottom_frac = min(0.35, bottom_budget_in / fig_height)
    fig.subplots_adjust(hspace=0.0, left=0.15, right=0.82, top=0.97, bottom=bottom_frac)

    x_min  = 0.0
    x_max  = np.ceil(max(d["hh"].max() for d in dist_data.values()) / 0.05) * 0.05
    x      = np.linspace(x_min, x_max, 800)
    xticks = np.arange(0.0, x_max + 0.05, 0.05)

    axes[0].text(1.04, 1.35, r"$p$",
                 transform=axes[0].transAxes,
                 ha="left", va="center", fontsize=6.5, color="#333333")

    for ax, (concept, d) in zip(axes, dist_data.items()):
        kde_a = gaussian_kde(d["hh"], bw_method="silverman")
        kde_b = gaussian_kde(d["lh"], bw_method="silverman")
        y_a   = kde_a(x)
        y_b   = kde_b(x)
        scale = max(y_a.max(), y_b.max())

        ax.fill_between(x, y_a / scale, color=GROUP_A_FACE,  alpha=0.60, zorder=2)
        ax.plot(        x, y_a / scale, color=GROUP_A_EDGE,  lw=0.7,     zorder=3)
        ax.fill_between(x, y_b / scale, color=GROUP_B_COLOR, alpha=0.30, zorder=4)
        ax.plot(        x, y_b / scale, color=GROUP_B_EDGE,  lw=0.7,     zorder=5)

        for mean, color in [(d["hh"].mean(), GROUP_A_EDGE), (d["lh"].mean(), GROUP_B_EDGE)]:
            ax.plot([mean, mean], [0, 0.18], color=color, lw=1.0,
                    solid_capstyle="round", zorder=8)

        p_str = "< .001" if d["p"] < 0.001 else f'{d["p"]:.3f}'
        ax.text(-0.01, 0.5, concept, transform=ax.transAxes,
                ha="right", va="center", fontsize=8, style="italic", color="#222222")
        ax.text(1.04, 0.5, p_str, transform=ax.transAxes,
                ha="left", va="center", fontsize=6.5, color="#555555")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([])
        ax.set_xticks([])

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["bottom"].set_linewidth(0.4)

    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{t:.2f}" for t in xticks],
                              fontsize=6.5, color="#555555")
    axes[-1].tick_params(colors="#555555", width=0.5, length=2.0)
    axes[-1].spines["bottom"].set_color("#888888")
    # Push the x-axis label further down so it doesn't collide with the legend below it
    axes[-1].set_xlabel("Cosine dissimilarity", fontsize=7,
                        labelpad=10, color="#333333")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=GROUP_A_FACE,  alpha=0.60,
                      ec=GROUP_A_EDGE, lw=0.7, label=group_a_label),
        plt.Rectangle((0, 0), 1, 1, fc=GROUP_B_COLOR, alpha=0.30,
                      ec=GROUP_B_EDGE, lw=0.7, label=group_b_label),
        Line2D([0], [0], color=GROUP_A_EDGE, lw=1.0, solid_capstyle="round",
               label=f"Mean ({group_a_label})"),
        Line2D([0], [0], color=GROUP_B_EDGE, lw=1.0, solid_capstyle="round",
               label=f"Mean ({group_b_label})"),
    ]
    # Placed below the x-axis tick labels and axis title, well clear of the
    # plot area, so it never overlaps the bottom panel or its ticks.
    legend_y = -0.012 * (8.0 / fig_height)  # scales the gap below xlabel with figure height
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.46, legend_y), ncol=4, fontsize=6.5,
               frameon=True, framealpha=1.0, edgecolor="#cccccc",
               handlelength=1.2, borderpad=0.7)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


# Backward-compatible alias for the LLM-specific naming used in earlier versions.
def plot_llm_distributions(dist_data: dict, figsize: tuple = (8, None), save_path: str = None) -> plt.Figure:
    """Deprecated alias for plot_distribution_comparison() with LLM-specific labels.

    Kept for backward compatibility. New code should call
    plot_distribution_comparison() directly with explicit group labels.
    """
    return plot_distribution_comparison(
        dist_data,
        group_a_label="Human\u2013human",
        group_b_label="LLM\u2013human",
        figsize=figsize,
        save_path=save_path,
    )


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def plot_sharedness(
    df_sharedness: pd.DataFrame,
    concept_col: str = "concept",
    value_col: str = "S_b",
    figsize: tuple = None,
    save_path: str = None,
) -> plt.Figure:
    """Ranked bar chart of S_b across a concept set.

    Parameters
    ----------
    df_sharedness : pd.DataFrame
        Output of compute_sharedness(). Must contain concept_col and value_col.
    concept_col : str
    value_col : str
    figsize : tuple or None
        Auto-sized to number of concepts if None.
    save_path : str or None

    Returns
    -------
    plt.Figure
    """
    df = df_sharedness.sort_values(value_col, ascending=True).reset_index(drop=True)
    n  = len(df)

    if figsize is None:
        figsize = (5.0, n * 0.28 + 0.6)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(n), df[value_col], color="#3b6fd4", alpha=0.75,
            height=0.65, linewidth=0)

    ax.set_yticks(range(n))
    ax.set_yticklabels(df[concept_col].tolist(), fontsize=9,
                       color="#111111", style="italic")
    ax.set_xlabel(r"Sharedness ($S_b$)", fontsize=9, labelpad=3, color="#222222")
    ax.tick_params(axis="x", labelsize=8, width=0.6, length=2.5, colors="#555555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(left=False)

    ax.set_xlim(
        max(0, df[value_col].min() - 0.02),
        min(1, df[value_col].max() + 0.02)
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

    return fig
