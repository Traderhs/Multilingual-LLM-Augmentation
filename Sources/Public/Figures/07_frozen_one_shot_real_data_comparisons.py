import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "Results"
    / "BinaryMatchedSizeExperiment"
    / "Downstream"
    / "AdaptivePolicy"
    / "OneShotTest"
    / "test_arm_summary.csv"
)

OUTPUT_PATH = (
    ROOT
    / "07_frozen_one_shot_real_data_comparisons.pdf"
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the paper figure.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output figure path.",
    )
    return parser.parse_args()


DATASETS = [
    {
        "cell_id": "en_binary_sst2",
        "title": "English\n(SST-2)",
    },
    {
        "cell_id": "bn_binary_cinexdrama",
        "title": "Bengali\n(CineXDrama)",
    },
]

ARMS = [
    {
        "arm": "base",
        "legend_label": "Base",
        "tick_label": "Base",
        "marker": "o",
    },
    {
        "arm": "frozen_adaptive",
        "legend_label": "Selected Hybrid",
        "tick_label": "Selected\nHybrid",
        "marker": "s",
    },
    {
        "arm": "matched_all_real",
        "legend_label": "Matched All-real",
        "tick_label": "Matched\nAll-real",
        "marker": "^",
    },
]

METRICS = [
    {
        "metric": "macro_f1_at_oof_threshold",
        "ylabel": "Test Macro-F1",
    },
    {
        "metric": "auroc",
        "ylabel": "Test AUROC",
    },
]


def select_figure_font() -> str:
    preferred_families = [
        "Source Sans 3",
        "IBM Plex Sans",
        "Noto Sans",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ]

    installed = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    for family in preferred_families:
        if family in installed:
            return family

    return "sans-serif"


FIGURE_FONT = select_figure_font()
SEMIBOLD = 600


def load_plot_data() -> pd.DataFrame:
    summary = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required_columns = {
        "cell_id",
        "arm",
        "metric",
        "mean",
        "ci_lower",
        "ci_upper",
    }

    missing_columns = sorted(
        required_columns.difference(
            summary.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    for column in [
        "mean",
        "ci_lower",
        "ci_upper",
    ]:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="raise",
        )

    selected_cells = {
        dataset["cell_id"]
        for dataset in DATASETS
    }

    selected_arms = {
        arm["arm"]
        for arm in ARMS
    }

    selected_metrics = {
        metric["metric"]
        for metric in METRICS
    }

    relevant = summary.loc[
        summary["cell_id"].isin(selected_cells)
        & summary["arm"].isin(selected_arms)
        & summary["metric"].isin(selected_metrics)
    ].copy()

    expected_rows = (
        len(DATASETS)
        * len(ARMS)
        * len(METRICS)
    )

    if len(relevant) != expected_rows:
        raise ValueError(
            "Unexpected number of rows: "
            f"{len(relevant)} found, "
            f"{expected_rows} expected."
        )

    duplicate_mask = relevant.duplicated(
        subset=[
            "cell_id",
            "arm",
            "metric",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = relevant.loc[
            duplicate_mask,
            [
                "cell_id",
                "arm",
                "metric",
            ],
        ]

        raise ValueError(
            "Duplicate rows found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    numeric_values = relevant[
        [
            "mean",
            "ci_lower",
            "ci_upper",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite values were found."
        )

    invalid_intervals = relevant.loc[
        (
            relevant["ci_lower"]
            > relevant["mean"]
        )
        | (
            relevant["mean"]
            > relevant["ci_upper"]
        )
    ]

    if not invalid_intervals.empty:
        raise ValueError(
            "Invalid confidence intervals found:\n"
            f"{invalid_intervals.to_string(index=False)}"
        )

    for dataset in DATASETS:
        for arm in ARMS:
            for metric in METRICS:
                row = relevant.loc[
                    relevant["cell_id"].eq(
                        dataset["cell_id"]
                    )
                    & relevant["arm"].eq(
                        arm["arm"]
                    )
                    & relevant["metric"].eq(
                        metric["metric"]
                    )
                ]

                if len(row) != 1:
                    raise ValueError(
                        "Expected one row for "
                        f"{dataset['cell_id']}, "
                        f"{arm['arm']}, "
                        f"{metric['metric']}, "
                        f"but found {len(row)}."
                    )

    return relevant


def set_panel_limits(
    axis: plt.Axes,
    panel_data: pd.DataFrame,
) -> None:
    lower = float(
        panel_data["ci_lower"].min()
    )

    upper = float(
        panel_data["ci_upper"].max()
    )

    span = upper - lower

    if span <= 0:
        padding = 0.001
    else:
        padding = span * 0.18

    axis.set_ylim(
        lower - padding,
        upper + padding,
    )


def format_tick(
    value: float,
    _position: int,
) -> str:
    return f"{value:.4f}"


def main() -> None:
    global INPUT_PATH, OUTPUT_PATH

    args = parse_args()
    INPUT_PATH = args.input.expanduser().resolve()
    OUTPUT_PATH = args.output.expanduser().resolve()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_data = load_plot_data()

    plt.rcParams.update(
        {
            "font.family": FIGURE_FONT,
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.titleweight": SEMIBOLD,
            "axes.labelsize": 16,
            "axes.labelweight": SEMIBOLD,
            "xtick.labelsize": 14,
            "ytick.labelsize": 13,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10.8, 7.4),
    )

    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.14,
        top=0.80,
        wspace=0.24,
        hspace=0.10,
    )

    panel_labels = [
        "a",
        "b",
    ]

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ].by_key()["color"]
    )

    x_positions = np.arange(
        len(ARMS),
        dtype=float,
    )

    for column_index, dataset in enumerate(
        DATASETS
    ):
        for row_index, metric in enumerate(
            METRICS
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            panel_data = plot_data.loc[
                plot_data["cell_id"].eq(
                    dataset["cell_id"]
                )
                & plot_data["metric"].eq(
                    metric["metric"]
                )
            ].copy()

            plotted_means = []

            for arm_index, arm in enumerate(
                ARMS
            ):
                row = panel_data.loc[
                    panel_data["arm"].eq(
                        arm["arm"]
                    )
                ]

                record = row.iloc[0]

                mean = float(
                    record["mean"]
                )
                lower = float(
                    record["ci_lower"]
                )
                upper = float(
                    record["ci_upper"]
                )

                plotted_means.append(mean)

                color = default_colors[
                    arm_index
                ]

                axis.errorbar(
                    x_positions[arm_index],
                    mean,
                    yerr=np.asarray(
                        [
                            [
                                mean - lower,
                            ],
                            [
                                upper - mean,
                            ],
                        ]
                    ),
                    fmt=arm["marker"],
                    linestyle="none",
                    markersize=7.2,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    color=color,
                    ecolor=color,
                    elinewidth=1.6,
                    capsize=4.0,
                    capthick=1.4,
                    zorder=3,
                )

            axis.plot(
                x_positions,
                plotted_means,
                color="0.45",
                linewidth=0.9,
                zorder=2,
            )

            axis.grid(
                axis="y",
                linewidth=0.45,
                alpha=0.35,
                zorder=0,
            )

            axis.set_xlim(
                -0.5,
                len(ARMS) - 0.5,
            )

            axis.set_xticks(
                x_positions
            )

            axis.tick_params(
                axis="both",
                which="major",
                length=3.5,
            )

            axis.tick_params(
                axis="x",
                which="major",
                top=True,
                labeltop=False,
                bottom=True,
                pad=6,
            )

            if row_index == 0:
                axis.tick_params(
                    axis="x",
                    labelbottom=False,
                )

                axis.set_title(
                    f"({panel_labels[column_index]}) "
                    f"{dataset['title']}",
                    pad=9,
                    fontweight=SEMIBOLD,
                )
            else:
                axis.set_xticklabels(
                    [
                        arm["tick_label"]
                        for arm in ARMS
                    ],
                    ha="center",
                )

            set_panel_limits(
                axis,
                panel_data,
            )

            axis.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=5
                )
            )

            axis.yaxis.set_major_formatter(
                FuncFormatter(
                    format_tick
                )
            )

            if column_index == 0:
                axis.set_ylabel(
                    metric["ylabel"],
                    fontweight=SEMIBOLD,
                    labelpad=8,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=arm["marker"],
            linestyle="none",
            markersize=7.2,
            markerfacecolor=default_colors[
                arm_index
            ],
            markeredgecolor=default_colors[
                arm_index
            ],
            color=default_colors[
                arm_index
            ],
            label=arm["legend_label"],
        )
        for arm_index, arm in enumerate(
            ARMS
        )
    ]

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.9,
        handletextpad=0.55,
    )

    figure.savefig(
        OUTPUT_PATH,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()