import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    ROOT
    / "Results"
    / "BinaryMatchedSizeExperiment"
    / "Downstream"
    / "AdaptivePolicy"
    / "OneShotTest"
)

INPUT_PATH = (
    DATA_DIR
    / "test_paired_comparison_summary.csv"
)
OUTPUT_PATH = (
    ROOT
    / "06_frozen_one_shot_test_vs_base.pdf"
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
    (
        "en_binary_sst2",
        "English",
    ),
    (
        "ko_binary_nsmc",
        "Korean",
    ),
    (
        "bn_binary_cinexdrama",
        "Bengali",
    ),
    (
        "ha_binary_hausa_movie_review",
        "Hausa",
    ),
    (
        "ml_binary_dravidian_codemix",
        "Malayalam",
    ),
]

POOLED_LABEL = "Equal-weight\npooled"
COMPARISON = "frozen_adaptive_minus_base"
POOLED_SCOPE = "all_five_cells_equal_weight"

METRICS = [
    {
        "metric": "macro_f1_at_oof_threshold",
        "title": "Macro-F1",
        "ylabel": "Mean ΔMacro-F1\n(vs Base)",
    },
    {
        "metric": "auroc",
        "title": "AUROC",
        "ylabel": "Mean ΔAUROC\n(vs Base)",
    },
]


def select_figure_font() -> str:
    """Select a publication-friendly sans-serif font."""
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
    comparisons = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required_columns = {
        "scope",
        "cell_id",
        "comparison",
        "metric",
        "mean",
        "ci_lower",
        "ci_upper",
    }

    missing_columns = sorted(
        required_columns.difference(
            comparisons.columns
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
        comparisons[column] = pd.to_numeric(
            comparisons[column],
            errors="raise",
        )

    relevant = comparisons.loc[
        comparisons["comparison"].eq(COMPARISON)
        & comparisons["metric"].isin(
            [
                specification["metric"]
                for specification in METRICS
            ]
        )
    ].copy()

    records: list[dict[str, object]] = []

    for cell_id, label in DATASETS:
        for specification in METRICS:
            row = relevant.loc[
                relevant["scope"].eq("cell")
                & relevant["cell_id"].eq(cell_id)
                & relevant["metric"].eq(
                    specification["metric"]
                )
            ]

            if len(row) != 1:
                raise ValueError(
                    "Expected one comparison row for "
                    f"{cell_id}, "
                    f"{specification['metric']}, "
                    f"but found {len(row)}."
                )

            record = row.iloc[0]

            records.append(
                {
                    "group_id": cell_id,
                    "label": label,
                    "metric": specification["metric"],
                    "mean": float(record["mean"]),
                    "ci_lower": float(
                        record["ci_lower"]
                    ),
                    "ci_upper": float(
                        record["ci_upper"]
                    ),
                    "is_pooled": False,
                }
            )

    for specification in METRICS:
        row = relevant.loc[
            relevant["scope"].eq(POOLED_SCOPE)
            & relevant["metric"].eq(
                specification["metric"]
            )
        ]

        if len(row) != 1:
            raise ValueError(
                "Expected one pooled comparison row for "
                f"{specification['metric']}, "
                f"but found {len(row)}."
            )

        record = row.iloc[0]

        records.append(
            {
                "group_id": "equal_weight_pooled",
                "label": POOLED_LABEL,
                "metric": specification["metric"],
                "mean": float(record["mean"]),
                "ci_lower": float(
                    record["ci_lower"]
                ),
                "ci_upper": float(
                    record["ci_upper"]
                ),
                "is_pooled": True,
            }
        )

    plot_data = pd.DataFrame.from_records(
        records
    )

    numeric_values = plot_data[
        [
            "mean",
            "ci_lower",
            "ci_upper",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite values were found in "
            "the plotted results."
        )

    invalid_intervals = plot_data.loc[
        (
            plot_data["ci_lower"]
            > plot_data["mean"]
        )
        | (
            plot_data["mean"]
            > plot_data["ci_upper"]
        )
    ]

    if not invalid_intervals.empty:
        raise ValueError(
            "Invalid confidence intervals were found:\n"
            f"{invalid_intervals.to_string(index=False)}"
        )

    return plot_data


def set_panel_limits(
    axis: plt.Axes,
    panel_data: pd.DataFrame,
) -> None:
    lower = min(
        0.0,
        float(panel_data["ci_lower"].min()),
    )

    upper = max(
        0.0,
        float(panel_data["ci_upper"].max()),
    )

    span = upper - lower

    if span <= 0:
        padding = 0.0002
    else:
        padding = span * 0.12

    axis.set_ylim(
        lower - padding,
        upper + padding,
    )


def tick_formatter(
    span: float,
) -> FuncFormatter:
    decimals = 4 if span < 0.02 else 3

    def format_value(
        value: float,
        _position: int,
    ) -> str:
        threshold = 0.5 * (
            10 ** (-decimals)
        )

        if abs(value) < threshold:
            value = 0.0

        return f"{value:.{decimals}f}"

    return FuncFormatter(
        format_value
    )


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
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.5, 5.4),
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.22,
        top=0.84,
        wspace=0.24,
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

    x_labels = [
        label
        for _, label in DATASETS
    ] + [
        POOLED_LABEL
    ]

    x_positions = np.array(
        [0.0, 1.2, 2.4, 3.6, 4.8, 6.4],
        dtype=float,
    )

    for panel_index, specification in enumerate(
        METRICS
    ):
        axis = axes[panel_index]

        panel_data = plot_data.loc[
            plot_data["metric"].eq(
                specification["metric"]
            )
        ].copy()

        ordered_rows: list[pd.Series] = []

        for cell_id, _ in DATASETS:
            row = panel_data.loc[
                panel_data["group_id"].eq(
                    cell_id
                )
            ]

            if len(row) != 1:
                raise ValueError(
                    "Expected one plotted row for "
                    f"{cell_id}, but found {len(row)}."
                )

            ordered_rows.append(
                row.iloc[0]
            )

        pooled_row = panel_data.loc[
            panel_data["is_pooled"]
        ]

        if len(pooled_row) != 1:
            raise ValueError(
                "Expected exactly one pooled result."
            )

        ordered_rows.append(
            pooled_row.iloc[0]
        )

        ordered = pd.DataFrame(
            ordered_rows
        ).reset_index(drop=True)

        for point_index, record in ordered.iterrows():
            mean = float(
                record["mean"]
            )
            lower = float(
                record["ci_lower"]
            )
            upper = float(
                record["ci_upper"]
            )
            pooled = bool(
                record["is_pooled"]
            )

            if pooled:
                marker = "s"
                color = default_colors[1]
                marker_size = 7.0
            else:
                marker = "o"
                color = default_colors[0]
                marker_size = 6.5

            axis.errorbar(
                x_positions[point_index],
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
                fmt=marker,
                markersize=marker_size,
                markerfacecolor=color,
                markeredgecolor=color,
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=4.0,
                capthick=1.4,
                zorder=3,
            )

        axis.axhline(
            0.0,
            color="black",
            linewidth=0.65,
            zorder=1,
        )

        axis.axvline(
            5.6,
            color="black",
            linewidth=0.65,
            zorder=1,
        )

        axis.grid(
            axis="y",
            linewidth=0.45,
            alpha=0.35,
            zorder=0,
        )

        axis.set_xlim(
            -0.6,
            7.0,
        )

        axis.set_xticks(
            x_positions
        )

        axis.set_xticklabels(
            x_labels,
            rotation=0,
            ha="center",
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
            labelbottom=True,
            length=3.5,
            pad=6,
        )

        set_panel_limits(
            axis,
            ordered,
        )

        y_lower, y_upper = axis.get_ylim()

        axis.yaxis.set_major_locator(
            MaxNLocator(
                nbins=5
            )
        )

        axis.yaxis.set_major_formatter(
            tick_formatter(
                y_upper - y_lower
            )
        )

        axis.set_title(
            f"({panel_labels[panel_index]}) "
            f"{specification['title']}",
            pad=8,
            fontweight=SEMIBOLD,
        )

        axis.set_ylabel(
            specification["ylabel"],
            fontweight=SEMIBOLD,
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