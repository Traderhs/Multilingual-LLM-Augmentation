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
)

PRESELECTION_PATH = DATA_DIR / "adaptive_preselection.csv"
SUMMARY_PATH = (
    DATA_DIR
    / "DecisionBoundaryDiagnostic"
    / "decision_boundary_summary.csv"
)
OUTPUT_PATH = (
    ROOT
    / "05_policy_selection_and_decision_boundary_calibration.pdf"
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the paper figure.")
    parser.add_argument(
        "--preselection",
        type=Path,
        default=PRESELECTION_PATH,
        help="Adaptive preselection CSV file.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Decision-boundary summary CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output figure path.",
    )
    return parser.parse_args()


DATASETS = [
    ("en_binary_sst2", "English\n(SST-2)"),
    (
        "bn_binary_cinexdrama",
        "Bengali\n(CineXDrama)",
    ),
    (
        "ml_binary_dravidian_codemix",
        "Malayalam\n(DravidianCodeMix)",
    ),
]

EXPECTED_GUARD_RESULTS = {
    "en_binary_sst2": True,
    "bn_binary_cinexdrama": True,
    "ml_binary_dravidian_codemix": False,
}

THRESHOLD_CONDITIONS = [
    {
        "label": "Fixed\n0.5 threshold",
        "mean_column": "default_threshold_delta_mean",
        "lower_column": "default_threshold_delta_ci_lower",
        "upper_column": "default_threshold_delta_ci_upper",
        "marker": "o",
    },
    {
        "label": "Training-derived\nthresholds",
        "mean_column": "training_oof_threshold_delta_mean",
        "lower_column": "training_oof_threshold_delta_ci_lower",
        "upper_column": "training_oof_threshold_delta_ci_upper",
        "marker": "s",
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


def parse_bool(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    parsed = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )

    if parsed.isna().any():
        invalid_values = sorted(
            normalized.loc[parsed.isna()].unique()
        )

        raise ValueError(
            f"Unexpected boolean values: {invalid_values}"
        )

    return parsed.astype(bool)


def load_plot_data() -> pd.DataFrame:
    preselection = pd.read_csv(
        PRESELECTION_PATH
    )
    summary = pd.read_csv(
        SUMMARY_PATH
    )

    required_preselection_columns = {
        "cell_id",
        "selected_arm",
        "selected_ratio",
        "selected_synthetic_weight",
        "diagnostic_required",
    }

    required_summary_columns = {
        "cell_id",
        "ratio",
        "synthetic_weight",
        "default_threshold_delta_mean",
        "default_threshold_delta_ci_lower",
        "default_threshold_delta_ci_upper",
        "training_oof_threshold_delta_mean",
        "training_oof_threshold_delta_ci_lower",
        "training_oof_threshold_delta_ci_upper",
        "threshold_guard_passed",
    }

    missing_preselection = sorted(
        required_preselection_columns.difference(
            preselection.columns
        )
    )

    missing_summary = sorted(
        required_summary_columns.difference(
            summary.columns
        )
    )

    if missing_preselection:
        raise ValueError(
            "adaptive_preselection.csv is missing columns: "
            + ", ".join(missing_preselection)
        )

    if missing_summary:
        raise ValueError(
            "decision_boundary_summary.csv is missing columns: "
            + ", ".join(missing_summary)
        )

    preselection["diagnostic_required"] = parse_bool(
        preselection["diagnostic_required"]
    )

    summary["threshold_guard_passed"] = parse_bool(
        summary["threshold_guard_passed"]
    )

    for column in [
        "selected_ratio",
        "selected_synthetic_weight",
    ]:
        preselection[column] = pd.to_numeric(
            preselection[column],
            errors="raise",
        )

    numeric_summary_columns = [
        "ratio",
        "synthetic_weight",
        "default_threshold_delta_mean",
        "default_threshold_delta_ci_lower",
        "default_threshold_delta_ci_upper",
        "training_oof_threshold_delta_mean",
        "training_oof_threshold_delta_ci_lower",
        "training_oof_threshold_delta_ci_upper",
    ]

    for column in numeric_summary_columns:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="raise",
        )

    diagnostic_candidates = preselection.loc[
        preselection["diagnostic_required"]
        & preselection["selected_arm"].eq(
            "weighted_hybrid"
        )
    ].copy()

    expected_ids = {
        cell_id
        for cell_id, _ in DATASETS
    }

    actual_ids = set(
        diagnostic_candidates["cell_id"].unique()
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "Unexpected diagnostic candidate set: "
            f"expected={sorted(expected_ids)}, "
            f"observed={sorted(actual_ids)}"
        )

    merged = diagnostic_candidates.merge(
        summary,
        left_on=[
            "cell_id",
            "selected_ratio",
            "selected_synthetic_weight",
        ],
        right_on=[
            "cell_id",
            "ratio",
            "synthetic_weight",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged["ratio"].isna().any():
        missing = merged.loc[
            merged["ratio"].isna(),
            [
                "cell_id",
                "selected_ratio",
                "selected_synthetic_weight",
            ],
        ]

        raise ValueError(
            "Missing decision-boundary summaries:\n"
            f"{missing.to_string(index=False)}"
        )

    for cell_id, expected_result in (
        EXPECTED_GUARD_RESULTS.items()
    ):
        row = merged.loc[
            merged["cell_id"].eq(cell_id)
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one summary row for {cell_id}; "
                f"found {len(row)}."
            )

        observed_result = bool(
            row.iloc[0]["threshold_guard_passed"]
        )

        if observed_result != expected_result:
            raise ValueError(
                "Unexpected calibrated-threshold decision for "
                f"{cell_id}: expected={expected_result}, "
                f"observed={observed_result}."
            )

    plotted_columns = [
        specification[column_key]
        for specification in THRESHOLD_CONDITIONS
        for column_key in [
            "mean_column",
            "lower_column",
            "upper_column",
        ]
    ]

    plotted_values = merged[
        plotted_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(plotted_values).all():
        raise ValueError(
            "Non-finite values were found in the "
            "decision-boundary summaries."
        )

    return merged


def set_dataset_specific_limits(
    axis: plt.Axes,
    lower_values: np.ndarray,
    upper_values: np.ndarray,
) -> None:
    lower = min(
        0.0,
        float(np.min(lower_values)),
    )

    upper = max(
        0.0,
        float(np.max(upper_values)),
    )

    span = upper - lower

    if span <= 0:
        padding = 0.001
    else:
        padding = span * 0.10

    axis.set_ylim(
        lower - padding,
        upper + padding,
    )


def y_tick_formatter(
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

    return FuncFormatter(format_value)


def main() -> None:
    global PRESELECTION_PATH, SUMMARY_PATH, OUTPUT_PATH

    args = parse_args()
    PRESELECTION_PATH = args.preselection.expanduser().resolve()
    SUMMARY_PATH = args.summary.expanduser().resolve()
    OUTPUT_PATH = args.output.expanduser().resolve()
    for path in (PRESELECTION_PATH, SUMMARY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
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
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(13.5, 4.9),
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.22,
        top=0.82,
        wspace=0.28,
    )

    panel_labels = [
        "a",
        "b",
        "c",
    ]

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ].by_key()["color"]
    )

    for index, (
        cell_id,
        dataset_title,
    ) in enumerate(DATASETS):
        axis = axes[index]

        row = plot_data.loc[
            plot_data["cell_id"].eq(cell_id)
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one row for {cell_id}; "
                f"found {len(row)}."
            )

        record = row.iloc[0]

        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for condition_index, specification in enumerate(
            THRESHOLD_CONDITIONS
        ):
            mean = float(
                record[
                    specification["mean_column"]
                ]
            )
            lower = float(
                record[
                    specification["lower_column"]
                ]
            )
            upper = float(
                record[
                    specification["upper_column"]
                ]
            )

            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)

            axis.errorbar(
                condition_index,
                mean,
                yerr=np.asarray(
                    [
                        [mean - lower],
                        [upper - mean],
                    ]
                ),
                fmt=specification["marker"],
                markersize=7.0,
                markerfacecolor=default_colors[
                    condition_index
                ],
                markeredgecolor=default_colors[
                    condition_index
                ],
                color=default_colors[
                    condition_index
                ],
                ecolor=default_colors[
                    condition_index
                ],
                elinewidth=1.6,
                capsize=4.0,
                capthick=1.4,
                zorder=3,
            )

        axis.plot(
            [0, 1],
            means,
            color="0.45",
            linewidth=1.0,
            zorder=2,
        )

        axis.axhline(
            0.0,
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
            -0.45,
            1.45,
        )

        axis.set_xticks(
            [0, 1]
        )

        axis.set_xticklabels(
            [
                specification["label"]
                for specification
                in THRESHOLD_CONDITIONS
            ]
        )

        axis.tick_params(
            axis="both",
            which="major",
            length=3.5,
        )

        set_dataset_specific_limits(
            axis,
            np.asarray(lowers),
            np.asarray(uppers),
        )

        y_lower, y_upper = axis.get_ylim()

        axis.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        axis.yaxis.set_major_formatter(
            y_tick_formatter(
                y_upper - y_lower
            )
        )

        axis.set_title(
            f"({panel_labels[index]}) "
            f"{dataset_title}",
            pad=8,
            fontweight=SEMIBOLD,
        )

        if index == 0:
            axis.set_ylabel(
                "Mean ΔMacro-F1\n(vs Base)",
                fontweight=SEMIBOLD,
            )

    figure.savefig(
        OUTPUT_PATH,
        format="pdf",
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()