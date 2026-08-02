import argparse
from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    ROOT
    / "Results"
    / "BinaryMatchedSizeExperiment"
    / "Downstream"
    / "DevelopmentGrid"
)

INPUT_PATH = DATA_DIR / "paired_results.csv"
OUTPUT_PATH = ROOT / "03_performance_across_augmentation_ratios.pdf"



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
    ("en_binary_sst2", "English\n(SST-2)"),
    ("ko_binary_nsmc", "Korean\n(NSMC)"),
    ("bn_binary_cinexdrama", "Bengali\n(CineXDrama)"),
    ("ha_binary_hausa_movie_review", "Hausa\n(HausaMovieReview)"),
    ("ml_binary_dravidian_codemix", "Malayalam\n(DravidianCodeMix)"),
]

RATIOS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    3.50,
    4.00,
]

# Fewer labels are shown to prevent overlap, while all 11 ratios remain plotted.
DISPLAY_RATIOS = [
    0.50,
    1.00,
    1.50,
    2.00,
    3.00,
    4.00,
]

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BASE_SEED = 20260713

METRICS = [
    {
        "metric": "macro_f1",
        "hybrid_column": "delta_synth_macro_f1",
        "matched_column": "delta_real_macro_f1",
        "ylabel": "ΔMacro-F1\n(vs Base)",
    },
    {
        "metric": "auroc",
        "hybrid_column": "delta_synth_auroc",
        "matched_column": "delta_real_auroc",
        "ylabel": "ΔAUROC\n(vs Base)",
    },
]


def select_figure_font() -> str:
    """Select a publication-friendly sans-serif font with semibold support."""
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


def parse_bool(series: pd.Series) -> pd.Series:
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


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    offset = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return (
        BOOTSTRAP_BASE_SEED + offset
    ) % (2**32)


def paired_bootstrap_summary(
    values: np.ndarray,
    *seed_parts: object,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            "Bootstrap input must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "Bootstrap input is empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Bootstrap input contains non-finite values."
        )

    rng = np.random.default_rng(
        stable_seed(*seed_parts)
    )

    sampled_indices = rng.integers(
        low=0,
        high=values.size,
        size=(
            BOOTSTRAP_REPLICATES,
            values.size,
        ),
    )

    bootstrap_means = values[
        sampled_indices
    ].mean(axis=1)

    mean = float(values.mean())
    ci_lower, ci_upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return (
        mean,
        float(ci_lower),
        float(ci_upper),
    )


def build_plot_data(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    paired = paired.copy()

    paired["feasible"] = parse_bool(
        paired["feasible"]
    )

    numeric_columns = {
        "ratio",
        "delta_synth_macro_f1",
        "delta_real_macro_f1",
        "delta_synth_auroc",
        "delta_real_auroc",
    }

    for column in numeric_columns:
        paired[column] = pd.to_numeric(
            paired[column],
            errors="raise",
        )

    # The ratio analysis uses only the unfiltered condition.
    off = paired.loc[
        paired["similarity_condition_id"].eq("OFF")
        & paired["feasible"]
    ].copy()

    expected_rows = (
        len(DATASETS)
        * len(RATIOS)
        * 50
    )

    if len(off) != expected_rows:
        raise ValueError(
            "Unexpected number of similarity-OFF rows: "
            f"{len(off)}; expected {expected_rows}."
        )

    duplicate_mask = off.duplicated(
        subset=[
            "cell_id",
            "repeat_seed",
            "ratio",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = off.loc[
            duplicate_mask,
            [
                "cell_id",
                "repeat_seed",
                "ratio",
            ],
        ]

        raise ValueError(
            "Duplicate similarity-OFF rows found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    records: list[dict[str, object]] = []

    for cell_id, _ in DATASETS:
        for ratio in RATIOS:
            condition = off.loc[
                off["cell_id"].eq(cell_id)
                & np.isclose(
                    off["ratio"],
                    ratio,
                )
            ].sort_values("repeat_seed")

            if len(condition) != 50:
                raise ValueError(
                    "Expected 50 paired repetitions for "
                    f"{cell_id}, ratio={ratio}; "
                    f"found {len(condition)}."
                )

            for specification in METRICS:
                metric = specification["metric"]

                arm_columns = [
                    (
                        "Unweighted Hybrid",
                        specification["hybrid_column"],
                    ),
                    (
                        "Matched All-real",
                        specification["matched_column"],
                    ),
                ]

                for arm, value_column in arm_columns:
                    values = condition[
                        value_column
                    ].to_numpy(dtype=float)

                    (
                        mean,
                        ci_lower,
                        ci_upper,
                    ) = paired_bootstrap_summary(
                        values,
                        cell_id,
                        ratio,
                        metric,
                        arm,
                    )

                    records.append(
                        {
                            "cell_id": cell_id,
                            "ratio": ratio,
                            "metric": metric,
                            "arm": arm,
                            "mean": mean,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                        }
                    )

    return pd.DataFrame.from_records(records)


def set_dataset_specific_limits(
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
        padding = 0.001
    else:
        padding = span * 0.08

    axis.set_ylim(
        lower - padding,
        upper + padding,
    )


def main() -> None:
    global INPUT_PATH, OUTPUT_PATH

    args = parse_args()
    INPUT_PATH = args.input.expanduser().resolve()
    OUTPUT_PATH = args.output.expanduser().resolve()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(INPUT_PATH)
    plot_data = build_plot_data(paired)

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
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(13.5, 7.8),
        sharex="col",
        constrained_layout=True,
    )

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ].by_key()["color"]
    )

    arm_styles = {
        "Unweighted Hybrid": {
            "color": default_colors[0],
            "marker": "o",
        },
        "Matched All-real": {
            "color": default_colors[1],
            "marker": "s",
        },
    }

    panel_labels = [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    legend_handles = []

    for column_index, (
        cell_id,
        dataset_title,
    ) in enumerate(DATASETS):
        for row_index, specification in enumerate(
            METRICS
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            metric = specification["metric"]

            panel_data = plot_data.loc[
                plot_data["cell_id"].eq(cell_id)
                & plot_data["metric"].eq(metric)
            ].copy()

            for arm in [
                "Unweighted Hybrid",
                "Matched All-real",
            ]:
                arm_data = panel_data.loc[
                    panel_data["arm"].eq(arm)
                ].sort_values("ratio")

                style = arm_styles[arm]

                line, = axis.plot(
                    arm_data["ratio"],
                    arm_data["mean"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.8,
                    markersize=4.5,
                    label=arm,
                    zorder=3,
                )

                axis.fill_between(
                    arm_data["ratio"].to_numpy(),
                    arm_data["ci_lower"].to_numpy(),
                    arm_data["ci_upper"].to_numpy(),
                    color=style["color"],
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )

                if (
                    row_index == 0
                    and column_index == 0
                ):
                    legend_handles.append(line)

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
            )

            axis.set_xlim(
                0.45,
                4.05,
            )

            axis.set_xticks(RATIOS)
            axis.set_xticklabels(
                [
                    f"{ratio:g}"
                    if ratio in DISPLAY_RATIOS
                    else ""
                    for ratio in RATIOS
                ],
                rotation=0,
                ha="center",
            )

            axis.yaxis.set_major_locator(
                MaxNLocator(nbins=5)
            )
            axis.yaxis.set_major_formatter(
                FormatStrFormatter("%.3f")
            )

            axis.tick_params(
                axis="both",
                which="major",
                length=3.5,
            )

            set_dataset_specific_limits(
                axis,
                panel_data,
            )

            if row_index == 0:
                axis.set_title(
                    f"({panel_labels[column_index]}) "
                    f"{dataset_title}",
                    pad=8,
                    fontweight=SEMIBOLD,
                )

                axis.tick_params(
                    axis="x",
                    which="major",
                    labelbottom=False,
                )
            else:
                axis.set_xlabel(
                    "Augmentation ratio",
                    fontweight=SEMIBOLD,
                )

            if column_index == 0:
                axis.set_ylabel(
                    specification["ylabel"],
                    fontweight=SEMIBOLD,
                )

    figure.legend(
        legend_handles,
        [
            "Unweighted Hybrid",
            "Matched All-real",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=2,
        frameon=False,
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