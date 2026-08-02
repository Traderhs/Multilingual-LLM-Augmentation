import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    ROOT
    / "Results"
    / "BinaryMatchedSizeExperiment"
    / "Downstream"
    / "DevelopmentGrid"
)

INPUT_PATH = DATA_DIR / "paired_results.csv"
OUTPUT_PATH = ROOT / "02_similarity_filtering_effects.pdf"



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

LOWER_BOUNDS = [0.1, 0.2, 0.3, 0.4]
UPPER_BOUNDS = [0.6, 0.7, 0.8, 0.9]

# Narrow intervals at the top, wider intervals at the bottom.
SIMILARITY_INTERVALS = sorted(
    [
        (lower, upper)
        for lower in LOWER_BOUNDS
        for upper in UPPER_BOUNDS
    ],
    key=lambda interval: (
        round(interval[1] - interval[0], 10),
        interval[0],
        interval[1],
    ),
)

INTERVAL_WIDTHS = [
    round(upper - lower, 2)
    for lower, upper in SIMILARITY_INTERVALS
]

INTERVAL_LABELS = [
    f"[{lower:.1f}, {upper:.1f}]"
    for lower, upper in SIMILARITY_INTERVALS
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

    installed = {font.name for font in font_manager.fontManager.ttflist}

    for family in preferred_families:
        if family in installed:
            return family

    return "sans-serif"


FIGURE_FONT = select_figure_font()
SEMIBOLD = 600


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
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


def build_plot_data(paired: pd.DataFrame) -> pd.DataFrame:
    paired = paired.copy()

    paired["feasible"] = parse_bool(paired["feasible"])
    paired["ratio"] = pd.to_numeric(
        paired["ratio"],
        errors="raise",
    )
    paired["similarity_lower"] = pd.to_numeric(
        paired["similarity_lower"],
        errors="coerce",
    )
    paired["similarity_upper"] = pd.to_numeric(
        paired["similarity_upper"],
        errors="coerce",
    )
    paired["hybrid_macro_f1"] = pd.to_numeric(
        paired["hybrid_macro_f1"],
        errors="coerce",
    )

    off = paired.loc[
        paired["similarity_condition_id"].eq("OFF"),
        [
            "cell_id",
            "repeat_seed",
            "ratio",
            "hybrid_macro_f1",
        ],
    ].rename(
        columns={
            "hybrid_macro_f1": "off_hybrid_macro_f1",
        }
    )

    filtered = paired.loc[
        ~paired["similarity_condition_id"].eq("OFF")
    ].copy()

    filtered = filtered.merge(
        off,
        on=[
            "cell_id",
            "repeat_seed",
            "ratio",
        ],
        how="left",
        validate="many_to_one",
    )

    if filtered["off_hybrid_macro_f1"].isna().any():
        missing = filtered.loc[
            filtered["off_hybrid_macro_f1"].isna(),
            [
                "cell_id",
                "repeat_seed",
                "ratio",
            ],
        ].drop_duplicates()

        raise ValueError(
            "Missing unfiltered comparison rows:\n"
            f"{missing.to_string(index=False)}"
        )

    filtered["delta_macro_f1_vs_off"] = (
        filtered["hybrid_macro_f1"]
        - filtered["off_hybrid_macro_f1"]
    )

    records: list[dict[str, object]] = []

    for cell_id, _ in DATASETS:
        cell = filtered.loc[
            filtered["cell_id"].eq(cell_id)
        ]

        if cell.empty:
            raise ValueError(
                f"No rows found for dataset: {cell_id}"
            )

        for lower, upper in SIMILARITY_INTERVALS:
            interval = cell.loc[
                np.isclose(
                    cell["similarity_lower"],
                    lower,
                )
                & np.isclose(
                    cell["similarity_upper"],
                    upper,
                )
            ]

            for ratio in RATIOS:
                condition = interval.loc[
                    np.isclose(interval["ratio"], ratio)
                ]

                if condition.empty:
                    mean_delta = np.nan
                    infeasible_any = True
                else:
                    infeasible_any = bool(
                        (~condition["feasible"]).any()
                    )

                    feasible_values = condition.loc[
                        condition["feasible"]
                        & condition[
                            "delta_macro_f1_vs_off"
                        ].notna(),
                        "delta_macro_f1_vs_off",
                    ]

                    mean_delta = (
                        float(feasible_values.mean())
                        if not feasible_values.empty
                        else np.nan
                    )

                records.append(
                    {
                        "cell_id": cell_id,
                        "ratio": ratio,
                        "lower": lower,
                        "upper": upper,
                        "mean_delta": mean_delta,
                        "infeasible_any": infeasible_any,
                    }
                )

    return pd.DataFrame.from_records(records)


def matrix_for_dataset(
    plot_data: pd.DataFrame,
    cell_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(
        (
            len(SIMILARITY_INTERVALS),
            len(RATIOS),
        ),
        np.nan,
        dtype=float,
    )

    infeasible = np.zeros_like(
        values,
        dtype=bool,
    )

    subset = plot_data.loc[
        plot_data["cell_id"].eq(cell_id)
    ]

    for row_index, (lower, upper) in enumerate(
        SIMILARITY_INTERVALS
    ):
        for column_index, ratio in enumerate(RATIOS):
            row = subset.loc[
                np.isclose(subset["lower"], lower)
                & np.isclose(subset["upper"], upper)
                & np.isclose(subset["ratio"], ratio)
            ]

            if len(row) != 1:
                raise ValueError(
                    "Expected exactly one aggregated row for "
                    f"{cell_id}, [{lower}, {upper}], "
                    f"ratio={ratio}; found {len(row)}."
                )

            record = row.iloc[0]

            values[row_index, column_index] = (
                record["mean_delta"]
            )

            infeasible[row_index, column_index] = bool(
                record["infeasible_any"]
            )

    return values, infeasible


def width_group_boundaries() -> list[float]:
    boundaries: list[float] = []

    for index in range(1, len(INTERVAL_WIDTHS)):
        if INTERVAL_WIDTHS[index] != INTERVAL_WIDTHS[index - 1]:
            boundaries.append(index - 0.5)

    return boundaries


def add_width_group_labels(axis: plt.Axes) -> None:
    widths = np.asarray(INTERVAL_WIDTHS)
    unique_widths = list(dict.fromkeys(INTERVAL_WIDTHS))

    # Keep the width column close to the similarity-interval labels.
    label_x = -0.3

    for width in unique_widths:
        row_indices = np.flatnonzero(
            np.isclose(widths, width)
        )
        center = float(row_indices.mean())

        axis.text(
            label_x,
            center,
            f"{width:.1f}",
            transform=axis.get_yaxis_transform(),
            rotation=0,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="normal",
            color="black",
            clip_on=False,
        )

    axis.text(
        label_x,
        1.025,
        "Band width",
        transform=axis.transAxes,
        rotation=0,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight=SEMIBOLD,
        color="black",
        clip_on=False,
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

    finite_values = plot_data.loc[
        np.isfinite(plot_data["mean_delta"]),
        "mean_delta",
    ].to_numpy()

    if finite_values.size == 0:
        raise ValueError(
            "No finite Macro-F1 differences were found."
        )

    color_limit = float(
        np.max(np.abs(finite_values))
    )
    color_limit = max(color_limit, 0.001)

    norm = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.82")

    plt.rcParams.update(
        {
            "font.family": FIGURE_FONT,
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.titleweight": SEMIBOLD,
            "axes.labelsize": 16,
            "axes.labelweight": SEMIBOLD,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(13.5, 14.5),
        constrained_layout=True,
    )

    axes = axes.ravel()
    panel_labels = ["a", "b", "c", "d", "e"]
    boundaries = width_group_boundaries()
    image = None

    # Bottom-most occupied panel of each column.
    bottom_label_indices = {3, 4}

    for index, (cell_id, title) in enumerate(DATASETS):
        axis = axes[index]

        values, infeasible = matrix_for_dataset(
            plot_data,
            cell_id,
        )

        masked_values = np.ma.masked_where(
            infeasible | ~np.isfinite(values),
            values,
        )

        image = axis.imshow(
            masked_values,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        axis.set_title(
            f"({panel_labels[index]}) {title}",
            pad=8,
            fontweight=SEMIBOLD,
        )

        axis.set_xticks(np.arange(len(RATIOS)))

        if index in bottom_label_indices:
            axis.set_xticklabels(
                [f"{ratio:g}" for ratio in RATIOS],
                rotation=0,
                ha="center",
            )
            axis.set_xlabel(
                "Augmentation ratio",
                fontweight=SEMIBOLD,
            )
        else:
            # Keep major ticks, but hide their labels.
            axis.tick_params(
                axis="x",
                which="major",
                labelbottom=False,
            )

            # Reserve exactly the same vertical layout space as the visible
            # x-axis label below panel (d). This reproduces the original
            # (c)/(d)–(e) separation for (a)/(b)–(c)/(d) without a spacer row.
            if index in {0, 1}:
                axis.set_xlabel(
                    "Augmentation ratio",
                    fontweight=SEMIBOLD,
                    color="none",
                )

        axis.tick_params(
            axis="x",
            which="major",
            bottom=True,
            length=3.5,
        )

        axis.set_yticks(
            np.arange(len(SIMILARITY_INTERVALS))
        )

        if index % 2 == 0:
            axis.set_yticklabels(
                INTERVAL_LABELS,
                rotation=0,
                ha="right",
            )
            add_width_group_labels(axis)
        else:
            axis.set_yticklabels([])

        axis.set_xticks(
            np.arange(-0.5, len(RATIOS), 1),
            minor=True,
        )
        axis.set_yticks(
            np.arange(
                -0.5,
                len(SIMILARITY_INTERVALS),
                1,
            ),
            minor=True,
        )

        axis.grid(
            which="minor",
            linewidth=0.45,
            color="white",
        )

        axis.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        axis.tick_params(
            axis="y",
            which="major",
            left=True,
            length=3.5,
            pad=4,
        )

        # Thin black separators between band-width groups.
        for boundary in boundaries:
            axis.axhline(
                boundary,
                linewidth=0.65,
                color="black",
                alpha=1.0,
                zorder=3,
            )

    if image is None:
        raise RuntimeError(
            "No heatmap was generated."
        )

    # Colorbar moved into the 6th (previously empty) panel slot.
    axes[5].axis("off")
    cbar_inset = axes[5].inset_axes([0.375, 0.15, 0.25, 0.7])

    colorbar = figure.colorbar(
        image,
        cax=cbar_inset,
    )

    colorbar.set_label("")
    colorbar.ax.set_xlabel(
        "Mean ΔMacro-F1\n"
        "(filtered Hybrid − unfiltered Hybrid)",
        rotation=0,
        labelpad=8,
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