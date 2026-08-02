import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    ROOT
    / "Results"
    / "BinaryMatchedSizeExperiment"
    / "Downstream"
    / "SyntheticWeightGrid"
)

INPUT_PATH = DATA_DIR / "paired_weight_results.csv"
OUTPUT_PATH = ROOT / "04_synthetic_weighting_effects.pdf"



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

SYNTHETIC_WEIGHTS = [
    0.0000,
    0.0625,
    0.1250,
    0.2500,
    0.5000,
    0.7500,
    1.0000,
]

DISPLAYED_RATIO_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
]

DISPLAYED_RATIO_LABELS = [
    "0.5",
    "1",
    "1.5",
    "2",
    "3",
    "4",
]

WEIGHT_LABELS = [
    "0",
    "0.0625",
    "0.125",
    "0.25",
    "0.5",
    "0.75",
    "1",
]

METRICS = [
    {
        "column": "delta_vs_base_macro_f1",
        "row_label": "ΔMacro-F1 (vs Base)",
    },
    {
        "column": "delta_vs_base_auroc",
        "row_label": "ΔAUROC (vs Base)",
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


def colorbar_formatter(limit: float) -> FuncFormatter:
    if limit < 0.01:
        decimals = 4
    elif limit < 0.1:
        decimals = 3
    else:
        decimals = 2

    return FuncFormatter(
        lambda value, _position: f"{value:.{decimals}f}"
    )


def compute_cell_edges(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    edges = np.empty(values_array.size + 1, dtype=float)

    edges[1:-1] = (
        values_array[:-1] + values_array[1:]
    ) / 2.0

    edges[0] = (
        values_array[0]
        - (values_array[1] - values_array[0]) / 2.0
    )

    edges[-1] = (
        values_array[-1]
        + (values_array[-1] - values_array[-2]) / 2.0
    )

    return edges


def validate_and_prepare(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    paired = paired.copy()

    required_columns = {
        "cell_id",
        "repeat_seed",
        "ratio",
        "synthetic_weight",
        "weighted_status",
        "delta_vs_base_macro_f1",
        "delta_vs_base_auroc",
    }

    missing_columns = sorted(
        required_columns.difference(paired.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    numeric_columns = [
        "ratio",
        "synthetic_weight",
        "delta_vs_base_macro_f1",
        "delta_vs_base_auroc",
    ]

    for column in numeric_columns:
        paired[column] = pd.to_numeric(
            paired[column],
            errors="raise",
        )

    incomplete = paired.loc[
        ~paired["weighted_status"].eq("completed")
    ]

    if not incomplete.empty:
        raise ValueError(
            "Incomplete weighted conditions were found:\n"
            f"{incomplete.head().to_string(index=False)}"
        )

    paired["ratio_key"] = paired["ratio"].round(4)
    paired["weight_key"] = (
        paired["synthetic_weight"].round(4)
    )

    expected_rows = (
        len(DATASETS)
        * len(RATIOS)
        * len(SYNTHETIC_WEIGHTS)
        * 50
    )

    if len(paired) != expected_rows:
        raise ValueError(
            "Unexpected number of weight-grid rows: "
            f"{len(paired)}; expected {expected_rows}."
        )

    duplicate_mask = paired.duplicated(
        subset=[
            "cell_id",
            "repeat_seed",
            "ratio_key",
            "weight_key",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = paired.loc[
            duplicate_mask,
            [
                "cell_id",
                "repeat_seed",
                "ratio",
                "synthetic_weight",
            ],
        ]

        raise ValueError(
            "Duplicate weight-grid rows found:\n"
            f"{duplicates.to_string(index=False)}"
        )

    expected_cells = {
        cell_id
        for cell_id, _ in DATASETS
    }

    actual_cells = set(paired["cell_id"].unique())
    missing_cells = sorted(
        expected_cells.difference(actual_cells)
    )

    if missing_cells:
        raise ValueError(
            "Missing datasets: "
            + ", ".join(missing_cells)
        )

    for cell_id, _ in DATASETS:
        for ratio in RATIOS:
            for weight in SYNTHETIC_WEIGHTS:
                condition = paired.loc[
                    paired["cell_id"].eq(cell_id)
                    & np.isclose(
                        paired["ratio_key"],
                        ratio,
                    )
                    & np.isclose(
                        paired["weight_key"],
                        weight,
                    )
                ]

                if len(condition) != 50:
                    raise ValueError(
                        "Expected 50 repetitions for "
                        f"{cell_id}, ratio={ratio}, "
                        f"weight={weight}; "
                        f"found {len(condition)}."
                    )

    return paired


def build_mean_matrix(
    paired: pd.DataFrame,
    cell_id: str,
    metric_column: str,
) -> np.ndarray:
    matrix = np.full(
        (
            len(SYNTHETIC_WEIGHTS),
            len(RATIOS),
        ),
        np.nan,
        dtype=float,
    )

    dataset = paired.loc[
        paired["cell_id"].eq(cell_id)
    ]

    for row_index, weight in enumerate(
        SYNTHETIC_WEIGHTS
    ):
        for column_index, ratio in enumerate(RATIOS):
            condition = dataset.loc[
                np.isclose(
                    dataset["weight_key"],
                    weight,
                )
                & np.isclose(
                    dataset["ratio_key"],
                    ratio,
                )
            ]

            values = condition[
                metric_column
            ].to_numpy(dtype=float)

            if values.size != 50:
                raise ValueError(
                    "Unexpected repetition count for "
                    f"{cell_id}, ratio={ratio}, "
                    f"weight={weight}, "
                    f"metric={metric_column}: "
                    f"{values.size}"
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    "Non-finite metric values found for "
                    f"{cell_id}, ratio={ratio}, "
                    f"weight={weight}, "
                    f"metric={metric_column}."
                )

            matrix[
                row_index,
                column_index,
            ] = float(values.mean())

    return matrix


def style_heatmap_axis(axis: plt.Axes) -> None:
    axis.grid(False)

    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
        spine.set_linestyle("-")

    axis.tick_params(
        axis="both",
        which="major",
        color="black",
        labelcolor="black",
        width=0.8,
        length=3.5,
        direction="out",
    )


def main() -> None:
    global INPUT_PATH, OUTPUT_PATH

    args = parse_args()
    INPUT_PATH = args.input.expanduser().resolve()
    OUTPUT_PATH = args.output.expanduser().resolve()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    paired = validate_and_prepare(paired)

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
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(
        figsize=(14.6, 8.8),
        layout="constrained",
    )

    grid = figure.add_gridspec(
        nrows=3,
        ncols=5,
        height_ratios=[
            1.0,
            1.0,
            0.07,
        ],
        hspace=0.10,
        wspace=0,
    )

    cmap = plt.get_cmap("RdBu_r").copy()
    panel_labels = ["a", "b", "c", "d", "e"]

    x_edges = compute_cell_edges(RATIOS)

    y_edges = np.arange(
        len(SYNTHETIC_WEIGHTS) + 1,
        dtype=float,
    )

    y_centers = (
        y_edges[:-1] + y_edges[1:]
    ) / 2.0

    first_column_axes = {}
    metric_texts = []

    for column_index, (
        cell_id,
        dataset_title,
    ) in enumerate(DATASETS):
        matrices = [
            build_mean_matrix(
                paired,
                cell_id,
                specification["column"],
            )
            for specification in METRICS
        ]

        finite_values = np.concatenate(
            [
                matrix[np.isfinite(matrix)]
                for matrix in matrices
            ]
        )

        if finite_values.size == 0:
            raise ValueError(
                "No finite values found for "
                f"{cell_id}."
            )

        color_limit = max(
            float(np.max(np.abs(finite_values))),
            1e-6,
        )

        norm = TwoSlopeNorm(
            vmin=-color_limit,
            vcenter=0.0,
            vmax=color_limit,
        )

        image = None

        for metric_index, matrix in enumerate(
            matrices
        ):
            axis = figure.add_subplot(
                grid[
                    metric_index,
                    column_index,
                ]
            )

            image = axis.pcolormesh(
                x_edges,
                y_edges,
                matrix,
                cmap=cmap,
                norm=norm,
                shading="flat",
                edgecolors="white",
                linewidth=0.65,
                antialiased=False,
                snap=True,
            )

            axis.set_xlim(
                x_edges[0],
                x_edges[-1],
            )

            axis.set_ylim(
                y_edges[0],
                y_edges[-1],
            )

            axis.set_yticks(y_centers)
            axis.set_xticks(RATIOS)

            style_heatmap_axis(axis)

            if metric_index == 0:
                axis.set_xticklabels([])

                axis.set_title(
                    f"({panel_labels[column_index]}) "
                    f"{dataset_title}",
                    pad=8,
                    fontweight=SEMIBOLD,
                )

                axis.tick_params(
                    axis="x",
                    which="major",
                    bottom=True,
                    top=False,
                    labelbottom=False,
                )
            else:
                axis.set_xticklabels(
                    [
                        DISPLAYED_RATIO_LABELS[
                            DISPLAYED_RATIO_VALUES.index(ratio)
                        ]
                        if ratio in DISPLAYED_RATIO_VALUES
                        else ""
                        for ratio in RATIOS
                    ],
                    rotation=0,
                    ha="center",
                )

                axis.set_xlabel(
                    "Augmentation ratio",
                    fontweight=SEMIBOLD,
                )

                axis.tick_params(
                    axis="x",
                    which="major",
                    bottom=True,
                    top=False,
                    labelbottom=True,
                )

            if column_index == 0:
                axis.set_yticklabels(
                    WEIGHT_LABELS,
                    rotation=0,
                    ha="right",
                )

                axis.set_ylabel(
                    "Synthetic weight",
                    fontweight=SEMIBOLD,
                    labelpad=14,
                )

                first_column_axes[metric_index] = axis
            else:
                axis.tick_params(
                    axis="y",
                    which="major",
                    labelleft=False,
                    left=True,
                )

        colorbar_axis = figure.add_subplot(
            grid[
                2,
                column_index,
            ]
        )

        colorbar = figure.colorbar(
            image,
            cax=colorbar_axis,
            orientation="horizontal",
        )

        colorbar.set_ticks(
            [
                -color_limit,
                0.0,
                color_limit,
            ]
        )

        colorbar.ax.xaxis.set_major_formatter(
            colorbar_formatter(color_limit)
        )

        colorbar.ax.tick_params(
            axis="x",
            labelsize=14,
            length=2.5,
            pad=2,
            color="black",
            labelcolor="black",
            width=0.8,
        )

        colorbar.outline.set_edgecolor("black")
        colorbar.outline.set_linewidth(0.8)
        colorbar.outline.set_linestyle("-")
        colorbar.ax.grid(False)

    # Finalize layout, then place metric labels relative to the actual
    # Synthetic weight label bounding boxes. This keeps the spacing stable.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    metric_label_gap = 0.02

    for metric_index, specification in enumerate(
        METRICS
    ):
        axis = first_column_axes[metric_index]
        axis_bbox = axis.get_position()

        ylabel_bbox_display = axis.yaxis.label.get_window_extent(
            renderer=renderer
        )

        ylabel_bbox_figure = ylabel_bbox_display.transformed(
            figure.transFigure.inverted()
        )

        metric_text = figure.text(
            ylabel_bbox_figure.x0 - metric_label_gap,
            (axis_bbox.y0 + axis_bbox.y1) / 2.0,
            specification["row_label"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=16,
            fontweight=SEMIBOLD,
            color="black",
        )

        metric_texts.append(metric_text)

    figure.savefig(
        OUTPUT_PATH,
        format="pdf",
        bbox_inches="tight",
        bbox_extra_artists=metric_texts,
    )

    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()