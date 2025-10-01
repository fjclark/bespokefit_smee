"""Functionality for analysing the results of a BespokeFitSMEE run."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .outputs import OutputStage, OutputType, StageKind
from .settings import WorkflowSettings

PLT_STYLE = "ggplot"

POTENTIAL_KEYS = Literal[
    "Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW", "Electrostatics"
]


def read_errors(
    paths_by_iter: dict[int, Path],
) -> dict[Literal["energy", "force"], dict[int, npt.NDArray[np.float64]]]:
    """Read the energy and force errors from the scatter files."""

    energy_errors = {
        # i: np.loadtxt(f)[:, 1] - np.loadtxt(f)[:, 0]
        # for i, f in enumerate(error_datafiles)
        i: np.loadtxt(f)[:, 0]
        for i, f in paths_by_iter.items()
    }
    # Drop nan values from the energy errors
    energy_errors = {
        i: e[~np.isnan(e)] for i, e in energy_errors.items() if len(e[~np.isnan(e)]) > 0
    }
    force_errors = {
        # i: np.loadtxt(f)[:, 3] - np.loadtxt(f)[:, 2]
        # for i, f in enumerate(error_datafiles)
        i: np.loadtxt(f)[:, 1]
        for i, f in paths_by_iter.items()
    }

    return {"energy": energy_errors, "force": force_errors}


def read_losses(paths_by_iter: dict[int, Path]) -> pd.DataFrame:
    idxs, losses_test, losses_train, iteration = [], [], [], []

    for i, loss_datafile in paths_by_iter.items():
        df = pd.read_csv(
            loss_datafile,
            sep=r"\s+",
            header=None,
            names=["idx", "loss_train", "loss_test"],
        )
        idxs.append(df["idx"].tolist())
        losses_test.append(df["loss_train"].tolist())
        losses_train.append(df["loss_test"].tolist())
        iteration.append(np.ones_like(df["idx"].tolist()) * i)

    return pd.DataFrame(
        data={
            "idx": np.concatenate(idxs),
            "loss_train": np.concatenate(losses_test),
            "loss_test": np.concatenate(losses_train),
            "iteration": np.concatenate(iteration),
        }
    )


def plot_loss(fig: Figure, ax: Axes, losses: pd.DataFrame) -> None:
    # Colour by iteration - full line for train, dotted for test
    for i in losses["iteration"].unique():
        ax.plot(
            losses[losses["iteration"] == i].index,
            losses[losses["iteration"] == i]["loss_train"],
            label=f"train-{i}",
            color=f"C{i}",
        )
        ax.plot(
            losses[losses["iteration"] == i].index,
            losses[losses["iteration"] == i]["loss_test"],
            label=f"test-{i}",
            color=f"C{i}",
            linestyle="--",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_distributions_of_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    # Colour by iteration
    # Use continuous colourmap for the iterations
    iterations = errors.keys()
    colours = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(iterations) + 1))

    for i in iterations:
        ax.hist(
            errors[i],
            label=f"Iteration {i}",
            alpha=0.8,
            color=colours[i],
            edgecolor="black",
        )

    ax.set_xlabel(
        f"Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )
    ax.set_ylabel("Frequency")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel(
        f"Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )
    ax.set_ylabel("Frequency")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_mean_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    mean_errors = {i: np.mean(errors[i]) for i in errors.keys()}

    ax.plot(
        list(mean_errors.keys()),
        list(mean_errors.values()),
        marker="o",
        color="black",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Mean Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Mean Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )


def plot_sd_of_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    sd_errors = {i: np.std(errors[i]) for i in errors.keys()}

    ax.plot(
        list(sd_errors.keys()),
        list(sd_errors.values()),
        marker="o",
        color="black",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Standard Deviation of Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Standard Deviation of Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )


def plot_rmse_of_errors(
    fig: Figure,
    ax: Axes,
    errors: dict[int, npt.NDArray[np.float64]],
    error_type: Literal["energy", "force"],
) -> None:
    rmsd_errors = [np.sqrt(np.mean(errors[i] ** 2)) for i in errors.keys()]

    ax.plot(
        list(errors.keys()),
        rmsd_errors,
        marker="o",
        color="black",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Root Mean Squared Relative Energy Error / kcal mol$^{-1}$"
        if error_type == "energy"
        else "Root Mean Squared Relative Force Error / kcal mol$^{-1}$ Å$^{-1}$"
    )


def plot_error_statistics(
    fig: Figure,
    axs: npt.NDArray[Any],
    errors: dict[Literal["energy", "force"], dict[int, npt.NDArray[np.float64]]],
) -> None:
    """Plot the error statistics for the energy and force errors."""

    axs = axs.flatten()
    plot_distributions_of_errors(fig, axs[0], errors["energy"], "energy")
    plot_distributions_of_errors(fig, axs[1], errors["force"], "force")
    # Hide the legend in the first plot
    axs[0].legend().set_visible(False)

    # Plot the rmsds of the errors
    plot_rmse_of_errors(fig, axs[2], errors["energy"], "energy")
    plot_rmse_of_errors(fig, axs[3], errors["force"], "force")

    # Plot the mean errors
    # plot_mean_errors(fig, axs[4], errors, "energy")
    # plot_mean_errors(fig, axs[5], errors, "force")

    # # Plot the standard deviation of the errors
    plot_sd_of_errors(fig, axs[4], errors["energy"], "energy")
    plot_sd_of_errors(fig, axs[5], errors["force"], "force")


def analyse_workflow(workflow_settings: WorkflowSettings) -> None:
    """Analyse the results of a BespokeFitSMEE workflow."""

    with plt.style.context(PLT_STYLE):
        # Plot the losses
        path_manager = workflow_settings.get_path_manager()
        stage = OutputStage(StageKind.PLOTS)
        path_manager.mk_stage_dir(stage)

        output_paths_by_output_type = path_manager.get_all_output_paths_by_output_type()
        training_metric_paths = dict(
            enumerate(output_paths_by_output_type[OutputType.TRAINING_METRICS])
        )
        losses = read_losses(training_metric_paths)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_loss(fig, ax, losses)
        fig.savefig(
            str(path_manager.get_output_path(stage, OutputType.LOSS_PLOT)),
            dpi=300,
            bbox_inches="tight",
        )

        # Plot the errors
        scatter_paths = dict(enumerate(output_paths_by_output_type[OutputType.SCATTER]))
        errors = read_errors(scatter_paths)
        fig, axs = plt.subplots(3, 2, figsize=(13, 18))
        plot_error_statistics(fig, axs, errors)
        fig.savefig(
            str(path_manager.get_output_path(stage, OutputType.ERROR_PLOT)),
            dpi=300,
            bbox_inches="tight",
        )
