"""Functionality for analysing the results of a BespokeFitSMEE run."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from openff import units
from openff.interchange.components.interchange import Interchange
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from tqdm import tqdm

from .settings import TrainingConfig

PLT_STYLE = "ggplot"

POTENTIAL_KEYS = Literal[
    "Bonds", "Angles", "ProperTorsions", "ImproperTorsions", "vdW", "Electrostatics"
]


@dataclass
class OutputData:
    """
    Dataclass to store the output data from the calculations
    """

    training_config: TrainingConfig
    """The config specifying the training parameters."""

    @property
    def path(self) -> Path:
        """Path to the output directory."""
        return Path(self.training_config.output_dir).resolve()

    @cached_property
    def n_iter(self) -> int:
        # Count the number of "trained-<i>.offxml" files, minus one for the initial force field
        return len(list(self.path.glob("trained-*.offxml"))) - 1

    @cached_property
    def molecule(self) -> Molecule:
        smiles = self.training_config.smiles
        mol = Molecule.from_smiles(smiles)
        mol.generate_conformers(n_conformers=1)
        return mol

    @cached_property
    def force_fields(self) -> dict[int, ForceField]:
        return {
            i: ForceField(self.path / f"trained-{i}.offxml")
            for i in range(self.n_iter + 1)
        }

    @cached_property
    def interchanges(self) -> dict[int, Interchange]:
        interchanges = {}
        for i in range(self.n_iter + 1):
            interchanges[i] = Interchange.from_smirnoff(
                force_field=self.force_fields[i], topology=[self.molecule]
            )
        return interchanges

    @cached_property
    def losses(self) -> pd.DataFrame:
        loss_datafiles = [
            self.path / f"training-{i}.data" for i in range(1, self.n_iter + 1)
        ]
        idxs, losses_test, losses_train, iteration = [], [], [], []

        for i, loss_datafile in enumerate(loss_datafiles):
            df = pd.read_csv(
                loss_datafile,
                delim_whitespace=True,
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

    @cached_property
    def energy_errors(self) -> dict[int, npt.NDArray[np.float64]]:
        return self._read_errors()[0]

    @cached_property
    def force_errors(self) -> dict[int, npt.NDArray[np.float64]]:
        return self._read_errors()[1]

    def _read_errors(
        self,
    ) -> tuple[dict[int, npt.NDArray[np.float64]], dict[int, npt.NDArray[np.float64]]]:
        error_datafiles = [
            self.path / f"trained-{i}.scat" for i in range(self.n_iter + 1)
        ]
        energy_errors = {
            # i: np.loadtxt(f)[:, 1] - np.loadtxt(f)[:, 0]
            # for i, f in enumerate(error_datafiles)
            i: np.loadtxt(f)[:, 0]
            for i, f in enumerate(error_datafiles)
        }
        # Drop nan values from the energy errors
        energy_errors = {
            i: e[~np.isnan(e)]
            for i, e in energy_errors.items()
            if len(e[~np.isnan(e)]) > 0
        }
        force_errors = {
            # i: np.loadtxt(f)[:, 3] - np.loadtxt(f)[:, 2]
            # for i, f in enumerate(error_datafiles)
            i: np.loadtxt(f)[:, 1]
            for i, f in enumerate(error_datafiles)
        }

        return energy_errors, force_errors


def plot_loss(fig: Figure, ax: Axes, output_data: OutputData) -> None:
    # Colour by iteration - full line for train, dotted for test
    for i in range(output_data.n_iter + 1):
        ax.plot(
            output_data.losses[output_data.losses["iteration"] == i].index,
            output_data.losses[output_data.losses["iteration"] == i]["loss_train"],
            label=f"train-{i}",
            color=f"C{i}",
        )
        ax.plot(
            output_data.losses[output_data.losses["iteration"] == i].index,
            output_data.losses[output_data.losses["iteration"] == i]["loss_test"],
            label=f"test-{i}",
            color=f"C{i}",
            linestyle="--",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.savefig(output_data.path / "loss.png", dpi=300, bbox_inches="tight")


def plot_distributions_of_errors(
    fig: Figure,
    ax: Axes,
    output_data: OutputData,
    error_type: Literal["energy", "force"],
) -> None:
    # Colour by iteration
    # Use continuous colourmap for the iterations
    colours = plt.cm.get_cmap("viridis")(np.linspace(0, 1, output_data.n_iter + 1))

    for i in range(output_data.n_iter + 1):
        ax.hist(
            (
                output_data.energy_errors[i]
                if error_type == "energy"
                else output_data.force_errors[i]
            ),
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


def plot_mean_errors(
    fig: Figure,
    ax: Axes,
    output_data: OutputData,
    error_type: Literal["energy", "force"],
) -> None:
    mean_errors = [
        (
            np.mean(output_data.energy_errors[i])
            if error_type == "energy"
            else np.mean(output_data.force_errors[i])
        )
        for i in range(output_data.n_iter + 1)
    ]

    ax.plot(
        range(output_data.n_iter + 1),
        mean_errors,
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
    output_data: OutputData,
    error_type: Literal["energy", "force"],
) -> None:
    sd_errors = [
        (
            np.std(output_data.energy_errors[i])
            if error_type == "energy"
            else np.std(output_data.force_errors[i])
        )
        for i in range(output_data.n_iter + 1)
    ]

    ax.plot(
        range(output_data.n_iter + 1),
        sd_errors,
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
    output_data: OutputData,
    error_type: Literal["energy", "force"],
) -> None:
    rmsd_errors = [
        (
            np.sqrt(np.mean(output_data.energy_errors[i] ** 2))
            if error_type == "energy"
            else np.sqrt(np.mean(output_data.force_errors[i] ** 2))
        )
        for i in range(output_data.n_iter + 1)
    ]

    ax.plot(
        range(output_data.n_iter + 1),
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
    fig: Figure, axs: npt.NDArray[Any], output_data: OutputData
) -> None:
    """Plot the error statistics for the energy and force errors."""

    axs = axs.flatten()
    plot_distributions_of_errors(fig, axs[0], output_data, "energy")
    plot_distributions_of_errors(fig, axs[1], output_data, "force")
    # Hide the legend in the first plot
    axs[0].legend().set_visible(False)

    # Plot the rmsds of the errors
    plot_rmse_of_errors(fig, axs[2], output_data, "energy")
    plot_rmse_of_errors(fig, axs[3], output_data, "force")

    # Plot the mean errors
    # plot_mean_errors(fig, axs[4], output_data, "energy")
    # plot_mean_errors(fig, axs[5], output_data, "force")

    # # Plot the standard deviation of the errors
    plot_sd_of_errors(fig, axs[4], output_data, "energy")
    plot_sd_of_errors(fig, axs[5], output_data, "force")


def plot_ff_differences(
    fig: Figure,
    ax: Axes,
    output_data: OutputData,
    potential_type: POTENTIAL_KEYS,
    parameter_key: str,
    iterations: tuple[int, int],
) -> dict[str, float]:
    # Get the initial and final potentials
    labeled_start = output_data.force_fields[iterations[0]].label_molecules(
        output_data.molecule.to_topology()
    )[0]
    labeled_end = output_data.force_fields[iterations[1]].label_molecules(
        output_data.molecule.to_topology()
    )[0]

    objects_start = set(labeled_start[potential_type].values())
    objects_end = set(labeled_end[potential_type].values())

    # Convert to dicts with the id as the key
    potentials_start = {p.id: p.to_dict() for p in objects_start}
    potentials_end = {p.id: p.to_dict() for p in objects_end}

    parameter_keys = [
        k
        for k in potentials_start[list(potentials_start.keys())[0]].keys()
        if k not in ["smirks", "id"]
    ]
    if parameter_key not in parameter_keys:
        raise ValueError(f"Parameter key {parameter_key} not found in {parameter_keys}")

    # Get the differences for each key id
    differences = {
        key: potentials_end[key][parameter_key] - potentials_start[key][parameter_key]
        for key in potentials_start.keys()
    }
    differences_first_key = list(differences.keys())[0]

    # Plot the differences
    q_units = (
        units.unit.degrees
        if potential_type == "Angles" and parameter_key == "angle"
        else differences[differences_first_key].units
    )
    ax.bar(
        list(differences.keys()),
        [float(differences[k] / q_units) for k in differences.keys()],
    )

    ax.set_ylabel(f"{parameter_key} difference / {q_units}")
    ax.set_xlabel("Key ID")
    ax.set_title(f"{potential_type} {parameter_key} differences")

    # Rotate tick labels 90
    ax.set_xticklabels(differences.keys(), rotation=90)

    return differences


def plot_ff_values(
    fig: Figure,
    ax: Axes,
    output_data: OutputData,
    potential_type: POTENTIAL_KEYS,
    parameter_key: str,
) -> None:
    # nice colour map for the iterations
    colours = plt.cm.get_cmap("viridis")(np.linspace(0, 1, output_data.n_iter + 1))

    for i in range(output_data.n_iter + 1):
        # Get the initial and final potentials
        labeled = output_data.force_fields[i].label_molecules(
            output_data.molecule.to_topology()
        )[0]

        objects = set(labeled[potential_type].values())

        # Convert to dicts with the id as the key
        potentials = {p.id: p.to_dict() for p in objects}

        parameter_keys = [
            k
            for k in potentials[list(potentials.keys())[0]].keys()
            if k not in ["smirks", "id"]
        ]
        if parameter_key not in parameter_keys:
            raise ValueError(
                f"Parameter key {parameter_key} not found in {parameter_keys}"
            )

        # Get the differences for each key id
        vals = {key: potentials[key][parameter_key] for key in potentials.keys()}
        vals_first_key = list(vals.keys())[0]

        # Plot the differences
        q_units = (
            units.unit.degrees
            if potential_type == "Angles" and parameter_key == "angle"
            else vals[vals_first_key].units
        )
        # Plot as circles with correct colour, not bars
        x_vals = np.arange(len(vals.keys()))
        ax.scatter(
            x_vals,
            [float(vals[k] / q_units) for k in vals.keys()],
            color=colours[i],
            label=f"Iteration {i}",
            # small
            s=10,
        )

    # Set x labels to be the key ids
    ax.set_xticks(np.arange(len(vals.keys())))
    ax.set_xticklabels(list(vals.keys()), rotation=90)
    ax.set_ylabel(f"{parameter_key} / {q_units}")
    ax.set_xlabel("Key ID")
    ax.set_title(f"{potential_type} {parameter_key}")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


pot_types_and_param_keys: dict[POTENTIAL_KEYS, list[str]] = {
    "Bonds": ["length", "k"],
    "Angles": ["angle", "k"],
    "ProperTorsions": ["k1", "k2", "k3", "k4", "phase1", "phase2", "phase3", "phase4"],
    "ImproperTorsions": ["k1", "phase1"],
}


def plot_all_ffs(
    output_data: OutputData, plot_type: Literal["values", "differences"]
) -> None:
    plt_fn = plot_ff_values if plot_type == "values" else plot_ff_differences
    extra_args = (
        {"iterations": (0, output_data.n_iter - 1)}
        if plot_type == "differences"
        else {}
    )

    # 1 column per potential type
    ncols = len(pot_types_and_param_keys)
    # 1 row for each of the greatest number of parameters
    nrows = max([len(v) for v in pot_types_and_param_keys.values()])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))

    for i, (potential_type, param_keys) in tqdm(
        enumerate(pot_types_and_param_keys.items()), total=len(pot_types_and_param_keys)
    ):
        for j, param_key in enumerate(param_keys):
            plt_fn(fig, axs[j, i], output_data, potential_type, param_key, **extra_args)

        # Hide the remaining axes
        for k in range(j + 1, nrows):
            axs[k, i].axis("off")

    axs[2, 3].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()

    fig.savefig(
        str(output_data.path / f"ff_{plot_type}.png"), dpi=300, bbox_inches="tight"
    )


def plot_all(output_data: OutputData) -> None:
    """Plot all the results from the BespokeFitSMEE run."""

    with plt.style.context(PLT_STYLE):
        # Plot the loss
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_loss(fig, ax, output_data)
        fig.savefig(str(output_data.path / "loss.png"), dpi=300, bbox_inches="tight")

        # Plot the error statistics
        fig, axs = plt.subplots(3, 2, figsize=(13, 18))
        plot_error_statistics(fig, axs, output_data)
        fig.savefig(
            str(output_data.path / "error_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
