"""Apply OpenFF parameters to molecule, cluster conformers by RMSD and train"""

import functools
import logging
from pathlib import Path
from typing import Protocol, TypedDict, Unpack

import datasets
import datasets.combine
import descent
import descent.optim
import loguru
import smee
import tensorboardX
import torch
from tqdm import tqdm

# from .sample import get_data_MLMD, get_data_MMMD
from .loss_functions import get_loss_closure_fn, prediction_loss
from .outputs import OutputType
from .settings import (
    TrainingSettings,
)
from .utils.register import get_registry_decorator
from .utils.typing import OptimiserName, PathLike
from .writers import (
    open_writer,
    report,
    write_metrics,
)

logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("descent").setLevel(logging.DEBUG)

logger = loguru.logger


class TrainingFnArgs(TypedDict):
    """Arguments for training functions."""

    trainable_parameters: torch.Tensor
    trainable: descent.train.Trainable
    topology: smee.TensorTopology
    dataset: datasets.Dataset
    dataset_test: datasets.Dataset
    settings: TrainingSettings
    output_paths: dict[OutputType, Path]
    device: torch.device


class TrainFn(Protocol):
    """A protocol for training functions."""

    def __call__(
        self, **kwargs: Unpack[TrainingFnArgs]
    ) -> tuple[torch.Tensor, descent.train.Trainable]: ...


_TRAINING_FNS_REGISTRY: dict[OptimiserName, TrainFn] = {}
"""Registry of training functions for different optimiser names."""

_register_training_fn = get_registry_decorator(_TRAINING_FNS_REGISTRY)


@_register_training_fn("lm")
def train_levenberg_marquardt(
    trainable_parameters: torch.Tensor,
    trainable: descent.train.Trainable,
    topology: smee.TensorTopology,
    dataset: datasets.Dataset,
    dataset_test: datasets.Dataset,
    settings: TrainingSettings,
    output_paths: dict[OutputType, PathLike],
    device: torch.device,
) -> tuple[torch.Tensor, descent.train.Trainable]:
    """
    Iterate the training process using the Levenberg-Marquardt algorithm.

    Parameters
    ----------
        trainable_parameters: torch.Tensor
            The parameters to be optimized.
        trainable: descent.train.Trainable
            The trainable object containing the parameters.
        topology: smee.TensorTopology
            The topology of the system.
        dataset: datasets.Dataset
            The dataset to be used for training.
        dataset_test: datasets.Dataset
            The dataset to be used for testing.
        settings: TrainingSettings
            The settings object containing training parameters.
        output_dir: PathLike
            The directory to write output files to.
        output_paths: dict[OutputType, PathLike]
            A mapping of output types to filesystem paths. The following keys are
            expected:
                - OutputType.TENSORBOARD
                - OutputType.TRAINING_METRICS
        device: torch.device
            The device to perform training on.

    Returns
    -------
        tuple[torch.Tensor, descent.train.Trainable]
            The updated parameters and the trainable object.
    """
    # Make sure we have all the required output paths and no others
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    # Run the training with the LM optimiser
    lm_config = descent.optim.LevenbergMarquardtConfig(
        mode="adaptive", n_convergence_criteria=2, max_steps=100
    )

    closure_fn = get_loss_closure_fn(
        trainable,
        topology,
        dataset,
    )

    correct_fn = trainable.clamp

    report_fn = functools.partial(
        report,
        trainable=trainable,
        topology=topology,
        dataset_test=dataset_test,
        metrics_file=output_paths[OutputType.TRAINING_METRICS],
        experiment_dir=Path(output_paths[OutputType.TENSORBOARD]),
    )

    trainable_parameters = descent.optim.levenberg_marquardt(
        trainable_parameters, lm_config, closure_fn, correct_fn, report_fn
    )
    trainable_parameters.requires_grad_(True)

    return trainable_parameters, trainable


@_register_training_fn("adam")
def train_adam(
    trainable_parameters: torch.Tensor,
    trainable: descent.train.Trainable,
    topology: smee.TensorTopology,
    dataset: datasets.Dataset,
    dataset_test: datasets.Dataset,
    settings: TrainingSettings,
    output_paths: dict[OutputType, PathLike],
    device: torch.device,
) -> tuple[torch.Tensor, descent.train.Trainable]:
    """
    Iterate the training process using the Adam optimizer.

    Parameters
    ----------
        trainable_parameters: torch.Tensor
            The parameters to be optimized.
        trainable: descent.train.Trainable
            The trainable object containing the parameters.
        topology: smee.TensorTopology
            The topology of the system.
        dataset: datasets.Dataset
            The dataset to be used for training.
        dataset_test: datasets.Dataset
            The dataset to be used for testing.
        settings: TrainingSettings
            The settings object containing training parameters.
        output_paths: dict[OutputType, PathLike]
            A mapping of output types to filesystem paths. The following keys are
            expected:
                - OutputType.TENSORBOARD
                - OutputType.TRAINING_METRICS
        device: torch.device
            The device to perform training on.

    Returns
    -------
        tuple[torch.Tensor, descent.train.Trainable]
            The updated parameters and the trainable object.
    """
    # Make sure we have all the required output paths and no others
    if set(output_paths.keys()) != settings.output_types:
        raise ValueError(
            f"Output paths must contain exactly the keys {settings.output_types}"
        )

    # run the ML training
    with open(output_paths[OutputType.TRAINING_METRICS], "w") as metrics_file:
        with open_writer(Path(output_paths[OutputType.TENSORBOARD])) as writer:
            optimizer = torch.optim.Adam(
                [trainable_parameters], lr=settings.learning_rate, amsgrad=True
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=settings.learning_rate_decay
            )
            for v in tensorboardX.writer.hparams(
                {"optimizer": "Adam", "lr": settings.learning_rate}, {}
            ):
                writer.file_writer.add_summary(v)
            for i in tqdm(
                range(settings.n_epochs),
                leave=False,
                colour="blue",
                desc="Optimising MM parameters",
            ):
                loss_trn = prediction_loss(
                    dataset,
                    trainable.to_force_field(trainable_parameters),
                    topology,
                    settings.loss_force_weight,
                    str(device),
                )
                if i % 10 == 0:
                    loss_tst = prediction_loss(
                        dataset_test,
                        trainable.to_force_field(trainable_parameters),
                        topology,
                        settings.loss_force_weight,
                        str(device),
                    )
                    write_metrics(i, loss_trn, loss_tst, writer, metrics_file)
                loss_trn.backward(retain_graph=True)  # type: ignore[no-untyped-call]
                # trainable.freeze_grad()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                trainable.clamp(trainable_parameters)
                if i % settings.learning_rate_decay_step == 0:
                    scheduler.step()
        # some book-keeping and outputting
        loss_tst = prediction_loss(
            dataset_test,
            trainable.to_force_field(trainable_parameters),
            topology,
            settings.loss_force_weight,
            str(device),
        )
        write_metrics(settings.n_epochs, loss_trn, loss_tst, writer, metrics_file)

        return trainable_parameters, trainable
