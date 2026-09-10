"""GPU memory management utilities."""

import torch
from openmm import Integrator
from openmm.app import Simulation


def free_gpu_memory() -> None:
    """Return cached GPU memory to the driver, if there is a GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
        torch.cuda.empty_cache()  # Now GPU is idle, memory can actually be freed


def cleanup_simulation(
    simulation: Simulation, integrator: Integrator | None = None
) -> None:
    """Clean up OpenMM simulation and free GPU memory.

    This function properly releases GPU resources by deleting the simulation
    and integrator objects, synchronizing CUDA operations, and emptying the
    GPU cache. The synchronization step is critical - without it, the cache
    may be cleared while GPU operations are still in flight.

    Parameters
    ----------
    simulation : Simulation
        The OpenMM simulation object to clean up.
    integrator : Integrator | None, optional
        The integrator object to clean up. If None, only the simulation
        is deleted.

    Examples:
    --------
    >>> integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)
    >>> simulation = Simulation(topology, system, integrator)
    >>> # ... use simulation ...
    >>> cleanup_simulation(simulation, integrator)
    """
    del simulation
    if integrator is not None:
        del integrator
    free_gpu_memory()
