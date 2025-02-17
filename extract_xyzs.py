"""Apply OpenFF parameters to molecule, cluster conformers by RMSD and train"""
from argparse import ArgumentParser
import datetime
import os
import sys
import pathlib
from contextlib import redirect_stdout, redirect_stderr

from data_maker     import *
from writers        import *
from parameterizer  import *
from loss_functions import *

import openff.interchange
import openff.toolkit
import openff.units
from openmm.app  import *
from openmm      import *
from openmm.unit import *
from openmmml    import MLPotential
import tensorboardX
import torch
import torch.distributed
import datasets
import datasets.distributed
import datasets.table
import typing

import descent.optim
import descent.targets.energy
import descent.utils.loss
import descent.utils.reporting

def main(world_size: int, args: list):
#   read in the command line inputs
    source_train = args.data    # Location of data"
    print("Extracting Training Dataset")
    dataset = datasets.Dataset.load_from_disk(f"{source_train}")
    model = openff.interchange.Interchange()
    for entry in dataset:
        smiles   = entry["smiles"]
        molecule = openff.toolkit.Molecule.from_smiles(smiles,hydrogens_are_explicit=False)
        coords   = (entry["coords"].reshape(len(entry["energy"]), -1, 3).tolist())
        molecule._conformers = [c * openff.units.unit.angstrom for c in coords]
#        print(molecule._conformers)
        molecule.to_file('data.pdb', file_format='pdb') 
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    parser = ArgumentParser()
    parser.add_argument("--data",    type=str,   help="Location of pre-calculated data set",  required=False, default="data")
    args = parser.parse_args()
    world_size = 1
    main(world_size, args)

