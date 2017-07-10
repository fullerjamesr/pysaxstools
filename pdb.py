import os
import math
import numpy as np
from scipy.spatial.distance import pdist
import Bio.PDB
from .pr import DistanceDistribution
from .fileio import fix_filepath


def read_pdb_from_file(path):
    path = fix_filepath(path)
    p = Bio.PDB.PDBParser()
    return p.get_structure(os.path.basename(path), path)


def center_of_mass(structure):
    mass_sum = 0.0
    mass_coord = np.array([0.0, 0.0, 0.0])
    for atom in structure.get_atoms():
        mass_sum += atom.mass
        mass_coord += atom.coord * atom.mass
    return mass_coord / mass_sum


def rg(structure, spherical_correction=True):
    center = center_of_mass(structure)
    mass_sum = 0.0
    distance_sum = 0.0
    for atom in structure.get_atoms():
        mass_sum += atom.mass
        distance_sum += np.sum(np.square(atom.coord - center)) * atom.mass
    distance_sum /= mass_sum
    if spherical_correction:
        distance_sum += 1.57464
    return math.sqrt(distance_sum)


def dmax(structure):
    coords = np.array([x.coord for x in structure.get_atoms()])
    distances = pdist(coords)
    return np.max(distances)


def distance_distribution(structure, bins='auto'):
    coords = np.array([x.coord for x in structure.get_atoms()])
    distances = pdist(coords)
    dmax = np.max(distances)
    hist, bin_edges = np.histogram(distances, bins=bins, range=(0.0, dmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return DistanceDistribution(bin_centers, hist, error=None, name=structure.id)
