import os
import numpy as np
from . import fileio


class DistanceDistribution(object):
    """
    A Prcurve represents the P(r) distribution that results from the fourier transform of SAXS data into real space.

    Parameters
    ----------
    distances : array_like
        Real space distances. Assumed to be sorted in ascending order. The final
        maximum value of this array is the Dmax of the P(r) data.
    pvals : array_like
        The relative proability of each `rvals`.
    error : array_like
        The uncertainty of each r,P(r) point. Generally unknowable.
    name : string
        User-friendly identifier for this P(r) dataset.
    """

    def __init__(self, distances, pvals, error=None, name=None):
        # turn into numpy arrays first for fast length equality checks
        distances = np.array(distances, copy=True, dtype=np.float64)
        pvals = np.array(pvals, copy=True, dtype=np.float64)
        if error is not None:
            error = np.array(error, copy=True, dtype=np.float64)

        # parameter checks
        if distances is None or pvals is None or len(distances) != len(pvals) \
                or (error is not None and len(error) != len(distances)):
            raise ValueError("Distance distributions should be constructed of two or three equal length arrays of "
                             "values")
        elif not np.all(distances[:-1] <= distances[1:]):
            raise ValueError("Provided distances array should be sorted")
        else:
            self.distances = distances
            self.pvals = pvals
            self.error = error
            self.name = name

    @property
    def dmax(self):
        """
        Dmax is assumed to be the largest measured distance
        :return: the final value in the sorted distances array
        """
        return self.distances[-1]

    def normalize_to_unity_area(self):
        area = np.trapz(self.pvals, x=self.distances)
        self.pvals /= area
        if self.error is not None:
            self.error /= area

    def real_space_rg(self):
        return np.trapz(self.pvals, x=self.distances * self.distances)


def read_gnom_file(path):
    f = open(fileio.fix_filepath(path), 'rU')
    # skip ahead to the relevant starting line
    for line in f:
        tokens = line.strip().split()
        if tokens == ['R', 'P(R)', 'ERROR']:
            break

    ret = DistanceDistribution(*fileio.load_data_columns(file_handle=f), name=os.path.basename(path))
    f.close()
    return ret


def read_distancedistribution_file(path):
    return DistanceDistribution(*load_data_columns(path), name=os.path.basename(path))
