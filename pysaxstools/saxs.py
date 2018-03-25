import os
import numpy as np
from . import util
from .fileio import load_data_columns


class Saxscurve(object):
    """
    A Saxscurve consists of 2 or 3 equal length arrays of floating point values representing and x-ray scattering
    profile:
    q: The scattering angle
    i: The scattering intensity
    error: (optional) The uncertainty in the scattering intensity
    name: string, optional, a user-friendly ID
    """

    def __init__(self, q, i, error=None, name=''):
        # TODO: Add an optional argument "copy=(True, True, True)" to define whether a copy is needed for (q, i, error)
        #       I think this would offer some big speedups. Note that the intention is still that Saxscurves are
        #       immutable except when operators like *= += etc are used.
        # TODO: Update functions in this file and elsewhere to NOT copy, when appropriate.
        # turn into numpy arrays first for fast length equality checks
        q = np.array(q, copy=True, dtype=np.float64)
        i = np.array(i, copy=True, dtype=np.float64)
        if error is not None:
            error = np.array(error, copy=True, dtype=np.float64)
        # parameter checks
        if q is None or i is None or len(q) != len(i) or (error is not None and len(q) != len(error)):
            raise ValueError("Saxscurves should be constructed of two or three equal length arrays of values")
        elif not np.all(q[:-1] <= q[1:]):
            raise ValueError("Provided q values array should be sorted")
        else:
            self.name = name
            self.q = q
            self.i = i
            self.error = error

    @property
    def has_error(self):
        return self.error is not None

    def __repr__(self):
        """
        Return a summary for interactive console usage
        """
        if self.q is None:
            return super().__repr__()
        else:
            return ("saxs data %s \n q range %f to %f" %
                    (self.name, self.q[0], self.q[-1]))

    def __str__(self):
        """
        Returns this Saxscurve as a string.

        Format is 3 columns q,I(q),error that are tab delimited.
        """
        if self.q is None or self.i is None:
            return super().__str__()
        elif self.has_error:
            return "\n".join(["{:.6f}\t{:.6f}\t{:.6f}".format(q, i, e)
                              for (q, i, e) in zip(self.q, self.i, self.error)])
        else:
            return "\n".join(["{:.6f}\t{:.6f}".format(q, i) for (q, i) in zip(self.q, self.i)])

    def integrate(self):
        """Find the integral of Intensity as a function of q

        A shortcut for using numpy.trapz to approximate the integral of I dq

        Returns
        -------
        float
            The integral of I dq as calculated by numpy.trapz
        """
        return np.trapz(self.i, x=self.q)

    def total_scattered_intensity(self):
        """Compute the total scattered intensity of this Saxscurve

        The total scattered intensity of a small angle scattering profile is the integral of q*I dq. See: Rambo and
        Tainer, Nature (2013).

        Returns
        -------
        float
            The result of integrating qI dq.
        """
        return np.trapz(self.i * self.q, x=self.q)

    def vc(self, i_0):
        """Volume of correlation calculation

        Rambo and Tainer (Nature 2013) define this convenient property, Vc, which can be used (like Rg) to monitor
        conformational changes. Its convenience comes from that it is easily calculated from a single scattering curve.
        Functionally, it is calculated as I(0) / Total Scattered Intensity

        Parameters
        ----------
        i_0 : float
            I(0), the intensity at q = 0. For experimental data, this is unlikely to be measured directly, and needs to
            be calculated, probably by fitting the gunier region.

        Returns
        -------
        float
            Vc, the Volume of Correlation for the scattering molecule
        """
        return i_0 / self.total_scattered_intensity()

    #
    # Begin block of functions to make Saxscurves behave like lists
    #
    def indexof(self, q, greedy=True, rtol=1e-05, atol=1e-08):
        """Returns the index for the point in this Saxscurve where the q value is closest to `q`.

        By default, always returns an answer, even if `q` is outside the q range of the Saxscurve, in which case this
        will return 0 or len(Saxscurve)-1. This behavior can be changed by setting `greedy=False`. "Closeness" is
        determined by `rtol` and `atol`, as in numpy.isclose() and related functions.

        Parameters
        ----------
        q : numeric
            The value to find in the q of this Saxscurve
        greedy : bool, optional
            By default (`greedy = True`) this method returns the closest qval regardless of how close the paramter `q`
            matches anything in the data. Setting this to False, a IndexError will be raised if `q` is not reasonably
            close to a value in this Saxscurve's q.
        rtol : numeric, optional
            Relative tolerance of different to use if `greedy=False`
        atol : numeric, optional
            Absolute tolerance of difference to use if `greedy=False`

        Returns
        -------
        int
            The index `i` where this Saxscurve.q[i] is closest to the
            provided `q`
        """
        search = np.isclose(self.q, q, rtol=rtol, atol=atol)
        if np.any(search):
            return np.argmax(search)
        elif greedy:
            return np.argmin(np.abs(self.q - q))
        else:
            raise IndexError("Saxscurve has no q near {}".format(q))

    def __len__(self):
        return len(self.q)

    def __getitem__(self, key):
        # a normal integer index
        if isinstance(key, int):
            ret = (self.q[key], self.i[key],)
            if self.has_error:
                ret += (self.error[key],)
            return ret
        # a floating point key behaves like Saxscurve[Saxscurve.indexof(key)]
        elif isinstance(key, float):
            index = self.indexof(key)
            ret = (self.q[index], self.i[index],)
            if self.has_error:
                ret += (self.error[index],)
            return ret
        # slicing can also use ints as true indices or floats to feed to
        #   indexOf(..) to get integer indices
        elif isinstance(key, slice):
            start = key.start
            stop = key.stop
            if isinstance(key.start, float):
                start = self.indexof(key.start)
            if isinstance(key.stop, float):
                stop = self.indexof(key.stop)
            return Saxscurve(q=self.q[start:stop:key.step],
                             i=self.i[start:stop:key.step],
                             error=self.error[start:stop:key.step] if self.has_error else None,
                             name=self.name + '_slice' + str(key.start) + '-' + str(min(key.stop, len(self.q))))
        elif isinstance(key, tuple):
            index_min = self.indexof(key[0])
            index_max = self.indexof(key[1])
            return self[index_min:index_max]
        else:
            # Fallback to numpy indexing
            return Saxscurve(q=self.q[key], i=self.i[key], error=self.error[key] if self.has_error else None)

    #
    # End block of functions to make Saxscurves behave like lists
    #

    #
    # Begin block of functions to make Saxscurves respond to native math
    #
    def __neg__(self):
        return Saxscurve(q=self.q, i=-self.i, error=self.error, name=self.name)

    def __mul__(self, factor):
        return Saxscurve(q=self.q,
                         i=self.i * factor,
                         error=None if self.error is None else self.error * factor,
                         name='' if (self.name is None or self.name == '') else self.name + '_scaled')

    def __imul__(self, factor):
        self.i *= factor
        if self.error is not None:
            self.error *= factor
        return self

    def __truediv__(self, factor):
        return self * (1.0 / factor)

    def __idiv__(self, factor):
        self.i /= factor
        if self.error is not None:
            self.error /= factor
        self.scale = factor
        return self

    def __add__(self, factor):
        if isinstance(factor, Saxscurve):
            return Saxscurve(*_common_saxscurve_add(self, factor))
        else:
            return Saxscurve(self.q, self.i + factor, self.error)

    def __iadd__(self, factor):
        if isinstance(factor, Saxscurve):
            return NotImplemented
        self.i += factor
        return self

    def __sub__(self, factor):
        if isinstance(factor, Saxscurve):
            return Saxscurve(*_common_saxscurve_add(self, -factor))
        else:
            return self + (-1.0 * factor)

    def __isub__(self, factor):
        if isinstance(factor, Saxscurve):
            return NotImplemented
        else:
            self.i -= factor
        return self
    #
    # End block of functions to make Saxscurves respond to native math
    #


def _common_saxscurve_add(curve1, curve2):
    """This internal function represents the common add logic implemented by __add__, __iadd__, etc concerning addition/
    subtraction of two Saxscurves.

    Properly deals with overlapping but not equal q grids, in which case the returned values will include only the
    shared grid. Propagates error where possible, else passes on any errors in either operand.

    Parameters
    ----------
    curve1 : Saxscurve
        Left hand operand in addition / subtraction.
    curve2 : Saxscurve
        Right hand operand in addition / subtraction.

    Returns
    -------
    q : 1d array
        The overlapping q grid of both operands. The actual values are taken from `curve1` (this might be relevant in
        case of insignificant floating point round off differences between `curve1.q` and `curve2.q`
    i : 1d array
        The result of addition of the I(q) values on the overlapping q values
    error : 1d array
        The uncertainty in returned `i`. If both operands have associated errors, this is the propagated error over the
        addition operation. If either operands have errors, theirs is returned. If neither, then this will be None.
    """
    curve1_q_filtered = curve1.q
    curve1_i_filtered = curve1.i
    curve1_e_filtered = curve1.error
    curve2_i_filtered = curve2.i
    curve2_e_filtered = curve2.error

    if len(curve1) == len(curve2) and np.allclose(curve1.q, curve2.q):
        pass
    else:
        curve1_overlap, curve2_overlap = util.whichclose(curve1.q, curve2.q)
        if np.sum(curve1_overlap) > 0:
            curve1_q_filtered = curve1.q[curve1_overlap]
            curve1_i_filtered = curve1.i[curve1_overlap]
            curve1_e_filtered = curve1.error[curve1_overlap] if curve1.has_error else None
            curve2_i_filtered = curve2.i[curve2_overlap]
            curve2_e_filtered = curve2.error[curve2_overlap] if curve2.has_error else None
        else:
            raise ValueError('Cannot subtract data with non-overlapping q grids')
    new_i = curve1_i_filtered + curve2_i_filtered
    if curve1.has_error and curve2.has_error:
        new_e = np.sqrt(curve1_e_filtered * curve1_e_filtered + curve2_e_filtered * curve2_e_filtered)
    elif curve1.has_error:
        new_e = curve1_e_filtered
    elif curve2.has_error:
        new_e = curve2_e_filtered
    else:
        new_e = None
    return curve1_q_filtered, new_i, new_e


def read_saxs_file(path):
    """Read a 3 column file describing a SAXS curve

    The expectation is that the file contains 2 or 3 columns  -- representing q, I(q), and (optionally) uncertainty --
    separated by whitespace. This function is merely convenient boilerplate around pysaxstools.fileio.load_data_columns.

    The underlying method doing the parsing (fileio.load_data_columns) disregards lines where the first non-whitespace
    character is non-numeric.

    Parameters
    ----------
    path : string-like or integer
        A valid argument to the Python built-in function `open` -- i.e., either a string containing a path/filename or
        an integer that will be treated as a file descriptor. This file should have 2 or 3 whitespace delimited columns
        of equal length.

    Returns
    -------
    Saxscurve
        The SAXS curve parsed from the file.
    """
    str_name = os.path.basename(path) if util.value_is_string(path) else str(path)
    return Saxscurve(*load_data_columns(path), name=str_name)


def subrange_average(saxscurves, ranges, weight_by_error=False, error_model='propagate'):
    """Merge SAXS curves across limited ranges

    This function replicates behavior of programs like almerge, where SAXS curves are merged, but only a particular
    range of each curve is used.

    E.g., this can be used to merge the low-q region of a low concentration dataset with
    the high-q region of a high concentration dataset. There are generally some overlapping points in the middle where
    the the average of the two would be taken in order to avoid discontinuities at the joints.

    Parameters
    ----------
    saxscurves : Iterable[Saxscurve]
        Some iterable yielding two or more Saxscurves.
    ranges
        Some iterable yielding tuples of indices corresponding to the ranges in `saxscurves` to be considered.
    weight_by_error : bool
        Where averaging is performed, should the average be weighted by variance (error ** -2)?
    error_model : {'propagate', 'sem'}
        Where averaging is performed, should the resulting uncertainties be calculated by simple propagation of error,
        or should the errors in the dataset be ignored and the errors estimated by standard error of the mean (SEM)?

    Returns
    -------
    Saxscurve
        The result of merging the specified ranges of each input SAXS curve
    """
    # use a masked array to mask out undesirable ranges
    def get_mask(saxscurve, r):
        mask = np.ones(len(saxscurve), dtype=np.int)
        mask[r[0]:r[1]] = 0
        return mask

    masked_i = np.ma.array([x.i for x in saxscurves], mask=[get_mask(x, y) for x, y in zip(saxscurves, ranges)])
    # error weights are the inverse variance
    weights = np.array([x.error for x in saxscurves]) ** -2.0 if weight_by_error else None
    avg_i = np.ma.average(masked_i, axis=0, weights=weights, returned=False)  # type: np.ndarray

    masked_e = np.ma.array([x.error for x in saxscurves], mask=masked_i.mask)
    if error_model == 'propagate':
        if weight_by_error:
            # error is sqrt( 1 / sum(errors^-2) )
            avg_e = np.ma.sqrt(1.0 / np.ma.sum(masked_e ** -2.0, axis=0))
        else:
            # error is sqrt( sum(errors^2) ) / n
            avg_e = np.ma.sqrt(np.ma.sum(masked_e, axis=0)) / np.ma.count(masked_e, axis=0)
    elif error_model == 'sem':
        if weight_by_error:
            # the standard error of weighted mean is...tricky
            # possibly the best estimator was discussed at
            # http://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation
            # it is a hideous long formula
            n = np.ma.count(masked_e, axis=0)
            mean_weights = np.ma.mean(weights, axis=0)
            means_product = (mean_weights * avg_i)[:, np.newaxis]
            weights_i = weights * masked_i
            avg_e = (
                     n / ((n - 1.0) * (np.ma.sum(weights, axis=0)**2.0))
                     * (
                        np.ma.sum((weights_i - means_product)**2.0, axis=0)
                        - avg_i * 2.0 * np.ma.sum((weights-mean_weights[:, np.newaxis])
                                                  * (weights_i - means_product), axis=0)
                        + avg_i**2.0 * np.ma.sum((weights-mean_weights[:, np.newaxis])**2.0, axis=0)))
        else:
            avg_e = np.ma.std(masked_i, axis=0, ddof=1) / (np.ma.sqrt(np.ma.count(masked_i, axis=0)))
    else:
        raise ValueError("parameter error_model must be one of 'propagate' or 'sem'")

    return Saxscurve(q=saxscurves[0].q, i=avg_i, error=avg_e)


def estimate_mass_by_qr(rg, vc=None, saxs=None, i_0=None, molecule='protein'):
    """Estimate the mass of the scattering particle.

    Rambo and Tainer (Nature, 2013) derived a method by which the mass of a particle could be estimated from a single
    SAXS dataset. This method produces an estimate that is usually within ~5% of the true particle mass, assuming
    reasonable data quality and sample homogeneity. As such, this can be used as an indicator of sample quality if the
    particle identity is known: mass estimates that deviate by more than ~10% from the true value suggest the sample may
    not be pure, or there is significant conformational heterogeneity.

    Parameters
    ----------
    rg : float
        This method requires a measurement of Rg for the scattering particle
    vc : float, optional
        This method requires either that the Volume of correlation (Vc) be provided, or that a SAXS dataset and its
        measured I(0) be provided so that Vc can be computed
    saxs : Saxscurve
        This is required if `vc` is not specified, so a value for `vc` (the Volume of correlation) can be calculated
    i_0 : float, optional
        This is required if `vc` is not specified, so a value for `vc` (the Volume of correlation) can be calculated
    molecule : {'protein', 'rna'}
        The type of molecule is needed in order to properly calculate mass

    Returns
    -------
    mass : float
        An estimate for the mass of the particle, in Daltons, given the Vc, Rg, and molecule type provided
    """
    # Must provide Vc or [SAXS data and I(0)]
    if vc is None:
        if i_0 is None or saxs is None:
            raise ValueError("Vc must either be provided or I(0) given to calculate it")
        vc = saxs.vc(i_0)
    qr = vc * vc / rg
    if molecule == 'protein':
        return qr / 0.1231
    elif molecule == 'rna':
        return (qr / 0.00934) ** 0.808
    else:
        raise ValueError("Parameter `molecule` must be either 'protein' or 'rna'")
