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

    def __repr__(self):
        """
        Return a summary for interactive console usage
        """
        if self.q is None:
            return super().__repr__(self)
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
        elif self.error is None:
            return "\n".join(["{:.6f}\t{:.6f}".format(q, i) for (q, i) in zip(self.q, self.i)])
        else:
            return "\n".join(["{:.6f}\t{:.6f}\t{:.6f}".format(q, i, e)
                              for (q, i, e) in zip(self.q, self.i, self.error)])

    def integrate(self):
        return np.trapz(self.i, x=self.q)

    def total_scattered_intensity(self):
        return np.trapz(self.i * self.q, x=self.q)

    def vc(self, izero=None):
        if izero is None:
            if hasattr(self, 'izero'):
                izero = self.izero
            elif np.isclose(self.q[0], 0.0):
                izero = self.q[0]
            else:
                raise ValueError('Could not determine I(0) and no value provided')
        return izero / self.total_scattered_intensity()

    #
    # Begin block of functions to make Saxscurves behave like lists
    #
    def indexof(self, q, greedy=True, rtol=1e-05, atol=1e-08):
        """
        Returns the index for the point in this Saxscurve where the q value is
        closest to `q`.

        By default, always returns an answer, even if `q` is outside the q range
        of the Saxscurve (in which case this will return 0 or len(Saxscurve)-1)

        Parameters
        ----------
        q : numeric
            The value to find in the q of this Saxscurve
        greedy : bool, optional
            By default (greedy = True) this method returns the closest qval
            regardless of how close the paramter `q` matches anything in the
            data. Setting this to False, a IndexError will be raised if `q` is
            not reasonably close to a value in this Saxscurve's q.
        rtol : numeric, optional
            Relative tolerance of different to use if greedy = false
        atol : numeric, optional
            Absolute tolerance of difference to use if greedy = false

        Returns
        -------
        indexOf : int
            The index `i` where this Saxscurve.q[i] is closest to the
            parameter `q`
        """

        # the int(..) call is neccessary here because the translation betweeen
        #    the return type of argmin(..), numpy int64 or int32, and python int
        #    are platform dependant and unpredictable.
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
            if self.error is not None:
                ret += (self.error[key],)
            return ret
        # a floating point key behaves like Saxscurve[Saxscurve.indexof(key)]
        elif isinstance(key, float):
            index = self.indexof(key)
            ret = (self.q[index], self.i[index],)
            if self.error is not None:
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
                             error=None if self.error is None else self.error[start:stop:key.step],
                             name=self.name + '_slice' + str(key.start) + '-' + str(min(key.stop, len(self.q))))
        elif isinstance(key, tuple):
            index_min = self.indexof(key[0])
            index_max = self.indexof(key[1])
            return self[index_min:index_max]
        else:
            raise IndexError

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
    """
    This internal function represents the common add logic implemented by
    __add__, __iadd__, etc concerning addition/subtraction of two Saxscurves.

    Properly deals with overlapping but not equal q grids, in which case the
    returned values will include only the shared grid.
    Propgates error where possible, else passes on any errors in either operand.

    Parameters
    ----------
    curve1 : Saxscurve
        Left hand operand in addition / subtraction.
    curve2 : Saxscurve
        Right hand operand in addition / subtraction.

    Returns
    -------
    q : 1d array
        The overlapping q grid of both operands
    i : 1d array
        The result of addition of the I(q) values on the overlapping q
    error : 1d array
        The uncertainty in returned `i`. If both operands have associated
        errors, this is the propgated error over the addition operation. If
    `   either operands have errors,
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
            curve1_e_filtered = curve1.error[curve1_overlap] if curve1.error is not None else None
            curve2_i_filtered = curve2.i[curve2_overlap]
            curve2_e_filtered = curve2.error[curve2_overlap] if curve2.error is not None else None
        else:
            raise ValueError('Cannot subtract data with non-overlapping q grids')
    new_i = curve1_i_filtered + curve2_i_filtered
    new_e = curve1_e_filtered if curve2.error is None else curve2_e_filtered if curve1.error is None else np.sqrt(
        curve1_e_filtered ** 2.0 + curve2_e_filtered ** 2.0)
    return curve1_q_filtered, new_i, new_e


def subrange_average(saxsdata, ranges, weight_by_error=False, error_model='propagate'):
    # use a masked array to mask out undesirable ranges
    def get_mask(saxscurve, range):
        mask = np.ones(len(saxscurve), dtype=np.int)
        mask[range[0]:range[1]] = 0
        return mask

    masked_i = np.ma.array([x.i for x in saxsdata], mask=[get_mask(x, y) for x, y in zip(saxsdata, ranges)])
    # error weights are the inverse variance
    weights = np.array([x.error for x in saxsdata])**-2.0 if weight_by_error else None
    avg_i = np.ma.average(masked_i, axis=0, weights=weights)

    masked_e = np.ma.array([x.error for x in saxsdata], mask=masked_i.mask)
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
            n = np.ma.count(masked_e, axis = 0)
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

    return Saxscurve(q=saxsdata[0].q, i=avg_i, error=avg_e)


def read_saxs_file(path):
    return Saxscurve(*load_data_columns(path), name=os.path.basename(path))
