import os
import subprocess
import shlex
import numpy as np
import sys


# The following attempts to stop unecessary console window popups on Windows
try:
    _SUINFO = subprocess.STARTUPINFO()
    _SUINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
except:
    _SUINFO = None


def value_is_string(value):
    """Python 2 and 3 compatible string-ness check

    Strings changed between Python 2 and 3 to bring unicode closer into the str fold, while Python 3 eliminated the
    abstract base class `basestring`. How convenient :-/

    See also:
    https://stackoverflow.com/questions/4843173/how-to-check-if-type-of-a-variable-is-string

    Parameters
    ----------
    value : any

    Returns
    -------
    bool
        Is `value` a string?
    """
    if sys.version_info[0] >= 3:
        basestring = str
    return isinstance(value, basestring)


def which(program):
    """
    A handy mimic of unix "which" command from
    http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    Used mainly to see if stuff is installed (i.e. if this doesn't return None).

    Parameters:
        program: a string partial or full path to execute

    Returns:
        either the full path to the executable or None if it's not in the path
    """

    def is_exe(filepath):
        return os.path.isfile(filepath) and os.access(filepath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def check_supported_kwargs(kwargs, supported_args, supported_values=None):
    formatted_kwargs = {}
    for flag in kwargs:
        # Is this a valid command line flag?
        if flag not in supported_args:
            raise TypeError("{!s} is not a recognized argument".format(flag))

        # Does this flag take a value? If so, try a type conversion to verify it's the correct type
        if supported_args[flag] is None:
            arg_value = None
        else:
            arg_value = supported_args[flag](kwargs[flag])

        # If the caller supplied supported_values and this flag is in it, check if the flag value is OK
        if supported_values and flag in supported_values and arg_value not in supported_values[flag]:
            raise ValueError("{!s} is not a supported value for {!s}".format(arg_value, flag))
        formatted_kwargs[flag] = arg_value
    return formatted_kwargs


def kwargs_to_cmd(kwargs):
    """
    Convert a dictionary of command line flags and values to a list of tokens in preparation for a call to
    subprocess.Popen(...)

    Parameters:
        kwargs : a dictionary whose keys are command line switches and whose
        values are the intended values for those switches.

    Returns:
        A list of ['-key','value'] for all keys of kwargs
    """
    arglist = []
    for arg in kwargs:
        if len(arg) > 1:
            arglist.append("--{!s}".format(arg))
        else:
            arglist.append("-{!s}".format(arg))

        if kwargs[arg] is not None:
            arglist.append(str(kwargs[arg]))

    return arglist


def runexternal(cmd, input_pipe=None):
    """
    Shortcut method for the "proper" way to run an external program via
    subprocess, because it's a lot to type.

    Parameters:
        cmd : The command to be passed to the OS, either as a string that will
                be processed by shlex.split or a pre-built list of tokens
        input_pipe : (optional) Information that will be passed to stdin

    Returns:
        A tuple of output,returncode:
            output is the output of the specified system call once the call has completed
                output[0] = stdout  output[1] = stderr
            returncode is the return code of the specified external call
    """
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    if isinstance(input_pipe, str):
        input_pipe = input_pipe.encode()

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                               startupinfo=_SUINFO)
    output = [x.decode() for x in process.communicate(input=input_pipe)]
    return output, process.returncode


def try_or_handle(func, handle=lambda x: x, *args, **kwargs):
    """
    try_or_handle attempts to call a function. If that function returns a value,
    that value is returned. Otherwise, the exception that halted the function is
    handled in some way.

    Parameters:
        func : a function to call
        handle : another function that will get passed any exceptions that occur
                    as a result of calling func
        *args, **kwargs : any number of arguments to pass to func

    Returns:
        the result of func(*args,**kwargs), or if func raises an exception, the
        result of passing the exception to handle
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)


def try_or_self(func, x, print_err=False):
    try:
        return func(x)
    except Exception as e:
        if print_err:
            print(e)
            print("Couldn't apply", func, "to", x, ", returning self.")
        return x


def whichclose(a, b, rtol=1e-05, atol=1e-08):
    """Determine the overlap between two sorted arrays

    whichclose finds boolean masks such that numpy.allclose(a[a_mask], b[b_mask] == True. The overlap may not be
    sequential in a given array.

    Parameters
    ----------
    a : ndarray, sorted
        a 1d array or arbirary length, sorted
    b : ndarray, sorted
        another 1d array of arbitrary length, sorted
    rtol : float
        see documentation for numpy.isclose(..)
    atol : float
        see documentation for numpy.isclose(..)

    Returns
    -------
    intersect_a : boolean ndarray
    intersect_b : boolean ndarray
        A tuple of boolean masks, (intersect_a, intersect_b) for the input arrays `a` and `b`, respectively, where True
        values represent an overlap in values within the provided tolerances. It is possible that the masks could be
        entirely False, which indicates that there was no overlap found.

    Raises
    ------
    ValueError
        If arrays `a` or `b` are not sorted, or if the difference between consecutive values in either array are within
        the floating point error tolerance given by `rtol` and `atol`.
    """
    # This method yields somewhat non-sensical results if values in `a` or `b` are closer together than the tolerance.
    # TODO: find a way to deal with this instead of raising an exception?
    a_diff = a[1:] - a[:-1]
    b_diff = b[1:] - b[:-1]
    if np.any(a_diff < (2 * a[-1] * rtol + atol)) or np.any(b_diff < (2 * b[-1] * rtol + atol)):
        raise ValueError("Differences between sucessive values is within floating point error tolerance")
    elif np.any(a_diff < 0.0) or np.any(b_diff < 0.0):
        raise ValueError("Input arrays must be sorted")

    # The problem is essentially to find the indices of values in `a` and `b` that form their intersection
    # Conveniently, we stipulate that `a` and `b` are sorted
    # `positions_right` are the indices in `a` where the items in `b` would fall
    # For each item in `b`, we need to test closeness to these indices and these indices - 1
    positions_right = np.searchsorted(a, b)
    positions_left = np.copy(positions_right)
    positions_left[positions_left > 0] -= 1
    # the other gotcha is those items in `b` where they are outside `a` entirely
    positions_right[positions_right == len(a)] -= 1
    # TODO: there is some minor speed gains that might be had here by truncating redundant checks?

    close_right = np.isclose(a[positions_right], b, rtol=rtol, atol=atol)
    close_left = np.isclose(a[positions_left], b, rtol=rtol, atol=atol)

    # the overlapping positions in `b` are the logical (x)or of close_left and close_right
    # xor lets us ignore cases where subsequent points in `a` are closer than the comparison tolerances
    intersect_b = np.logical_xor(close_left, close_right)

    # the overlapping positions in `a` are the indices in close_left/right
    intersect_a_idx = np.setxor1d(positions_right[close_right], positions_left[close_left])
    intersect_a = np.zeros(len(a), dtype=np.bool)
    intersect_a[intersect_a_idx] = True

    return intersect_a, intersect_b
