import os
import subprocess
import shlex
import numpy as np

# The following attempts to stop unecessary console window popups on Windows
try:
    _SUINFO = subprocess.STARTUPINFO()
    _SUINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
except:
    _SUINFO = None


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
    """
    whichclose is an analog of numpy's isclose(..) that accepts arrays of
        differing lengths. The returned boolean masks mark the first overlapping
        range of values such from each array for which numpy.allclose(..)
        returns True.

    Parameters:
        a: a 1d array of arbitrary length
        b: another 1d array of arbitrary length, regardless of a's length
        rtol: see documentation for numpy.isclose(..)
        atol: see documentation for numpy.isclose(..)

    Returns:
        A tuple of boolean masks, (a_mask,b_mask) for the input arrays a and b,
        respectively, where True values represent the first overlap in values.
    """
    # record lengths for fast access and make all False boolean masks
    len_a = len(a)
    a_mask = np.zeros(len_a, dtype=np.bool)
    len_b = len(b)
    b_mask = np.zeros(len_b, dtype=np.bool)

    # hunt for the overlap
    for i in range(1, len_a + len_b):
        if np.allclose(a[max(i - len_b, 0):min(i, len_a)], b[max(len_b - i, 0):min(len_b, len_b + len_a - i)],
                       rtol=rtol, atol=atol):
            # set boolean masks
            a_mask[max(i - len_b, 0):min(i, len_a)] = True
            b_mask[max(len_b - i, 0):min(len_b, len_b + len_a - i)] = True
            # cut out early to save needless comparisons
            return a_mask, b_mask
    # if we get to here there's no match, womp womp
    return a_mask, b_mask
