import os
import warnings
import numpy as np
from .. import util as saxstoolsutil
from .. import fileio
from .. import saxs

# check if almerge exists
if saxstoolsutil.which('almerge') is None:
    warnings.warn("Can't find almerge in $PATH",ImportWarning)

_supportedargs = {'overlap': int, 'step': int, 'zeroconc': None, 'output': str}

_output_fields = ['concentration', 'scale', 'first', 'last']


def parse_almerge_stdout(output):
    # break into lines
    output = output.strip().split("\n")

    # pop the header line
    output.pop(0)

    # do the parsing
    curves = {}
    for line in output:
        tokens = line.strip().split()
        if tokens[1] == '(reference)':
            file = tokens[2]
            values = [float(tokens[0]), None, None, None]
        else:
            file = tokens[6]
            values = [float(tokens[0]), float(tokens[2]), int(tokens[3]), int(tokens[5])]
        curves[file] = dict(zip(_output_fields, values))

    return curves


def almerge_merge(saxsdata, concentrations, all=False, **kwargs):
    """
    Use the almerge program to merge data from multiple concentrations together.

    3 modes are available:
    (1) Using only the highest and lowest curves that can be overlapped, splice them together with the overlapping
    region averaged.
    (2) The above, but include all provided SAXS curves in the average (where they each overlap)
    (3) #2, but also extrapolate the low-q region to infinite dilution (concentration zero).

    For (1), provide two or more SAXS curves and leaves `all` as False and do not specify `zeroconc`
    for (2), do as above but specifiy `all=True`. Note that the first SAXS curve in the last will act as the reference.
    Generally, this should be the highest concentration sample.
    for (3), Specify `zeroconc=True` (`all` is irrelevant)
    """
    if 'zeroconc' in kwargs:
        mode = 3
    elif all:
        mode = 2
    else:
        mode = 1

    # Setup call to almerge
    cmd = ['almerge']

    # The output of almerge will be exactly what we want if mode 1 or 3
    # In mode 2, we actually have to suppress almerge writing an output
    if mode == 1 or mode == 3:
        if 'output' not in kwargs:
            cmd.append('--output')
            path = fileio.find_avail_tempfile_name()
            cmd.append(path)
            to_parse = path
        else:
            to_parse = kwargs['output']
    else:
        to_parse = kwargs.pop('output', None)

    cmd.extend(saxstoolsutil.kwargs_to_cmd(saxstoolsutil.check_supported_kwargs(kwargs, _supportedargs)))

    # Write out all samples to temporary files and pair them with their concentrations in the command
    with fileio.open_tempfile(n=len(saxsdata)) as (fds,fnames):
        for (data, fd, path, concentration) in zip(saxsdata, fds, fnames, concentrations):
            with os.fdopen(fd, 'w') as f:
                f.write(str(data))
            cmd.extend(['-c', str(concentration), path])
        # call almerge
        output, returncode = saxstoolsutil.runexternal(cmd)

    # if successful...
    if returncode == 0:
        # get stdout for returning the scaling factors/overlaps
        stdout = parse_almerge_stdout(output[0])
        scale_list = [stdout[x] for x in fnames]
        # if mode 1 or 3, the output from almerge is the correct answer;
        # in mode 2 we have to do our own magic because almerge won't do it out of the box
        if mode == 1 or mode == 3:
            merged_curve = saxs.read_saxs_file(to_parse)
        else:
            # extract the overlaps as a list of tuples
            ranges = [(stdout[fname]['first']-1, stdout[fname]['last'])
                      if stdout[fname]['first'] is not None else (0, len(x))
                      for x, fname in zip(saxsdata, fnames)]
            # the lowest concentration curve needs to be used for the low-q data
            ranges[np.argmin(concentrations)][0] = 0
            # the highest concentration curve needs to be used for the high-q data
            high_idx = np.argmax(concentrations)
            ranges[high_idx][1] = len(saxsdata[high_idx])
            # scale all the data
            scaled_data = [x * stdout[fname]['scale'] for x, fname in zip(saxsdata, fnames)]
            merged_curve = saxs.subrange_average(scaled_data, ranges)

        return scale_list, merged_curve
    else:
        raise RuntimeError(output[1])

def almerge_scale(curves,reference=0,return_output=False,**kwargs):
    """
    Use the (very powerful) curve-aligning feature of almerge to scale different
    SAXS datasets of the same sample but different concentrations so that they
    overlay.

    Returns a list of the scaled results, in the same order as the originals.

    Parameters
    ----------
    curves : list-like
        A group of Saxscurves to overlay. Unless otherwise specified, the first
        Saxscurve is used as the reference.
    reference : int, optional
        Specify and integer index such that `curves[reference]` is used as the
        reference data for almerge.
    return_output : bool, optional
        If True, return the results of `almerge.parse_output` from the call to
        almerge.

    Returns
    -------
    scaled : list
        The Saxscurves from `curves` scaled by almerge.
    almerge_results : list, optional
        The result of `almerge.parse_output` from the almerge run. Instead of a
        dictionary, this is a list of tuples in the same order as `curves`.
    """
    with fileio.open_tempfile(n=len(curves)) as (fds,fnames):
        for (curve,fd) in zip(curves,fds):
            with os.fdopen(fd,'w') as tmpfile:
                curve.save(fobj=tmpfile)

        cmd=['almerge']
        cmd+=saxstoolsutil.kwargs_to_cmd(kwargs,_supportedargs)
        if reference > 0:
            fnames_cpy=fnames[:]
            refcurve=fnames_cpy.pop(reference)
            fnames_cpy.insert(0,refcurve)
            cmd+=fnames_cpy
        else:
            cmd+=fnames
        output,returncode=saxstoolsutil.runexternal(cmd)

    if returncode == 0:
        result=parse_output(output[0])
        scaled=[curve*(result[fnames[i]][1]) for (i,curve) in enumerate(curves)]
        if return_output:
            results_fixed=[result[fname] for fname in fnames]
            return scaled,results_fixed
        else:
            return scaled
    else:
        print output[1]
        warnings.warn('Call to almerge failed',RuntimeWarning)
        return None
