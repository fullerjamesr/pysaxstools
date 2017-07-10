import os
import re
import warnings
from functools import reduce
from .. import util as saxstoolsutil
from ..saxs import Saxscurve
from .. import fileio


# check if autorg exists
if saxstoolsutil.which('autorg') is None:
    # raise ImportError('Error finding atsas autorg in path')
    warnings.warn("Can't find autorg in $PATH", ImportWarning)

# autorg command line argument configuration
_supportedargs = {'mininterval': int, 'minrg': float, 'smaxrg': float, 'sminrg': float, 'output': str, 'format': str}
_supportedargvalues = {'format': ['ssv', 'csv']}

# autorg output formats
_csv_output_fields = ['filename', 'rg', 'rg stdev', 'I(0)', 'I(0) stdev', 'first', 'last', 'quality',
                      'aggregated']
_csv_output_types = [str, float, float, float, float, int, int, float, float]

_ssv_output_fields = ['rg', 'rg stdev', 'I(0)', 'I(0) stdev', 'first', 'last', 'quality', 'aggregated',
                      'filename']
_ssv_output_types = [float, float, float, float, int, int, float, float, str]


def parse_output(output, fmt=None):
    """
    Parse the results of a call to autorg.
    """
    # break output into lines
    output = re.split("\n", output.strip())

    # determine format and set up parsing
    if fmt is None:
        if output[0].strip().count(',') == len(_csv_output_fields) - 1:
            fmt = 'csv'
        elif output[0].strip().count(' ') >= len(_ssv_output_fields) - 1:
            fmt = 'ssv'
        else:
            raise ValueError("Formatting of output could not be determined and was not provided")

    if fmt == 'csv':
        sep = ','
        funcs = _csv_output_types
        keys = _csv_output_fields
    elif fmt == 'ssv':
        sep = '\s+'
        funcs = _ssv_output_types
        keys = _ssv_output_fields
    else:
        raise ValueError("Specified value for 'fmt' must be one of {!s}".format(_supportedargvalues['fmt']))

    # CSV output has a header line
    if fmt == 'csv' and output[0].strip().lower().startswith('file'):
        output.pop(0)

    # do the parsing
    results = {}
    for line in output:
        tokens = re.split(sep, line.strip())
        values = dict(zip(keys, map(lambda x, y: x(y), funcs, tokens)))
        results[values.pop('filename')] = values

    return results


def autorg(saxsdata, names=None, **kwargs):
    """
    Call autorg from the atsas package on a single SAXS dataset.

    Patameters:
        saxsdata: An object whose string representation is a 3 column formated
                    SAXS data. Any file formatting recognized by ATSAS is fine.
                    This gets fed via stdin to autorg
        **kwargs: Any command line args to pass to autorg (e.g, sminrg=1.0)

    Returns:
        The parsed stdout of the autorg command as a dictionary of values
            e.g., {'rg' => ... 'I(0)' => ...}
        If the -o /--output command line args were used, the resulting file is
        opened for parsing.
    """
    # by default use the ssv notation which is most straightforward to parse
    if 'format' not in kwargs:
        kwargs['format'] = 'csv'

    # build call to autorg and execute
    cmd = ['autorg']
    cmd.extend(
        saxstoolsutil.kwargs_to_cmd(
            saxstoolsutil.check_supported_kwargs(kwargs, _supportedargs, _supportedargvalues)))

    if type(saxsdata) == Saxscurve:
        cmd.append('-')
        output, returncode = saxstoolsutil.runexternal(cmd, input_pipe=str(saxsdata).encode())
    elif isinstance(saxsdata, list):
        with fileio.open_tempfile(len(saxsdata)) as (tmp_file_handles, tmp_file_paths):
            cmd.extend(tmp_file_paths)
            for handle, data in zip(tmp_file_handles, saxsdata):
                with os.fdopen(handle, 'w') as f:
                    f.write(str(data))
            output, returncode = saxstoolsutil.runexternal(cmd)
            # Optional: replace results keys to be something more sensible than temporary file names
            if names is not None:
                replacements = zip(tmp_file_paths, names)
            else:
                replacements = [p for p in zip(tmp_file_paths, [x.name for x in saxsdata])
                                if p[1] is not None and p[1].strip() != '']
            output = reduce(lambda x, y: x.replace(y[0], y[1]), replacements, output)
    else:
        raise TypeError("expected Saxscurve or list of Saxscurve for argument saxsdata")

    # if call to autorg was successful...
    if returncode == 0:
        # the relevant output was either printed to the indicated file, or stdout
        if 'output' in kwargs:
            with open(kwargs['output'], 'rU') as outfile:
                to_parse = outfile.readlines()
        else:
            to_parse = output[0]

        results = parse_output(to_parse, fmt=kwargs['format'])
        if type(saxsdata) == Saxscurve:
            return results['-']
        else:
            return results
    else:
        raise RuntimeError(output[1])
