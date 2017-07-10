import warnings
import os
from .. import util as saxstoolsutil
from .. import fileio
from .. import pr

# check if datgnom exists
if saxstoolsutil.which('datgnom') is None:
    # raise ImportError('Error finding atsas datgnom in path')
    warnings.warn("Can't find datgnom in $PATH", ImportWarning)

_supportedargs = {'rg': float, 'skip': int, 'output': str, 'seed': int}


def datgnom(saxsdata, **kwargs):
    # build call to autorg and execute
    cmd = ['datgnom4']
    cmd.extend(saxstoolsutil.kwargs_to_cmd(saxstoolsutil.check_supported_kwargs(kwargs, _supportedargs)))
    cmd.append('-')

    if 'output' not in kwargs:
        cmd.append('--output')
        path = fileio.find_avail_tempfile_name()
        cmd.append(path)
        to_parse = path
    else:
        to_parse = kwargs['output']

    with fileio.open_tempfile() as (fd, path):
        cmd.append(path)
        with os.fdopen(fd) as f:
            f.write(str(saxsdata))
        output, returncode = saxstoolsutil.runexternal(cmd)

    # if successful...
    if returncode == 0:
        path = fileio.fix_filepath(to_parse)
        ret = pr.read_gnom_file(path)
        # remove temporary gnom output file and fix gross temp file name if possible
        if 'output' not in kwargs:
            os.remove(path)
            if hasattr(saxsdata, 'name') and len(saxsdata.name.strip()) > 0:
                ret.name = saxsdata.name
        return ret
    else:
        raise RuntimeError(output[1])
