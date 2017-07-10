import os
import io
import collections
import tempfile
import warnings
import uuid
from contextlib import contextmanager
import numpy as np


def fix_filepath(path):
    return os.path.expandvars(os.path.expanduser(path))


def join_fix_filepath(fname, dirpath):
    dirpath = os.path.realpath(fix_filepath(dirpath))
    path = os.path.join(dirpath, fname)
    return path


def load_data_columns(path=None, file_handle=None, delimiter=None):
    if path is not None:
        file_handle = open(fix_filepath(path), 'rU')

    # Numpy doesn't provide a graceful way to separate bad from good lines if they are not known ahead of time.
    # Specifically, what if the number of header or footer lines are unknown? What if some lines have excess crap after
    # the data, but the data is still good?
    good_lines = {}
    columns_counts = collections.Counter()
    for line in file_handle:
        # blank lines are ok, just skip them
        line = line.strip()
        if len(line) == 0:
            continue
        tokens = line.split(delimiter)
        float_values = []
        count = 0
        for token in tokens:
            try:
                float_values.append(float(token))
                count += 1
            except ValueError:
                break
        if count > 0:
            columns_counts[count] += 1
            if count in good_lines:
                good_lines[count].append(float_values)
            else:
                good_lines[count] = [float_values]

    if path is not None:
        file_handle.close()

    ncols, nrows = columns_counts.most_common(1)[0]
    return np.array(good_lines[ncols]).transpose().copy()


def save_data_columns(path, cols, fmt="%f", delimiter="\t", newline="\n", header=""):
    if not isinstance(path, io.IOBase):
        path = fix_filepath(path)
        if os.path.exists(path):
            warnings.warn("Overwriting file {}".format(path))
    np.savetxt(path, np.transpose(cols), fmt=fmt, delimiter=delimiter, newline=newline, header=header)


@contextmanager
def open_tempfile(n=1):
    filedescs = []
    filenames = []
    try:
        for i in range(0, n):
            tmp = tempfile.mkstemp(text=True)
            filedescs.append(tmp[0])
            filenames.append(tmp[1])
        if n == 1:
            yield filedescs[0], filenames[0]
        else:
            yield filedescs, filenames
    finally:
        for fname in filenames:
            os.remove(fname)


def find_avail_tempfile_name(dir=''):
    dir = fix_filepath(dir)
    fullpath = os.path.join(dir, str(uuid.uuid4()))
    while os.path.exists(fullpath):
        fullpath = os.path.join(dir, str(uuid.uuid4()))
    return fullpath
