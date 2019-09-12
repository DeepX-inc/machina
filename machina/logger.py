# The MIT License (MIT)
#
# Copyright (c) 2016 rllab contributors
#
# rllab uses a shared copyright model: each contributor holds copyright over
# their contributions to rllab. The project versioning records all such
# contribution and copyright details.
# By contributing to the rllab repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# This code is taken from rllab which is MIT-licensed


import collections
from collections import namedtuple
from contextlib import contextmanager
import csv
import datetime
from enum import Enum
import errno
import inspect
import os
import os.path as osp
from platform import python_version_tuple
import pydoc
import re
import shlex
import sys
import time
import warnings

import base64
from cached_property import cached_property
import dateutil.tz
import joblib
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import terminaltables


if python_version_tuple()[0] < "3":
    from itertools import izip_longest
    from functools import partial
    _none_type = type(None)
    _int_type = int
    _float_type = float
    _text_type = str
    _binary_type = str
else:
    from itertools import zip_longest as izip_longest
    from functools import reduce, partial
    _none_type = type(None)
    _int_type = int
    _float_type = float
    _text_type = str
    _binary_type = bytes


__all__ = ["tabulate", "tabulate_formats", "simple_separated_format"]


Line = namedtuple("Line", ["begin", "hline", "sep", "end"])


DataRow = namedtuple("DataRow", ["begin", "sep", "end"])


# A table structure is suppposed to be:
#
#     --- lineabove ---------
#         headerrow
#     --- linebelowheader ---
#         datarow
#     --- linebewteenrows ---
#     ... (more datarows) ...
#     --- linebewteenrows ---
#         last datarow
#     --- linebelow ---------
#
# TableFormat's line* elements can be
#
#   - either None, if the element is not used,
#   - or a Line tuple,
#   - or a function: [col_widths], [col_alignments] -> string.
#
# TableFormat's *row elements can be
#
#   - either None, if the element is not used,
#   - or a DataRow tuple,
#   - or a function: [cell_values], [col_widths], [col_alignments] -> string.
#
# padding (an integer) is the amount of white space around data values.
#
# with_header_hide:
#
#   - either None, to display all table elements unconditionally,
#   - or a list of elements not to be displayed if the table has column headers.
#
TableFormat = namedtuple("TableFormat", ["lineabove", "linebelowheader",
                                         "linebetweenrows", "linebelow",
                                         "headerrow", "datarow",
                                         "padding", "with_header_hide"])


def _pipe_segment_with_colons(align, colwidth):
    """Return a segment of a horizontal line with optional colons which
    indicate column's alignment (as in `pipe` output format)."""
    w = colwidth
    if align in ["right", "decimal"]:
        return ('-' * (w - 1)) + ":"
    elif align == "center":
        return ":" + ('-' * (w - 2)) + ":"
    elif align == "left":
        return ":" + ('-' * (w - 1))
    else:
        return '-' * w


def _pipe_line_with_colons(colwidths, colaligns):
    """Return a horizontal line with optional colons to indicate column's
    alignment (as in `pipe` output format)."""
    segments = [_pipe_segment_with_colons(a, w)
                for a, w in zip(colaligns, colwidths)]
    return "|" + "|".join(segments) + "|"


def _mediawiki_row_with_attrs(separator, cell_values, colwidths, colaligns):
    alignment = {"left":    '',
                 "right":   'align="right"| ',
                 "center":  'align="center"| ',
                 "decimal": 'align="right"| '}
    # hard-coded padding _around_ align attribute and value together
    # rather than padding parameter which affects only the value
    values_with_attrs = [' ' + alignment.get(a, '') + c + ' '
                         for c, a in zip(cell_values, colaligns)]
    colsep = separator*2
    return (separator + colsep.join(values_with_attrs)).rstrip()


def _latex_line_begin_tabular(colwidths, colaligns):
    alignment = {"left": "l", "right": "r", "center": "c", "decimal": "r"}
    tabular_columns_fmt = "".join([alignment.get(a, "l") for a in colaligns])
    return "\\begin{tabular}{" + tabular_columns_fmt + "}\n\hline"


_table_formats = {"simple":
                  TableFormat(lineabove=Line("", "-", "  ", ""),
                              linebelowheader=Line("", "-", "  ", ""),
                              linebetweenrows=None,
                              linebelow=Line("", "-", "  ", ""),
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0,
                              with_header_hide=["lineabove", "linebelow"]),
                  "plain":
                  TableFormat(lineabove=None, linebelowheader=None,
                              linebetweenrows=None, linebelow=None,
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0, with_header_hide=None),
                  "grid":
                  TableFormat(lineabove=Line("+", "-", "+", "+"),
                              linebelowheader=Line("+", "=", "+", "+"),
                              linebetweenrows=Line("+", "-", "+", "+"),
                              linebelow=Line("+", "-", "+", "+"),
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "pipe":
                  TableFormat(lineabove=_pipe_line_with_colons,
                              linebelowheader=_pipe_line_with_colons,
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1,
                              with_header_hide=["lineabove"]),
                  "orgtbl":
                  TableFormat(lineabove=None,
                              linebelowheader=Line("|", "-", "+", "|"),
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "rst":
                  TableFormat(lineabove=Line("", "=", "  ", ""),
                              linebelowheader=Line("", "=", "  ", ""),
                              linebetweenrows=None,
                              linebelow=Line("", "=", "  ", ""),
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0, with_header_hide=None),
                  "mediawiki":
                  TableFormat(lineabove=Line("{| class=\"wikitable\" style=\"text-align: left;\"",
                                             "", "", "\n|+ <!-- caption -->\n|-"),
                              linebelowheader=Line("|-", "", "", ""),
                              linebetweenrows=Line("|-", "", "", ""),
                              linebelow=Line("|}", "", "", ""),
                              headerrow=partial(
                                  _mediawiki_row_with_attrs, "!"),
                              datarow=partial(_mediawiki_row_with_attrs, "|"),
                              padding=0, with_header_hide=None),
                  "latex":
                  TableFormat(lineabove=_latex_line_begin_tabular,
                              linebelowheader=Line("\\hline", "", "", ""),
                              linebetweenrows=None,
                              linebelow=Line(
                                  "\\hline\n\\end{tabular}", "", "", ""),
                              headerrow=DataRow("", "&", "\\\\"),
                              datarow=DataRow("", "&", "\\\\"),
                              padding=1, with_header_hide=None),
                  "tsv":
                  TableFormat(lineabove=None, linebelowheader=None,
                              linebetweenrows=None, linebelow=None,
                              headerrow=DataRow("", "\t", ""),
                              datarow=DataRow("", "\t", ""),
                              padding=0, with_header_hide=None)}


tabulate_formats = list(sorted(_table_formats.keys()))


_invisible_codes = re.compile("\x1b\[\d*m")  # ANSI color codes
_invisible_codes_bytes = re.compile(b"\x1b\[\d*m")  # ANSI color codes


def simple_separated_format(separator):
    """Construct a simple TableFormat with columns separated by a separator.

    >>> tsv = simple_separated_format("\\t") ; \
        tabulate([["foo", 1], ["spam", 23]], tablefmt=tsv) == 'foo \\t 1\\nspam\\t23'
    True

    """
    return TableFormat(None, None, None, None,
                       headerrow=DataRow('', separator, ''),
                       datarow=DataRow('', separator, ''),
                       padding=0, with_header_hide=None)


def _isconvertible(conv, string):
    try:
        n = conv(string)
        return True
    except ValueError:
        return False


def _isnumber(string):
    """
    >>> _isnumber("123.45")
    True
    >>> _isnumber("123")
    True
    >>> _isnumber("spam")
    False
    """
    return _isconvertible(float, string)


def _isint(string):
    """
    >>> _isint("123")
    True
    >>> _isint("123.45")
    False
    """
    return type(string) is int or \
        (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
        _isconvertible(int, string)


def _type(string, has_invisible=True):
    """The least generic type (type(None), int, float, str, unicode).

    >>> _type(None) is type(None)
    True
    >>> _type("foo") is type("")
    True
    >>> _type("1") is type(1)
    True
    >>> _type('\x1b[31m42\x1b[0m') is type(42)
    True
    >>> _type('\x1b[31m42\x1b[0m') is type(42)
    True

    """

    if has_invisible and \
       (isinstance(string, _text_type) or isinstance(string, _binary_type)):
        string = _strip_invisible(string)

    if string is None:
        return _none_type
    elif hasattr(string, "isoformat"):  # datetime.datetime, date, and time
        return _text_type
    elif _isint(string):
        return int
    elif _isnumber(string):
        return float
    elif isinstance(string, _binary_type):
        return _binary_type
    else:
        return _text_type


def _afterpoint(string):
    """Symbols after a decimal point, -1 if the string lacks the decimal point.

    >>> _afterpoint("123.45")
    2
    >>> _afterpoint("1001")
    -1
    >>> _afterpoint("eggs")
    -1
    >>> _afterpoint("123e45")
    2

    """
    if _isnumber(string):
        if _isint(string):
            return -1
        else:
            pos = string.rfind(".")
            pos = string.lower().rfind("e") if pos < 0 else pos
            if pos >= 0:
                return len(string) - pos - 1
            else:
                return -1  # no point
    else:
        return -1  # not a number


def _padleft(width, s, has_invisible=True):
    """Flush right.

    >>> _padleft(6, '\u044f\u0439\u0446\u0430') == '  \u044f\u0439\u0446\u0430'
    True

    """
    iwidth = width + len(s) - len(_strip_invisible(s)
                                  ) if has_invisible else width
    fmt = "{0:>%ds}" % iwidth
    return fmt.format(s)


def _padright(width, s, has_invisible=True):
    """Flush left.

    >>> _padright(6, '\u044f\u0439\u0446\u0430') == '\u044f\u0439\u0446\u0430  '
    True

    """
    iwidth = width + len(s) - len(_strip_invisible(s)
                                  ) if has_invisible else width
    fmt = "{0:<%ds}" % iwidth
    return fmt.format(s)


def _padboth(width, s, has_invisible=True):
    """Center string.

    >>> _padboth(6, '\u044f\u0439\u0446\u0430') == ' \u044f\u0439\u0446\u0430 '
    True

    """
    iwidth = width + len(s) - len(_strip_invisible(s)
                                  ) if has_invisible else width
    fmt = "{0:^%ds}" % iwidth
    return fmt.format(s)


def _strip_invisible(s):
    "Remove invisible ANSI color codes."
    if isinstance(s, _text_type):
        return re.sub(_invisible_codes, "", s)
    else:  # a bytestring
        return re.sub(_invisible_codes_bytes, "", s)


def _visible_width(s):
    """Visible width of a printed string. ANSI color codes are removed.

    >>> _visible_width('\x1b[31mhello\x1b[0m'), _visible_width("world")
    (5, 5)

    """
    if isinstance(s, _text_type) or isinstance(s, _binary_type):
        return len(_strip_invisible(s))
    else:
        return len(_text_type(s))


def _align_column(strings, alignment, minwidth=0, has_invisible=True):
    """[string] -> [padded_string]

    >>> list(map(str,_align_column(["12.345", "-1234.5", "1.23", "1234.5", "1e+234", "1.0e234"], "decimal")))
    ['   12.345  ', '-1234.5    ', '    1.23   ', ' 1234.5    ', '    1e+234 ', '    1.0e234']

    >>> list(map(str,_align_column(['123.4', '56.7890'], None)))
    ['123.4', '56.7890']

    """
    if alignment == "right":
        strings = [s.strip() for s in strings]
        padfn = _padleft
    elif alignment == "center":
        strings = [s.strip() for s in strings]
        padfn = _padboth
    elif alignment == "decimal":
        decimals = [_afterpoint(s) for s in strings]
        maxdecimals = max(decimals)
        strings = [s + (maxdecimals - decs) * " "
                   for s, decs in zip(strings, decimals)]
        padfn = _padleft
    elif not alignment:
        return strings
    else:
        strings = [s.strip() for s in strings]
        padfn = _padright

    if has_invisible:
        width_fn = _visible_width
    else:
        width_fn = len

    maxwidth = max(max(list(map(width_fn, strings))), minwidth)
    padded_strings = [padfn(maxwidth, s, has_invisible) for s in strings]
    return padded_strings


def _more_generic(type1, type2):
    types = {_none_type: 0, int: 1, float: 2, _binary_type: 3, _text_type: 4}
    invtypes = {4: _text_type, 3: _binary_type,
                2: float, 1: int, 0: _none_type}
    moregeneric = max(types.get(type1, 4), types.get(type2, 4))
    return invtypes[moregeneric]


def _column_type(strings, has_invisible=True):
    """The least generic type all column values are convertible to.

    >>> _column_type(["1", "2"]) is _int_type
    True
    >>> _column_type(["1", "2.3"]) is _float_type
    True
    >>> _column_type(["1", "2.3", "four"]) is _text_type
    True
    >>> _column_type(["four", '\u043f\u044f\u0442\u044c']) is _text_type
    True
    >>> _column_type([None, "brux"]) is _text_type
    True
    >>> _column_type([1, 2, None]) is _int_type
    True
    >>> import datetime as dt
    >>> _column_type([dt.datetime(1991,2,19), dt.time(17,35)]) is _text_type
    True

    """
    types = [_type(s, has_invisible) for s in strings]
    return reduce(_more_generic, types, int)


def _format(val, valtype, floatfmt, missingval=""):
    """Format a value accoding to its type.

    Unicode is supported:

    >>> hrow = ['\u0431\u0443\u043a\u0432\u0430', '\u0446\u0438\u0444\u0440\u0430'] ; \
        tbl = [['\u0430\u0437', 2], ['\u0431\u0443\u043a\u0438', 4]] ; \
        good_result = '\\u0431\\u0443\\u043a\\u0432\\u0430      \\u0446\\u0438\\u0444\\u0440\\u0430\\n-------  -------\\n\\u0430\\u0437             2\\n\\u0431\\u0443\\u043a\\u0438           4' ; \
        tabulate(tbl, headers=hrow) == good_result
    True

    """
    if val is None:
        return missingval

    if valtype in [int, _text_type]:
        return "{0}".format(val)
    elif valtype is _binary_type:
        return _text_type(val, "ascii")
    elif valtype is float:
        return format(float(val), floatfmt)
    else:
        return "{0}".format(val)


def _align_header(header, alignment, width):
    if alignment == "left":
        return _padright(width, header)
    elif alignment == "center":
        return _padboth(width, header)
    elif not alignment:
        return "{0}".format(header)
    else:
        return _padleft(width, header)


def _normalize_tabular_data(tabular_data, headers):
    """Transform a supported data type to a list of lists, and a list of headers.

    Supported tabular data types:

    * list-of-lists or another iterable of iterables

    * list of named tuples (usually used with headers="keys")

    * 2D NumPy arrays

    * NumPy record arrays (usually used with headers="keys")

    * dict of iterables (usually used with headers="keys")

    * pandas.DataFrame (usually used with headers="keys")

    The first row can be used as headers if headers="firstrow",
    column indices can be used as headers if headers="keys".

    """

    if hasattr(tabular_data, "keys") and hasattr(tabular_data, "values"):
        # dict-like and pandas.DataFrame?
        if hasattr(tabular_data.values, "__call__"):
            # likely a conventional dict
            keys = list(tabular_data.keys())
            # columns have to be transposed
            rows = list(zip_longest(*list(tabular_data.values())))
        elif hasattr(tabular_data, "index"):
            # values is a property, has .index => it's likely a pandas.DataFrame (pandas 0.11.0)
            keys = list(tabular_data.keys())
            vals = tabular_data.values  # values matrix doesn't need to be transposed
            names = tabular_data.index
            rows = [[v]+list(row) for v, row in zip(names, vals)]
        else:
            raise ValueError(
                "tabular data doesn't appear to be a dict or a DataFrame")

        if headers == "keys":
            headers = list(map(_text_type, keys))  # headers should be strings

    else:  # it's a usual an iterable of iterables, or a NumPy array
        rows = list(tabular_data)

        if (headers == "keys" and
            hasattr(tabular_data, "dtype") and
                getattr(tabular_data.dtype, "names")):
            # numpy record array
            headers = tabular_data.dtype.names
        elif (headers == "keys"
              and len(rows) > 0
              and isinstance(rows[0], tuple)
              and hasattr(rows[0], "_fields")):  # namedtuple
            headers = list(map(_text_type, rows[0]._fields))
        elif headers == "keys" and len(rows) > 0:  # keys are column indices
            headers = list(map(_text_type, list(range(len(rows[0])))))

    # take headers from the first row if necessary
    if headers == "firstrow" and len(rows) > 0:
        headers = list(map(_text_type, rows[0]))  # headers should be strings
        rows = rows[1:]

    headers = list(headers)
    rows = list(map(list, rows))

    # pad with empty headers for initial columns if necessary
    if headers and len(rows) > 0:
        nhs = len(headers)
        ncols = len(rows[0])
        if nhs < ncols:
            headers = [""]*(ncols - nhs) + headers

    return rows, headers


def tabulate(tabular_data, headers=[], tablefmt="simple",
             floatfmt="g", numalign="decimal", stralign="left",
             missingval=""):
    """Format a fixed width table for pretty printing.

    >>> print(tabulate([[1, 2.34], [-56, "8.999"], ["2", "10001"]]))
    ---  ---------
      1      2.34
    -56      8.999
      2  10001
    ---  ---------

    The first required argument (`tabular_data`) can be a
    list-of-lists (or another iterable of iterables), a list of named
    tuples, a dictionary of iterables, a two-dimensional NumPy array,
    NumPy record array, or a Pandas' dataframe.


    Table headers
    -------------

    To print nice column headers, supply the second argument (`headers`):

      - `headers` can be an explicit list of column headers
      - if `headers="firstrow"`, then the first row of data is used
      - if `headers="keys"`, then dictionary keys or column indices are used

    Otherwise a headerless table is produced.

    If the number of headers is less than the number of columns, they
    are supposed to be names of the last columns. This is consistent
    with the plain-text format of R and Pandas' dataframes.

    >>> print(tabulate([["sex","age"],["Alice","F",24],["Bob","M",19]],
    ...       headers="firstrow"))
           sex      age
    -----  -----  -----
    Alice  F         24
    Bob    M         19


    Column alignment
    ----------------

    `tabulate` tries to detect column types automatically, and aligns
    the values properly. By default it aligns decimal points of the
    numbers (or flushes integer numbers to the right), and flushes
    everything else to the left. Possible column alignments
    (`numalign`, `stralign`) are: "right", "center", "left", "decimal"
    (only for `numalign`), and None (to disable alignment).


    Table formats
    -------------

    `floatfmt` is a format specification used for columns which
    contain numeric data with a decimal point.

    `None` values are replaced with a `missingval` string:

    >>> print(tabulate([["spam", 1, None],
    ...                 ["eggs", 42, 3.14],
    ...                 ["other", None, 2.7]], missingval="?"))
    -----  --  ----
    spam    1  ?
    eggs   42  3.14
    other   ?  2.7
    -----  --  ----

    Various plain-text table formats (`tablefmt`) are supported:
    'plain', 'simple', 'grid', 'pipe', 'orgtbl', 'rst', 'mediawiki',
    and 'latex'. Variable `tabulate_formats` contains the list of
    currently supported formats.

    "plain" format doesn't use any pseudographics to draw tables,
    it separates columns with a double space:

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
    ...                 ["strings", "numbers"], "plain"))
    strings      numbers
    spam         41.9999
    eggs        451

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="plain"))
    spam   41.9999
    eggs  451

    "simple" format is like Pandoc simple_tables:

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
    ...                 ["strings", "numbers"], "simple"))
    strings      numbers
    ---------  ---------
    spam         41.9999
    eggs        451

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="simple"))
    ----  --------
    spam   41.9999
    eggs  451
    ----  --------

    "grid" is similar to tables produced by Emacs table.el package or
    Pandoc grid_tables:

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
    ...                ["strings", "numbers"], "grid"))
    +-----------+-----------+
    | strings   |   numbers |
    +===========+===========+
    | spam      |   41.9999 |
    +-----------+-----------+
    | eggs      |  451      |
    +-----------+-----------+

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="grid"))
    +------+----------+
    | spam |  41.9999 |
    +------+----------+
    | eggs | 451      |
    +------+----------+

    "pipe" is like tables in PHP Markdown Extra extension or Pandoc
    pipe_tables:

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
    ...                ["strings", "numbers"], "pipe"))
    | strings   |   numbers |
    |:----------|----------:|
    | spam      |   41.9999 |
    | eggs      |  451      |

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="pipe"))
    |:-----|---------:|
    | spam |  41.9999 |
    | eggs | 451      |

    "orgtbl" is like tables in Emacs org-mode and orgtbl-mode. They
    are slightly different from "pipe" format by not using colons to
    define column alignment, and using a "+" sign to indicate line
    intersections:

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
    ...                ["strings", "numbers"], "orgtbl"))
    | strings   |   numbers |
    |-----------+-----------|
    | spam      |   41.9999 |
    | eggs      |  451      |


    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="orgtbl"))
    | spam |  41.9999 |
    | eggs | 451      |

    "rst" is like a simple table format from reStructuredText; please
    note that reStructuredText accepts also "grid" tables:

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]],
    ...                ["strings", "numbers"], "rst"))
    =========  =========
    strings      numbers
    =========  =========
    spam         41.9999
    eggs        451
    =========  =========

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="rst"))
    ====  ========
    spam   41.9999
    eggs  451
    ====  ========

    "mediawiki" produces a table markup used in Wikipedia and on other
    MediaWiki-based sites:

    >>> print(tabulate([["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]],
    ...                headers="firstrow", tablefmt="mediawiki"))
    {| class="wikitable" style="text-align: left;"
    |+ <!-- caption -->
    |-
    ! strings   !! align="right"|   numbers
    |-
    | spam      || align="right"|   41.9999
    |-
    | eggs      || align="right"|  451
    |}

    "latex" produces a tabular environment of LaTeX document markup:

    >>> print(tabulate([["spam", 41.9999], ["eggs", "451.0"]], tablefmt="latex"))
    \\begin{tabular}{lr}
    \\hline
     spam &  41.9999 \\\\
     eggs & 451      \\\\
    \\hline
    \\end{tabular}

    """

    list_of_lists, headers = _normalize_tabular_data(tabular_data, headers)

    # optimization: look for ANSI control codes once,
    # enable smart width functions only if a control code is found
    plain_text = '\n'.join(['\t'.join(map(_text_type, headers))] +
                           ['\t'.join(map(_text_type, row)) for row in list_of_lists])
    has_invisible = re.search(_invisible_codes, plain_text)
    if has_invisible:
        width_fn = _visible_width
    else:
        width_fn = len

    # format rows and columns, convert numeric values to strings
    cols = list(zip(*list_of_lists))
    coltypes = list(map(_column_type, cols))
    cols = [[_format(v, ct, floatfmt, missingval) for v in c]
            for c, ct in zip(cols, coltypes)]

    # align columns
    aligns = [numalign if ct in [int, float] else stralign for ct in coltypes]
    minwidths = [width_fn(h)+2 for h in headers] if headers else [0]*len(cols)
    cols = [_align_column(c, a, minw, has_invisible)
            for c, a, minw in zip(cols, aligns, minwidths)]

    if headers:
        # align headers and add headers
        minwidths = [max(minw, width_fn(c[0]))
                     for minw, c in zip(minwidths, cols)]
        headers = [_align_header(h, a, minw)
                   for h, a, minw in zip(headers, aligns, minwidths)]
        rows = list(zip(*cols))
    else:
        minwidths = [width_fn(c[0]) for c in cols]
        rows = list(zip(*cols))

    if not isinstance(tablefmt, TableFormat):
        tablefmt = _table_formats.get(tablefmt, _table_formats["simple"])

    return _format_table(tablefmt, headers, rows, minwidths, aligns)


def _build_simple_row(padded_cells, rowfmt):
    "Format row according to DataRow format without padding."
    begin, sep, end = rowfmt
    return (begin + sep.join(padded_cells) + end).rstrip()


def _build_row(padded_cells, colwidths, colaligns, rowfmt):
    "Return a string which represents a row of data cells."
    if not rowfmt:
        return None
    if hasattr(rowfmt, "__call__"):
        return rowfmt(padded_cells, colwidths, colaligns)
    else:
        return _build_simple_row(padded_cells, rowfmt)


def _build_line(colwidths, colaligns, linefmt):
    "Return a string which represents a horizontal line."
    if not linefmt:
        return None
    if hasattr(linefmt, "__call__"):
        return linefmt(colwidths, colaligns)
    else:
        begin, fill, sep,  end = linefmt
        cells = [fill*w for w in colwidths]
        return _build_simple_row(cells, (begin, sep, end))


def _pad_row(cells, padding):
    if cells:
        pad = " "*padding
        padded_cells = [pad + cell + pad for cell in cells]
        return padded_cells
    else:
        return cells


def _format_table(fmt, headers, rows, colwidths, colaligns):
    """Produce a plain-text representation of the table."""
    lines = []
    hidden = fmt.with_header_hide if (headers and fmt.with_header_hide) else []
    pad = fmt.padding
    headerrow = fmt.headerrow

    padded_widths = [(w + 2*pad) for w in colwidths]
    padded_headers = _pad_row(headers, pad)
    padded_rows = [_pad_row(row, pad) for row in rows]

    if fmt.lineabove and "lineabove" not in hidden:
        lines.append(_build_line(padded_widths, colaligns, fmt.lineabove))

    if padded_headers:
        lines.append(_build_row(padded_headers,
                                padded_widths, colaligns, headerrow))
        if fmt.linebelowheader and "linebelowheader" not in hidden:
            lines.append(_build_line(
                padded_widths, colaligns, fmt.linebelowheader))

    if padded_rows and fmt.linebetweenrows and "linebetweenrows" not in hidden:
        # initial rows with a line below
        for row in padded_rows[:-1]:
            lines.append(_build_row(
                row, padded_widths, colaligns, fmt.datarow))
            lines.append(_build_line(
                padded_widths, colaligns, fmt.linebetweenrows))
        # the last row without a line below
        lines.append(_build_row(
            padded_rows[-1], padded_widths, colaligns, fmt.datarow))
    else:
        for row in padded_rows:
            lines.append(_build_row(
                row, padded_widths, colaligns, fmt.datarow))

    if fmt.linebelow and "linebelow" not in hidden:
        lines.append(_build_line(padded_widths, colaligns, fmt.linebelow))

    return "\n".join(lines)


try:
    import matplotlib  # NOQA

    matplotlib.use('Agg')
    _available = True

except (ImportError, TypeError):
    _available = False


@cached_property
def _check_available():
    if not _available:
        warnings.warn('matplotlib is not installed on your environment, '
                      'so nothing will be plotted at this time. '
                      'Please install matplotlib to plot figures.\n\n'
                      '  $ pip install matplotlib\n')
    return _available


def plot_scores(filename, key, x_key, scores=None, label=None, title=None,
                color=None, result_directory=None,
                plot_legend=True, xlim=None, ylim=None,
                x_label='steps', y_label=None):
    if _check_available:
        import matplotlib.pyplot as plt
    else:
        return

    if label is None:
        label = key

    f = plt.figure()
    a = f.add_subplot(111)
    if scores is None:
        scores = pd.read_csv(filename)
    x = scores[x_key]
    mean = scores[key + 'Average']
    std = scores[key + 'Std']
    a.plot(x, mean, label=label)
    a.fill_between(x,
                   mean - std,
                   mean + std,
                   color=color,
                   alpha=0.2)

    a.set_xlabel(x_label)
    if y_label is not None:
        a.set_ylabel(y_label)
    if plot_legend is True:
        a.legend(loc='best')
    if title is not None:
        a.set_title(title)
    if xlim is not None:
        a.set_xlim(xlim)
    if ylim is not None:
        a.set_ylim(ylim)

    if result_directory is not None:
        fig_fname = os.path.join(result_directory,
                                 "{}.png".format(label))
    else:
        fig_fname = os.path.join(os.path.dirname(filename),
                                 "{}.png".format(label))
    f.savefig(fig_fname)
    plt.close()
    return fig_fname


def csv2table(filename, save_dir=None, output_filename='scores-table.txt'):
    with open(filename, newline='') as f:
        data = list(csv.reader(f))
        table = terminaltables.AsciiTable(data)
        table.inner_row_border = False
        table.CHAR_H_INNER_HORIZONTAL = '-'
        table.CHAR_OUTER_TOP_LEFT = ''
        table.CHAR_OUTER_TOP_RIGHT = ''
        table.CHAR_OUTER_TOP_HORIZONTAL = ''
        table.CHAR_OUTER_TOP_INTERSECT = ''
        table.CHAR_OUTER_BOTTOM_LEFT = '|'
        table.CHAR_OUTER_BOTTOM_RIGHT = '|'
        table.CHAR_OUTER_BOTTOM_HORIZONTAL = ' '
        table.CHAR_OUTER_BOTTOM_INTERSECT = '|'
        table.CHAR_H_OUTER_LEFT_INTERSECT = '|'
        table.CHAR_H_OUTER_RIGHT_INTERSECT = '|'
        output = table.table.lstrip().rstrip()

    if save_dir is None:
        save_dir = os.path.dirname(filename)
    with open(os.path.join(save_dir, output_filename), 'w') as f:
        f.write(output)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def log(s):  # , send_telegram=False):
    print(s)
    sys.stdout.flush()


class SimpleMessage(object):

    def __init__(self, msg, logger=log):
        self.msg = msg
        self.logger = logger

    def __enter__(self):
        print(self.msg)
        self.tstart = time.time()

    def __exit__(self, etype, *args):
        maybe_exc = "" if etype is None else " (with exception)"
        self.logger("done%s in %.3f seconds" %
                    (maybe_exc, time.time() - self.tstart))


MESSAGE_DEPTH = 0


class Message(object):

    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        global MESSAGE_DEPTH  # pylint: disable=W0603
        print(colorize('\t' * MESSAGE_DEPTH + '=: ' + self.msg, 'magenta'))
        self.tstart = time.time()
        MESSAGE_DEPTH += 1

    def __exit__(self, etype, *args):
        global MESSAGE_DEPTH  # pylint: disable=W0603
        MESSAGE_DEPTH -= 1
        maybe_exc = "" if etype is None else " (with exception)"
        print(colorize('\t' * MESSAGE_DEPTH + "done%s in %.3f seconds" %
                       (maybe_exc, time.time() - self.tstart), 'magenta'))


def prefix_log(prefix, logger=log):
    return lambda s: logger(prefix + s)


def tee_log(file_name):
    f = open(file_name, 'w+')

    def logger(s):
        log(s)
        f.write(s)
        f.write('\n')
        f.flush()
    return logger


def collect_args():
    splitted = shlex.split(' '.join(sys.argv[1:]))
    return {arg_name[2:]: arg_val
            for arg_name, arg_val in zip(splitted[::2], splitted[1::2])}


def type_hint(arg_name, arg_type):
    def wrap(f):
        meta = getattr(f, '__tweak_type_hint_meta__', None)
        if meta is None:
            f.__tweak_type_hint_meta__ = meta = {}
        meta[arg_name] = arg_type
        return f
    return wrap


def tweak(fun_or_val, identifier=None):
    if isinstance(fun_or_val, collections.Callable):
        return tweakfun(fun_or_val, identifier)
    return tweakval(fun_or_val, identifier)


def tweakval(val, identifier):
    if not identifier:
        raise ValueError('Must provide an identifier for tweakval to work')
    args = collect_args()
    for k, v in args.items():
        stripped = k.replace('-', '_')
        if stripped == identifier:
            log('replacing %s in %s with %s' % (stripped, str(val), str(v)))
            return type(val)(v)
    return val


def tweakfun(fun, alt=None):
    """Make the arguments (or the function itself) tweakable from command line.
    See tests/test_misc_console.py for examples.

    NOTE: this only works for the initial launched process, since other processes
    will get different argv. What this means is that tweak() calls wrapped in a function
    to be invoked in a child process might not behave properly.
    """
    cls = getattr(fun, 'im_class', None)
    method_name = fun.__name__
    if alt:
        cmd_prefix = alt
    elif cls:
        cmd_prefix = cls + '.' + method_name
    else:
        cmd_prefix = method_name
    cmd_prefix = cmd_prefix.lower()
    args = collect_args()
    if cmd_prefix in args:
        fun = pydoc.locate(args[cmd_prefix])
    if type(fun) == type:
        argspec = inspect.getargspec(fun.__init__)
    else:
        argspec = inspect.getargspec(fun)
    # TODO handle list arguments
    defaults = dict(
        list(zip(argspec.args[-len(argspec.defaults or []):], argspec.defaults or [])))
    replaced_kwargs = {}
    cmd_prefix += '-'
    if type(fun) == type:
        meta = getattr(fun.__init__, '__tweak_type_hint_meta__', {})
    else:
        meta = getattr(fun, '__tweak_type_hint_meta__', {})
    for k, v in args.items():
        if k.startswith(cmd_prefix):
            stripped = k[len(cmd_prefix):].replace('-', '_')
            if stripped in meta:
                log('replacing %s in %s with %s' %
                    (stripped, str(fun), str(v)))
                replaced_kwargs[stripped] = meta[stripped](v)
            elif stripped not in argspec.args:
                raise ValueError(
                    '%s is not an explicit parameter of %s' % (stripped, str(fun)))
            elif stripped not in defaults:
                raise ValueError(
                    '%s does not have a default value in method %s' % (stripped, str(fun)))
            elif defaults[stripped] is None:
                raise ValueError(
                    'Cannot infer type of %s in method %s from None value' % (stripped, str(fun)))
            else:
                log('replacing %s in %s with %s' %
                    (stripped, str(fun), str(v)))
                # TODO more proper conversions
                replaced_kwargs[stripped] = type(defaults[stripped])(v)

    def tweaked(*args, **kwargs):
        all_kw = dict(list(zip(argspec[0], args)) +
                      list(kwargs.items()) + list(replaced_kwargs.items()))
        return fun(**all_kw)
    return tweaked


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


_prefixes = []
_prefix_str = ''

_tabular_prefixes = []
_tabular_prefix_str = ''

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = 'all'
_snapshot_gap = 1

_log_tabular_only = False
_header_printed = False

_running_processes = []
_async_plot_flag = False

_summary_writer = None
_global_step = 0


def _add_output(file_name, arr, fds, mode='a'):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, mode, newline='')


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode='a')


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    _add_output(file_name, _tabular_outputs, _tabular_fds, mode='w')


def add_tensorboard_output(logdir):
    try:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
    except ImportError:
        print("tensorboard is not available")
        return
    global _summary_writer
    if _summary_writer is not None:
        print("Currently only support one SummaryWriter")
        return
    _summary_writer = SummaryWriter(logdir)


def write_to_tensorboard(tabular_dict):
    if _summary_writer is None:
        return
    global _global_step
    stat_keys = set()
    normal_keys = []
    for k in tabular_dict.keys():
        for j in ["Average", "Std", "Median", "Min", "Max"]:
            idx = k.find(j)
            if idx != -1:
                break
        if idx != -1:
            stat_keys.add(k[:idx])
        else:
            normal_keys.append(k)
    for k in normal_keys:
        _summary_writer.add_scalar(
            "data/" + k, float(tabular_dict[k]), _global_step)
    for k in stat_keys:
        _summary_writer.add_scalars("stat/" + k,
                                    {k + "Max": float(tabular_dict[k + "Max"]),
                                     k + "Median": float(tabular_dict[k + "Median"]),
                                     k + "Min": float(tabular_dict[k + "Min"])}, _global_step)
    _global_step += 1


def remove_tabular_output(file_name):
    if _tabular_fds[file_name] in _tabular_header_written:
        _tabular_header_written.remove(_tabular_fds[file_name])
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def set_snapshot_dir(dir_name):
    global _snapshot_dir
    _snapshot_dir = dir_name


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode


def get_snapshot_gap():
    return _snapshot_gap


def set_snapshot_gap(gap):
    global _snapshot_gap
    _snapshot_gap = gap


def set_log_tabular_only(log_tabular_only):
    global _log_tabular_only
    _log_tabular_only = log_tabular_only


def get_log_tabular_only():
    return _log_tabular_only


def log(s, with_prefix=True, with_timestamp=True, color=None):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
        out = "%s | %s" % (timestamp, out)
    if color is not None:
        out = colorize(out, color)
    if not _log_tabular_only:
        # Also log to stdout
        print(out)
        for fd in list(_text_fds.values()):
            fd.write(out + '\n')
            fd.flush()
        sys.stdout.flush()


def record_tabular(key, val):
    _tabular.append((_tabular_prefix_str + str(key), str(val)))


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = ''.join(_tabular_prefixes)


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()


def dump_tabular(*args, **kwargs):
    wh = kwargs.pop("write_header", None)
    if len(_tabular) > 0:
        no_print = kwargs.pop('no_print', None)
        if no_print:
            pass
        else:
            if _log_tabular_only:
                table_printer.print_tabular(_tabular)
            else:
                for line in tabulate(_tabular).split('\n'):
                    log(line, *args, **kwargs)
        tabular_dict = dict(_tabular)
        # Also write to the csv files
        # This assumes that the keys in each iteration won't change!
        for tabular_fd in list(_tabular_fds.values()):
            writer = csv.DictWriter(
                tabular_fd, fieldnames=list(tabular_dict.keys()))
            if wh or (wh is None and tabular_fd not in _tabular_header_written):
                writer.writeheader()
                _tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()
        write_to_tensorboard(tabular_dict)
        del _tabular[:]


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = ''.join(_prefixes)


def save_itr_params(itr, params):
    if _snapshot_dir:
        if _snapshot_mode == 'all':
            file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'last':
            # override previous params
            file_name = osp.join(_snapshot_dir, 'params.pkl')
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "gap":
            if itr % _snapshot_gap == 0:
                file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
                joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == 'none':
            pass
        else:
            raise NotImplementedError


def log_parameters(log_file, args, classes):
    log_params = {}
    for param_name, param_value in args.__dict__.items():
        if any([param_name.startswith(x) for x in list(classes.keys())]):
            continue
        log_params[param_name] = param_value
    for name, cls in classes.items():
        if isinstance(cls, type):
            params = get_all_parameters(cls, args)
            params["_name"] = getattr(args, name)
            log_params[name] = params
        else:
            log_params[name] = getattr(cls, "__kwargs", dict())
            log_params[name]["_name"] = cls.__module__ + \
                "." + cls.__class__.__name__
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True)


def stub_to_json(stub_sth):
    from rllab.misc import instrument
    if isinstance(stub_sth, instrument.StubObject):
        assert len(stub_sth.args) == 0
        data = dict()
        for k, v in stub_sth.kwargs.items():
            data[k] = stub_to_json(v)
        data["_name"] = stub_sth.proxy_class.__module__ + \
            "." + stub_sth.proxy_class.__name__
        return data
    elif isinstance(stub_sth, instrument.StubAttr):
        return dict(
            obj=stub_to_json(stub_sth.obj),
            attr=stub_to_json(stub_sth.attr_name)
        )
    elif isinstance(stub_sth, instrument.StubMethodCall):
        return dict(
            obj=stub_to_json(stub_sth.obj),
            method_name=stub_to_json(stub_sth.method_name),
            args=stub_to_json(stub_sth.args),
            kwargs=stub_to_json(stub_sth.kwargs),
        )
    elif isinstance(stub_sth, instrument.BinaryOp):
        return "binary_op"
    elif isinstance(stub_sth, instrument.StubClass):
        return stub_sth.proxy_class.__module__ + "." + stub_sth.proxy_class.__name__
    elif isinstance(stub_sth, dict):
        return {stub_to_json(k): stub_to_json(v) for k, v in stub_sth.items()}
    elif isinstance(stub_sth, (list, tuple)):
        return list(map(stub_to_json, stub_sth))
    elif type(stub_sth) == type(lambda: None):
        if stub_sth.__module__ is not None:
            return stub_sth.__module__ + "." + stub_sth.__name__
        return stub_sth.__name__
    elif "theano" in str(type(stub_sth)):
        return repr(stub_sth)
    return stub_sth


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {'$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name}
        return json.JSONEncoder.default(self, o)


def log_parameters_lite(log_file, args):
    log_params = {}
    for param_name, param_value in args.__dict__.items():
        log_params[param_name] = param_value
    # if args.args_data is not None:
    #    stub_method = pickle.loads(base64.b64decode(args.args_data))
    #    method_args = stub_method.kwargs
    #    log_params["json_args"] = dict()
    #    for k, v in list(method_args.items()):
    #        log_params["json_args"][k] = stub_to_json(v)
    #    kwargs = stub_method.obj.kwargs
    #    for k in ["baseline", "env", "policy"]:
    #        if k in kwargs:
    #            log_params["json_args"][k] = stub_to_json(kwargs.pop(k))
    #    log_params["json_args"]["algo"] = stub_to_json(stub_method.obj)
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(log_params, f, indent=2, sort_keys=True, cls=MyEncoder)


def log_variant(log_file, variant_data):
    mkdir_p(os.path.dirname(log_file))
    if hasattr(variant_data, "dump"):
        variant_data = variant_data.dump()
    variant_json = stub_to_json(variant_data)
    with open(log_file, "w") as f:
        json.dump(variant_json, f, indent=2, sort_keys=True, cls=MyEncoder)


def record_tabular_misc_stat(key, values):
    record_tabular(key + "Average", np.average(values))
    record_tabular(key + "Std", np.std(values))
    record_tabular(key + "Median", np.median(values))
    record_tabular(key + "Min", np.amin(values))
    record_tabular(key + "Max", np.amax(values))


def record_results(log_dir, result_dict, score_file,
                   total_epi,
                   step, total_step,
                   rewards=None,
                   plot_title=None,
                   async_plot=True,
                   ):
    log("outdir {}".format(os.path.abspath(log_dir)))

    for key, value in result_dict.items():
        if not hasattr(value, '__len__'):
            record_tabular(key, value)
        elif len(value) >= 1:
            record_tabular_misc_stat(key, value)
    if rewards is not None:
        record_tabular_misc_stat('Reward', rewards)
        record_tabular('EpisodePerIter', len(rewards))
    record_tabular('TotalEpisode', total_epi)
    record_tabular('StepPerIter', step)
    record_tabular('TotalStep', total_step)
    dump_tabular()

    if not async_plot:
        for key, value in result_dict.items():
            if hasattr(value, '__len__'):
                fig_fname = plot_scores(score_file, key, 'TotalStep',
                                        title=plot_title)
                log('Saved a figure as {}'.format(os.path.abspath(fig_fname)))
        if rewards is not None:
            fig_fname = plot_scores(score_file, 'Reward', 'TotalStep',
                                    title=plot_title)
            log('Saved a figure as {}'.format(os.path.abspath(fig_fname)))
    else:
        global _async_plot_flag
        if not _async_plot_flag:
            global plot_process
            plot_process = mp.Pool(processes=1)
            _async_plot_flag = True
        if _running_processes:
            p = _running_processes[0]
            if p.ready():
                del _running_processes[:]
        if not _running_processes:
            p = plot_process.apply_async(func=async_plot_scores, args=(
                score_file, plot_title, result_dict, rewards))
            _running_processes.append(p)


def record_results_bc(log_dir, result_dict, score_file,
                      epoch,
                      rewards=None,
                      plot_title=None,
                      async_plot=True,
                      ):
    log("outdir {}".format(os.path.abspath(log_dir)))

    for key, value in result_dict.items():
        if not hasattr(value, '__len__'):
            record_tabular(key, value)
        elif len(value) >= 1:
            record_tabular_misc_stat(key, value)
    if rewards is not None:
        record_tabular_misc_stat('Reward', rewards)
    record_tabular('CurrentEpoch', epoch)
    dump_tabular()

    if not async_plot:
        for key, value in result_dict.items():
            if hasattr(value, '__len__'):
                fig_fname = plot_scores(score_file, key, 'CurrentEpoch',
                                        title=plot_title, x_label='epoch')
                log('Saved a figure as {}'.format(os.path.abspath(fig_fname)))
        if rewards is not None:
            fig_fname = plot_scores(score_file, 'Reward', 'CurrentEpoch',
                                    title=plot_title, x_label='epoch')
            log('Saved a figure as {}'.format(os.path.abspath(fig_fname)))
    else:
        global _async_plot_flag
        if not _async_plot_flag:
            global plot_process
            plot_process = mp.Pool(processes=1)
            _async_plot_flag = True
        if _running_processes:
            p = _running_processes[0]
            if p.ready():
                del _running_processes[:]
        if not _running_processes:
            p = plot_process.apply_async(func=async_plot_scores, args=(
                score_file, plot_title, result_dict, rewards, 'CurrentEpoch', 'epoch'))
            _running_processes.append(p)


def async_plot_scores(filename, title, result_dict, rewards, x_key='TotalStep', x_label='steps'):
    scores = pd.read_csv(filename)
    for key, value in result_dict.items():
        if hasattr(value, '__len__'):
            fig_fname = plot_scores(filename, key, x_key,
                                    title=title, scores=scores, x_label=x_label)
    if rewards is not None:
        fig_fname = plot_scores(filename, 'Reward', x_key,
                                title=title, scores=scores, x_label=x_label)
