'''
Parses a text MLPerf log into a structured format.
'''

from __future__ import print_function

import collections
import json
import re
import sys

# For example, 
# :::MLPv0.5.0 ncf 1538678136 (./convert.py:119) preproc_hp_num_eval: 999
# :::MLPv0.5.0 ncf 1538678136 (./convert.py:119) run_start

LogLine = collections.namedtuple('LogLine', [
    'full_string', # the complete line as a string
    'token', # the token, i.e. ':::MLP'
    'version_str', # the version string, e.g. 'v0.5.0'
    'benchmark', # the benchmark, e.g. 'ncf'
    'timestamp', # seconds as a float, e.g. 1234.567
    'filename', # the which generated the log line, e.g. './convert.py'
    'lineno', # the line in the file which generated the log line, e.g. 119
    'tag', # the string tag
    'value', # the parsed value associated with the tag, or None if no value
    ])


TOKEN = ':::MLP'

# ^.*
LINE_PATTERN = '''
^
(:::MLP)(v[\d]+\.[\d+]\.[\d+]) [ ] # token and version
([a-z]+) [ ] # benchmark
([\d\.]+) [ ] # timestamp
\(([^: ]+):(\d+)\) [ ] # file and lineno
([A-Za-z0-9_]+) [ ]? # tag
(:\s+(.+))? # optional value
$
'''

LINE_REGEX = re.compile(LINE_PATTERN, re.X)


def string_to_logline(string):
    ''' Returns a LogLine or raises a ValueError '''
    m = LINE_REGEX.match(string)

    if m is None:
        raise ValueError('does not match regex')

    args = []
    args.append('') # full string
    args.append(m.group(1)) # token
    args.append(m.group(2)) # version
    args.append(m.group(3)) # benchmark

    ts = float(m.group(4)) # may raise error, e.g. "1.2.3"
    # TODO check for weird values
    args.append(ts)

    
    args.append(m.group(5)) # file name

    lineno = int(m.group(6)) # may raise error
    # TODO check for weird values
    args.append(lineno)

    args.append(m.group(7)) # tag

    if m.group(9) is not None:
        j = json.loads(m.group(9))
        args.append(j)
    else:
        # No Value
        args.append(None)

    return LogLine(*args)


def parse_file(filename):
    ''' Reads a file by name and returns list of loglines and list of errors'''
    with open(filename) as f:
        return parse_generator(f)


def strip_and_dedup(gen):
  lines = []
  for l in gen:
    if ':::MLP' not in l:
      continue
    lines.append(re.sub(".*:::MLP", ":::MLP", l))
  return list(sorted(list(set(lines))))



def parse_generator(gen):
    ''' Reads a generator of lines and returns (loglines, errors)
    The list of errors are any parsing issues as a tuple (str_line, error_msg)
    '''
    loglines = []
    failed = []
    for line in strip_and_dedup(gen):
        line = line.strip()
        if TOKEN not in line:
            continue
        try:
            ll = string_to_logline(line)
            loglines.append(ll)
        except ValueError as e:
            failed.append((line, str(e)))
    return loglines, failed


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: mlp_parser.py FILENAME')
        print('       tests parsing on the file.')
        sys.exit(1)

    filename = sys.argv[1]
    lines, errors = parse_file(filename)

    print('Parsed {} log lines with {} errors.'.format(len(lines), len(errors)))

    if len(errors) > 0:
        print('Lines which failed to parse:')
        for line, error in errors:
            print('  Following line failed: {}'.format(error))
            print(line)

