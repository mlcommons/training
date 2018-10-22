''' Checks that an MLPerf log is compliant.

Compliance Checking is done in the following phases. We abort if there is an
error at any pharse.
'''

from __future__ import print_function

import os
import sys
import yaml

import mlp_common_checks
import mlp_parser


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__)) + '/configs/'

def common_checks(loglines):
    ''' Preforms common checks which all benchmark logs should pass. '''
    try:
        mlp_common_checks.check_clock(loglines)
    except mlp_common_checks.CCError as e:
        print(e)
        return False
    return True


def do_check(loglines, check):
    check_key = list(check.keys())[0] 

    def check_it(call, *args):
        try:
            call(*args)
            return True
        except mlp_common_checks.CCError as e:
            print('FAIL: ', e)
            return False

    if check_key == 'EXACTLY_ONE':
        return check_it(mlp_common_checks.check_exactly_one_tag, loglines, check['EXACTLY_ONE'])
    elif check_key == 'TAG_EVAL_CHECK':
        d = check['TAG_EVAL_CHECK']
        return check_it(mlp_common_checks.check_eval_tag, loglines, d['NAME'], d['TAG'], d['CODE'], d['EXPECT'])
    elif check_key == 'TAGS_PAIR':
        d = check['TAGS_PAIR']
        return check_it(mlp_common_checks.check_tags_pair, loglines, d['FIRST'], d['SECOND'])
    elif check_key == 'TAGS_COUNT_SAME':
        d = check['TAGS_COUNT_SAME']
        return check_it(mlp_common_checks.check_tags_count_same, loglines, d)
    else:
        raise Exception('Unknown check: ', check)


def configured_checks(loglines, config_file):
    with open(config_file) as f:
        checks = yaml.load(f)

    l = []
    for check in checks:
        l.append(do_check(loglines, check))
    return False not in l



def check_log(loglines):
    #if not common_checks(loglines):
    #    return False

    if not configured_checks(loglines, CONFIG_DIR + '/v0.5.0_level1.yaml'):
        return False

    #benchmark = loglines[0].benchmark
    #benchmark_file = 'configs/v0.5.0_{}.yaml'.format(benchmark)

    #if not configured_checks(loglines, benchmark_file):
    #    return False

    return True


def main():
    filename = sys.argv[1]

    loglines, errors = mlp_parser.parse_file(filename)

    if len(errors) > 0:
        print('Found parsing errors:')
        for line, error in errors:
            print(line)
            print('  ^^ ', error)
        print()
        print('FAILED: log lines had parsing errors.')
        sys.exit(1)

    status = check_log(loglines)

    if not status:
        print('FAILED: compliance errors.')
        sys.exit(1)
    else:
        print('SUCCESS: logs are compliant.')
        sys.exit(0)


if __name__ == '__main__':
    main()
