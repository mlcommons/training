''' Checks that an MLPerf log is compliant.

Compliance Checking is done in the following phases. We abort if there is an
error at any pharse.
'''

from __future__ import print_function

import argparse
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
    elif check_key == 'AT_LEAST_ONE':
        return check_it(mlp_common_checks.check_at_least_one_tag, loglines, check['AT_LEAST_ONE'])
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


    #if not configured_checks(loglines, benchmark_file):
    #    return False

    return True


def get_value(x):
  if isinstance(x, dict):
    return x.get("value")
  return x


def get_model_accuracy(loglines):
    eval_target = {get_value(i.value) for i in loglines if i.tag == 'eval_target'}
    if len(eval_target) == 1:
        eval_target = eval_target.pop()
    else:
        print("Failed to extract eval target.")
        eval_target = None

    values = [0]
    for i in loglines:
        if i.tag != 'eval_accuracy':
            continue
        try:
            values.append(i.value["value"])
        except:
            pass
    return max(values), eval_target


def l1_check_file(filename):
    loglines, errors = mlp_parser.parse_file(filename)

    if len(errors) > 0:
        print('Found parsing errors:')
        for line, error in errors:
            print(line)
            print('  ^^ ', error)
        print()
        print('FAILED: log lines had parsing errors.')
        print('FAILED: Logs are NOT L1 compliant.')
        return False, 0, 0, None

    check_ok = check_log(loglines)

    if not check_ok:
        print('FAILED: Logs are NOT L1 compliant.')
        return False, 0, 0, None

    dt = mlp_common_checks.check_clock(loglines)
    accuracy, eval_target = get_model_accuracy(loglines)

    print('SUCCESS: logs are L1 compliant.')
    return True, dt, accuracy, eval_target

def l1_check_file_w_starttime(filename):
    loglines, errors = mlp_parser.parse_file(filename)
    success, dt, accuracy, eval_target = l1_check_file(filename)
    # Get start time to order logs
    start = mlp_common_checks.find_tag(loglines, 'run_start', expect=1)[0]

    return start.timestamp, success, dt, accuracy, eval_target

def check_loglines_l2(loglines):
    l1_check_ok = check_log(loglines)

    general_l2_file = os.path.join(CONFIG_DIR, 'v0.5.0_level2.yaml')
    l2_check = configured_checks(loglines,  general_l2_file)

    if not loglines:
      print("No log lines detected.")
      return False

    benchmark = loglines[0].benchmark
    benchmark_file = os.path.join(CONFIG_DIR, 'v0.5.0_l2_{}.yaml'.format(benchmark))

    if not os.path.exists(benchmark_file):
      raise Exception('Could not find a compliance file for benchmark: ' + benchmark)

    specific_l2_check = configured_checks(loglines, benchmark_file)
    return l1_check_ok and l2_check and specific_l2_check


def l2_check_file(filename):
    loglines, errors = mlp_parser.parse_file(filename)

    if len(errors) > 0:
        print('Found parsing errors:')
        for line, error in errors:
            print(line)
            print('  ^^ ', error)
        print()
        print('FAILED: log lines had parsing errors.')
        print('FAILED: Logs are NOT L2 compliant.')
        return False, 0, 0, None

    check_ok = check_loglines_l2(loglines)

    if not check_ok:
        print('FAILED: Logs are NOT L2 compliant.')
        return False, 0, 0, None

    dt = mlp_common_checks.check_clock(loglines)
    accuracy, eval_target = get_model_accuracy(loglines)

    print('SUCCESS: logs are L2 compliant.')
    return True, dt, accuracy, eval_target

def l2_check_file_w_starttime(filename):
    loglines, errors = mlp_parser.parse_file(filename)
    success, dt, accuracy, eval_target = l2_check_file(filename)
    # Get start time to order logs
    start = mlp_common_checks.find_tag(loglines, 'run_start', expect=1)[0]

    return start.timestamp, success, dt, accuracy, eval_target

def main():
    parser = argparse.ArgumentParser(description='Lint MLPerf Compliance Logs.')

    parser.add_argument('filename', metavar='FILENAME', type=str,
                    help='the file to check for compliance.')
    parser.add_argument('-l', '--level', dest='level', type=int, metavar='LEVEL',
                    help='checks the logs coply at the given level.')

    args = parser.parse_args()
    print(args)

    lj_len = 20
    if args.level == 1:
      status, dt, qual, target = l1_check_file(args.filename)
      if status:
          print('Measured time: '.ljust(lj_len), dt)
          if qual:
              print('Best Eval Accuracy: '.ljust(lj_len), qual)
          if target:
              print('Target: '.ljust(lj_len), target)
          sys.exit(0)
      else:
          sys.exit(1)

    if args.level == 2:
      status, dt, qual, target = l2_check_file(args.filename)
      if status:
          print('Measured time: '.ljust(lj_len), dt)
          if qual:
              print('Best Eval Accuracy: '.ljust(lj_len), qual)
          if target:
              print('Target: '.ljust(lj_len), target)
          sys.exit(0)
      else:
          sys.exit(1)


if __name__ == '__main__':
    main()
