'''
Runs the set of rules defined in provided config YAML file.
'''

from __future__ import print_function

import argparse
import os
import sys
import yaml

import mlp_parser



class CCError(Exception): 
    pass

def run_check_eval(ll, tag, code, state):
    if code is None: return

    try:
        result = eval(code.strip(), state, {'ll': ll, 'v': ll.value})
    except:
        raise CCError('Failed executing CHECK code triggered by line :\n{}'.format(ll.full_string))

    if not result:
        raise CCError('CHECK failed in line \n \'{}\' for \'{}\',\n v={},\n s={},\n code \'{} \''.format(ll.full_string, tag, ll.value, state['s'], code))


def run_check_exec(ll, tag, code, state, action):
    if code is None: return

    try:
        exec(code.strip(), state, {'ll': ll, 'v': ll.value})
    except:
        raise CCError('Failed executing code {} code triggered by line :\n{}'.format(action, ll.full_string))


enqueued_configs = []

# this function can be called from yaml
def enqueue_config(config):
    enqueued_configs.append(config)

def configured_checks(loglines, config_file):
    with open(config_file) as f:
        checks = yaml.load(f, Loader=yaml.BaseLoader)

    if checks is None:
        return

    s = {}  # this would be visible from inside configs
    state = {'enqueue_config':enqueue_config , 's':s}

    #execute begin block
    begin_blocks = [x for x in checks if list(x)[0]=='BEGIN']
    assert(len(begin_blocks)<=1) # up to one begin block
    if len(begin_blocks)==1:
        exec(begin_blocks[0]['BEGIN']['CODE'].strip(), state)

    key_records = {}
    for k in checks:
        if list(k)[0]=='KEY':
            key_records.update({k['KEY']['NAME']:k['KEY']}) 

    occurrence_counter = {k:0 for k in key_records.keys()}

    # executing the rules through log records
    for line in loglines:
        key_record = None
        try:
            occurrence_counter[line.key] = occurrence_counter[line.key]+1
            key_record = key_records[line.key]
        except:
            # unknown key - it's allowed, skip to next record
            continue

        if 'PRE' in key_record: run_check_exec(line, line.key, key_record['PRE'], state, 'PRE')
        if 'CHECK' in key_record: run_check_eval(line, line.key, key_record['CHECK'], state)
        if 'POST' in key_record: run_check_exec(line, line.key, key_record['POST'], state, 'POST')

    # verify occurrences requirements
    for k,v in key_records.items():
        if 'REQ' not in v: continue
        if v['REQ']=='EXACTLY_ONE':
            if occurrence_counter[k]!=1:
                 raise CCError("Required EXACTLY_ONE occurrence of \'{}\' but found {}".format(k, occurrence_counter[k]))

        if v['REQ']=='AT_LEAST_ONE':
            if occurrence_counter[k]<1:
                 raise CCError("Required AT_LEAST_ONE occurrence of \'{}\' but found {}".format(k, occurrence_counter[k]))

    # execute end block
    end_blocks = [x for x in checks if list(x)[0]=='END']
    assert(len(end_blocks)<=1) # up to one end block
    if len(end_blocks)==1:
        end_record = end_blocks[0]['END']
        if 'PRE' in end_record: exec(end_record['PRE'].strip(), state)
        if 'CHECK' in end_record:
            end_result = eval(end_record['CHECK'].strip(), state)
            if not end_result:
                raise CCError('Failed executing END CHECK with \n s={},\n code \'{} \''.format(state, end_record['CHECK'].strip()))


def check_loglines(loglines, config):
    if not loglines:
      raise CCError('No log lines detected')

    enqueue_config(config)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    while len(enqueued_configs)>0:
        current_config = enqueued_configs.pop(0)
        config_file = general_file = os.path.join(current_dir, current_config)

        if not os.path.exists(config_file):
            raise CCError('Could not find config file: {}'.format(config_file))

        # processing a config may have a side affect of pushing another config(s) to be checked
        configured_checks(loglines,  config_file)


def check_file(args):

    loglines, errors = mlp_parser.parse_file(args.filename)

    if len(errors) > 0:
        print('Found parsing errors:')
        for line, error in errors:
            print(line)
            print('  ^^ ', error)
        print()
        raise CCError('Log lines had parsing errors.')

    check_loglines(loglines, args.config)


def main():
    parser = argparse.ArgumentParser(description='Lint MLPerf Compliance Logs.')

    parser.add_argument('filename', type=str,
                    help='the file to check for compliance')
    parser.add_argument('--config',  type=str,
                    help='mlperf logging config', default='0.6.0/common.yaml')

    args = parser.parse_args()

    try:
        check_file(args)
        print('SUCCESS')
    except CCError as e:
        print('FAIL: ', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
