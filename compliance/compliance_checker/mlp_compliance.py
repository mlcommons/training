'''
Runs the set of rules defined in provided config YAML file.
'''

from __future__ import print_function

import argparse
import os
import sys
import yaml
import json

import mlp_parser



class CCError(Exception): 
    pass


def preety_dict(d):
    return json.dumps(d, sort_keys=True, indent=2)


enqueued_configs = []

# this function can be called from yaml
def enqueue_config(config):
    enqueued_configs.append(config)


class ComplianceChecker:

    def __init__(self, ruleset):
        self.ruleset = ruleset

        self.overwritable = {}
        self.not_overwritable = []

    def raise_exception(self, msg):
        raise CCError(msg)


    def put_message(self, msg, key=None):
        if key:
            self.overwritable[key] = msg
        else:
            self.not_overwritable.append(msg)


    def overwrite_messages(self, keys):
        for key in keys:
            self.overwritable.pop(key, None)


    def log_messages(self):
        message_separator = '-' * 30
        for key in self.overwritable:
            print(message_separator)
            print(self.overwritable[key])

        for msg in self.not_overwritable:
            print(message_separator)
            print(msg)


    def has_messages(self):
        return self.not_overwritable or self.overwritable


    def run_check_eval(self, ll, tag, tests, state):
        if type(tests) is not list:
            tests = [tests]

        try:
            failed_test = ''

            for test in tests:
                if not eval(test.strip(), state, {'ll': ll, 'v': ll.value }):
                    failed_test = test
                    break
        except:
            self.put_message(f'Failed executing CHECK code triggered by line :\n{ll.full_string}',
                             key=ll.key)
            return

        if failed_test:
            self.put_message(
                f"CHECK for '{tag}' failed in line {ll.lineno}:"
                f"\n{ll.full_string}"
                f"\nfailed test: {failed_test}"
                f"\ncurrent context[s]={preety_dict(state['s'])}"
                f"\ncurrent line[v]={preety_dict(ll.value)}",
                key=ll.key)


    def run_check_exec(self, ll, tag, code, state, action):
        if code is None: return

        try:
            exec(code.strip(), state, {'ll': ll, 'v': ll.value})
        except:
            self.put_message(f'Failed executing code {action} code triggered by line :\n{ll.full_string}',
                             key=ll.key)


    def parse_alternatives(self, string):
        in_pharentises = string[len('AT_LEAST_ONE_OR(') : -1]
        alternatives = in_pharentises.split(',')
        return [s.strip() for s in alternatives]

    def configured_checks(self, loglines, config_file):
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

        # if config overrides some rules from previous config, corresponding messages are not needed
        self.overwrite_messages(key_records)

        # executing the rules through log records
        for line in loglines:
            key_record = None
            try:
                occurrence_counter[line.key] = occurrence_counter[line.key]+1
                key_record = key_records[line.key]
            except:
                # unknown key - it's allowed, skip to next record
                continue

            if 'PRE' in key_record: self.run_check_exec(line, line.key, key_record['PRE'], state, 'PRE')
            if 'CHECK' in key_record: self.run_check_eval(line, line.key, key_record['CHECK'], state)
            if 'POST' in key_record: self.run_check_exec(line, line.key, key_record['POST'], state, 'POST')

        alternatives = set()
        # verify occurrences requirements
        for k,v in key_records.items():
            if 'REQ' not in v:
                continue

            if v['REQ']=='EXACTLY_ONE':
                if occurrence_counter[k]!=1:
                     self.put_message(f"Required EXACTLY_ONE occurrence of '{k}' but found {occurrence_counter[k]}",
                                      key=k)

            if v['REQ']=='AT_LEAST_ONE':
                if occurrence_counter[k]<1:
                     self.put_message(f"Required AT_LEAST_ONE occurrence of '{k}' but found {occurrence_counter[k]}",
                                      key=k)

            if v['REQ'].startswith('AT_LEAST_ONE_OR'):
                alternatives.add(tuple({k, *self.parse_alternatives(v['REQ'])}))

        for alts in alternatives:
            if not any(occurrence_counter[k] for k in alts):
                self.put_message("Required AT_LEAST_ONE occurrence of {}".format(' or '.join(f"'{s}'" for s in alts)))

        # execute end block
        end_blocks = [x for x in checks if list(x)[0]=='END']
        assert(len(end_blocks)<=1) # up to one end block
        if len(end_blocks)==1:
            end_record = end_blocks[0]['END']
            if 'PRE' in end_record: exec(end_record['PRE'].strip(), state)
            if 'CHECK' in end_record:
                end_result = eval(end_record['CHECK'].strip(), state)
                if not end_result:
                    self.put_message('Failed executing END CHECK with \n s={},\n code \'{} \''.format(state, end_record['CHECK'].strip()))


    def check_loglines(self, loglines, config):
        if not loglines:
          self.put_message('No log lines detected')

        enqueue_config(config)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        while len(enqueued_configs)>0:
            current_config = enqueued_configs.pop(0)
            config_file = general_file = os.path.join(current_dir, current_config)

            if not os.path.exists(config_file):
                self.put_message('Could not find config file: {}'.format(config_file))

            # processing a config may have a side affect of pushing another config(s) to be checked
            self.configured_checks(loglines,  config_file)


    def check_file(self, args):

        loglines, errors = mlp_parser.parse_file(args.filename, ruleset=self.ruleset)

        if len(errors) > 0:
            print('Found parsing errors:')
            for line, error in errors:
                print(line)
                print('  ^^ ', error)
            print()
            self.put_message('Log lines had parsing errors.')

        self.check_loglines(loglines, args.config)

        self.log_messages()

        return not self.has_messages()

def get_parser():
    parser = argparse.ArgumentParser(description='Lint MLPerf Compliance Logs.')

    parser.add_argument('filename', type=str,
                    help='the file to check for compliance')
    parser.add_argument('--ruleset', type=str, default='0.7.0',
                    help='what version of rules to check the log against')
    parser.add_argument('--config',  type=str,
                    help='mlperf logging config, by default it loads {ruleset}/common.yaml', default=None)

    return parser


def fill_defaults(args):
    if not args.config:
        args.config = f'{args.ruleset}/common.yaml'

    return args


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = fill_defaults(args)

    checker = ComplianceChecker(args.ruleset)

    if checker.check_file(args):
        print('SUCCESS')
    else:
        print('FAIL')
        sys.exit(1)

if __name__ == '__main__':
    main()
