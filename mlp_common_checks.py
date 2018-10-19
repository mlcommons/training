'''
Common checks to perform on all MLP Logs.


Convention is to raise Common Compliance Exception (CCError) if errors are found.
'''

import re


class CCError(Exception): 
    pass


def find_tag(loglines, tag, expect=None):
    l = []
    for ll in loglines:
        if ll.tag == tag:
            l.append(ll)

    if expect is not None and len(l) != expect:
            raise CCError('Expected exactly {} copy of tag \'{}\''.format(expect, tag))
    return l
 

def check_clock(loglines):
     ''' Ensures start and stop exist and behave sanely '''
     start = find_tag(loglines, 'run_start', expect=1)[0]
     end = find_tag(loglines, 'run_stop', expect=1)[0]

     delta_t = end.timestamp - start.timestamp

     # Sanity check the runtine isn't off by a sign or orders of magnitude
     if delta_t <= 0:
         raise CCError('Runtime is less than or equal to zero. (time between run_start and run_stop)')
     if delta_t > 60 * 60 * 24 * 365:
         raise CCError('Runtime (time between run_start and run_stop) exceeds one year... assuming this is wrong.')


def check_exactly_one_tag(loglines, tag):
    find_tag(loglines, tag, expect=1)


def _check_eval(ll, name, tag, code):
    try:
        return eval(code.strip(), {}, {'ll': ll, 'v': ll.value})
    except:
        raise CCError('Failed evaluating {} on:\n{}'.format(name, ll.full_string))


def check_eval_tag(loglines, name, tag, code, expect):
    count = 0
    total = 0
    for ll in loglines:
        if re.match(tag, ll.tag):
            total += 1
            ok = _check_eval(ll, name, tag, code)
            if not ok and expect == 'ALL':
                raise CCError('Check {} failed on\n{}'.format(name, ll.full_string))
            if ok:
                count += 1
    if expect == 'AT_LEAST_ONE':
        if count == 0:
            raise CCError('Check {} failed.'.format(name))


def check_tags_pair(loglines, first, second):
    expect_first = True

    for ll in loglines:
        if ll.tag == first:
            if not expect_first:
                raise CCError('Expected a "{}" tag before:\n{}'.format(second, ll.full_string))
            expect_first = False
        if ll.tag == second:
            if expect_first:
                raise CCError('Expected a "{}" tag before:\n{}'.format(first, ll.full_string))
            expect_first = True
    if not expect_first:
        raise CCError('Expected a "{}" tag before end.'.format(second, ll.full_string))


def check_tags_count_same(loglines, tags):
    counts_per_tag = {}

    for t in tags:
        counts_per_tag[t] = 0
    for ll in loglines:
        for t in tags:
            if ll.tag == t:
                counts_per_tag[t] += 1

    if len(set(counts_per_tag.values())) != 1:
        raise CCError('Expected the following tags to have the same count: {}'.format(counts_per_tag))





