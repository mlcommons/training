from .ruleset_060 import parse_file as parse_file_060
from .ruleset_070 import parse_file as parse_file_070

def parse_file(filename, ruleset='0.6.0'):
    if ruleset == '0.6.0':
        return parse_file_060(filename)
    elif ruleset == '0.7.0':
        return parse_file_070(filename)
    else:
        raise Exception(f'Ruleset "{ruleset}" is not supported')
