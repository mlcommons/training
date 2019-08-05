# MLPerf training logging compliance checker

### Requirements
The checker works with both python2 and python3, requires PyYaml package.
[See exact versions tested](#tested-software-versions)

### Usage

To check a log file for compliance:

    python mlp_compliance.py [--config YAML] FILENAME

Default config is set to `0.6.0/common.yaml`. This config will check all common keys and enqueue benchmark specific config to be checked as well.

Note that `minigo` has a specialized config, therefore to check minigo please use

    python mlp_compliance.py --config 0.6.0/minigo.yaml FILENAME 


Prints `SUCCESS` when no issues were found. Otherwise will print error details.
Fails on the first error.

As log examples use [NVIDIA's v0.6 training logs](https://github.com/mlperf/training_results_v0.6/tree/master/NVIDIA/results).

### Existing config files

    0.6.0/common.yaml        - currently the default config file, checks common fields complience(excluding minigo) and equeues benchmark-specific config file
    0.6.0/score.yaml         - printing out the score (in sec) from a log of any benchmark - example of how this infrastructure can be used
    0.6.0/resnet.yaml        
    0.6.0/ssd.yaml
    0.6.0/minigo.yaml
    0.6.0/maskrcnn.yaml
    0.6.0/gnmt.yaml
    0.6.0/transformer.yaml

### Implementation details
Compliance checking is done following below algorithm. It will be aborted in case of an error at any phase.

1. Parser converts the log into a list of records, each record corresponds to MLL 
   line and contains all relevant extracted information
2. Set of rules to be checked in loaded from provided config yaml file
3. Process optional `BEGIN` rule if present by executing provided `CODE` section
4. Loop through the records of the log
   1. If the key in the record is defined in rules process the rule:
      1. If present, execute `PRE` section
      2. If present, evaluate `CHECK` section, and raise an exception if the result is false
      3. If present, execute `POST` section
   2. Increment occurrences counter
5. Fail if any occurrences requirements (`AT_LEAST_ONE`/`EXACTLY_ONE`) were violated
6. Process optional `END` rule if present:
   1. If present, execute `PRE`
   2. If present, evaluate `CHECK` section, and raise an exception if the result is false

Possible side effects of yaml sections execution can be [printing output](#other-operations), or [enqueueing 
additional yaml files to be verified](#enqueuing-additional-config-files).

### Config file syntax
Rules to be checked are provided in yaml (config) file. A config file contains the following records:

#### `BEGIN` record
Defines `CODE` to be executed before any other rules defined in the current file. This record is optional
and there can be up to a single `BEGIN` record per config file. 

Example:

    - BEGIN:
        CODE: " s.update({'run_start':None}) "


#### `KEY` record
Defines the actions to be triggered while processing a specific `KEY`. The name of the `KEY` is specified in field `NAME`.

The following fields are optional:
- `REQ` - specifies the requirement regarding occurrence. Possible values : `AT_LEAST_ONE` or `EXACTLY_ONE`
- `PRE` - code to be executed before performing checks
- `CHECK` - expression to be evaluated as part of checking this key. False result would mean a failure.
- `POST` - code to be executed after performing checks

Example:

    - KEY:
        NAME:  epoch_start
        REQ:   AT_LEAST_ONE
        CHECK: " s['run_started'] and not s['in_epoch'] and ( v['metadata']['epoch_num'] == (s['last_epoch']+1) ) and not s['run_stopped']"
        POST:  " s['in_epoch'] = True; s['last_epoch'] = v['metadata']['epoch_num'] "


#### `END` record
Specifies actions to be taken after processing all the lines in log file. This record is optional and
there can be up to a single `END` record per config file.

The following fields are optional:
- `PRE` - code to be executed before performing checks
- `CHECK` - expression to be evaluated as part of checking this key. False result would mean a failure.

#### Global and local state access

During processing of the records there is a global state `s` maintained, accessible from 
code provided in yaml. In addition, rules can access the information fields (values) `v`
of the record, as well as timestamp and the original line string as part of the record `ll`.

Global state `s` can be used to enforce any cross keys rules, by updating the global state 
in `POST` (or `PRE`) of one `KEY` and using that information for `CHECK` of another `KEY`.
For each config file, `s` starts as an empty dictionary, so in order to track global state 
it would require adding an entry to `s`. 

Example:

    - BEGIN:
        CODE: " s.update({'run_start':None}) "

`ll` is a structure representing current log line that triggered `KEY` record. `ll` has the following fields
that can be accessed:
- `full_string` - the complete line as a string
- `timestamp` - seconds as a float, e.g. 1234.567
- `key` - the string key
- `value` - the parsed value associated with the key, or None if no value

`v` is a shortcut for `ll.value`

Example:

    - KEY:
        NAME:  run_stop
        CHECK: " ( v['metadata']['status'] == 'success' )"
        POST:  " print('score [sec]:' , ll.timestamp - s['run_start']) "



#### Enqueuing additional config files

To enqueue additional rule config files to be verified use `enqueue_config(YAML)` function.
Config files in the queue are processed independently, meaning that they do not share state or any rules.
A failure in any rule will result in an immediate termination and remaining configs will not be processed.

Each config file may define it's `BEGIN` and `END` records, as well as any other `KEY` rules.

Example: 

    - KEY:
        NAME:  submission_benchmark
        REQ:   EXACTLY_ONE
        CHECK: " v['value'] in ['resnet', 'ssd', 'maskrcnn', 'transformer', 'gnmt'] "
        POST:  " enqueue_config('0.6.0/{}.yaml'.format(v['value'])) "


#### Other operations

`CODE`, `REQ`, and `POST` fields are executed using python's `exec` function. `CHECK` is performed
using `eval` call. As such, any legal python code would be suitable for use. 

For instance, can define rules that would print out information as shown in the [example above](#global-and-local-state-access).


### Tested software versions
Tested and confirmed working using the following software versions:
- Python 2.7.12 + PyYAML 3.11
- Python 3.6.8  + PyYAML 5.1

### How to install PyYaML

    pip install pyyaml
