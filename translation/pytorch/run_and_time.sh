#/bin/bash

# runs benchmark and reports time to convergence

# to use the script:

#   run_and_time.sh <random seed 1-5>


set -e

# start timing

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


# run benchmark

seed=${1:-1}

echo "running benchmark with seed $seed"
# Quality of 5 is roughly 1 epoch
./run.sh $seed 25

sleep 3

ret_code=$?; if [[ $ret_code != 0 ]]; then exit $ret_code; fi



# end timing

end=$(date +%s)

end_fmt=$(date +%Y-%m-%d\ %r)

echo "ENDING TIMING RUN AT $end_fmt"



# report result

result=$(( $end - $start ))

result_name="transformer"



echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
