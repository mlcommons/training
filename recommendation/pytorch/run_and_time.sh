#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=0.635
BASEDIR=$(dirname -- "$0")

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Get command line seed
seed=${1:-1}

echo "unzip ml-20m.zip"
if unzip -u ml-20m.zip
then
    echo "Duplicating each entry and changing to new user ID..."
    python $BASEDIR/multiply_users.py ml-20m/ratings.csv 2
    echo "Duplicating each entry and changing to new item ID..."
    python $BASEDIR/multiply_items.py ml-20m/ratings_users_expanded.csv 2
    echo "Copying expanded file (2x users 2x items) to ./ml-20m_2xui..."
    mkdir -p ml-20m_2xui
    cp ./ml-20m/ratings_users_expanded_items_expanded.csv ./ml-20m_2xui/ratings.csv
    echo "Start processing ml-20m_2xui/ratings.csv"
    t0=$(date +%s)
	python $BASEDIR/convert.py ml-20m_2xui/ratings.csv ml-20m_2xui --negatives 999
    t1=$(date +%s)
	delta=$(( $t1 - $t0 ))
    echo "Finish processing ml-20m_2xui/ratings.csv in $delta seconds"

    echo "Start training"
    t0=$(date +%s)
	python $BASEDIR/ncf.py ml-20m_2xui -l 0.0005 -b 2048 --layers 512 512 512 512 -f 64 \
		--seed $seed --threshold $THRESHOLD --processes 10
    t1=$(date +%s)
	delta=$(( $t1 - $t0 ))
    echo "Finish training in $delta seconds"

	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="recommendation"


	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Problem unzipping ml-20.zip"
	echo "Please run 'download_data.sh && verify_datset.sh' first"
fi





