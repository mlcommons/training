#!/bin/bash
set -ex

timestamp=$(date "+%Y%m%d-%H%M%S")
results_dir="results/closed/$timestamp/ssd"
report_file="results/closed/report.txt"

if [ -f $report_file ]; then
rm $report_file
fi

# Generate the output directory
mkdir -p ./$results_dir
ln -sfn $timestamp/ ./results/closed/latest

# Run the training 5 times
counter=1
while [ $counter -le 5 ]
do
export COMPLIANCE_FILE="/workspace/$results_dir/result_${counter}.txt"
. ./ssd/run_and_time.sh
echo $result >> $report_file
((counter++))
done
