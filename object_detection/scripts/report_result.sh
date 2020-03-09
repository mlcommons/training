#!/bin/bash
set -ex

declare -a run_times

readarray -t run_times < "./results/closed/report.txt"

sorted_test_arr=( $( printf "%s\n" "${run_times[@]}" | sort -n ) )
unset sorted_test_arr[0]
unset sorted_test_arr[4]

sum=0

for i in ${sorted_test_arr[@]}
do
  sum=`expr $sum + $i`
done

BENCHMARK_RESULT=$(echo "$sum/3" | bc -l)
echo $BENCHMARK_RESULT
