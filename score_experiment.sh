#! /bin/zsh
set -eu

results_f=${1}/results.log

for i in `ls ${1}/models/*.pt | sort -V`; do
    echo $i | tee -a $results_f
    echo `python -m app_transfer.score ${i}` |& tee -a $results_f
    echo "" | tee -a $results_f
done
