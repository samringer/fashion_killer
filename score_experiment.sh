#! /bin/zsh
set -eu

results_f=${1}/results.log

for i in `ls ${1}/models/*.pt | sort -V`; do
    echo $i >> $results_f
    echo `python -m DeformGAN.score ${i}` >> $results_f
    echo "" >> $results_f
done
