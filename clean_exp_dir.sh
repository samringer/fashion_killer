#! /bin/zsh
set -eu
if [[ $# -ne 1 ]]; then
    echo "Please provide the exp dir to clean"
    exit 1
fi
mv `ls ${1}/models/*.pt | sort -rV | head -1` ${1}/models/final_model
mv `ls ${1}/models/*.chk | sort -rV | head -1` ${1}/models/final_checkpoint
ls ${1}/models/*.pt | xargs rm
ls ${1}/models/*.chk | xargs rm
mv ${1}/models/final_model ${1}/models/final.pt
mv ${1}/models/final_checkpoint ${1}/models/final.chk

