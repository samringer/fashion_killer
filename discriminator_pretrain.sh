#! /bin/zsh
set -eu

experiment=128x128_fix_data_aug_disc_pretrain
learning_rate=1e-4
batch_size=32
overtrain=False
use_fp16=False
num_epochs=500
generator_path=/home/sam/experiments/app_transfer/18_08_fix_lr_dependency_data_aug/models/final.pt

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/app_transfer/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz app_transfer/* ${EXP_DIR}/code
rsync --quiet -avhz discriminator_pretrain.sh ${EXP_DIR}/

python -m app_transfer.discriminator_pretrain \
    --task_path='/home/sam/experiments/app_transfer' \
    --data_dir='/home/sam/data/asos/1307_clean/train' \
    --exp_name=$exp_name \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --overtrain=$overtrain \
    --use_fp16=$use_fp16 \
    --gen_path=$generator_path \
    --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
