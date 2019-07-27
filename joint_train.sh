#! /bin/zsh
set -eu

experiment=64x64_joint
generator_lr=2e-5
discriminator_lr=2e-4
batch_size=32
over_train=False
use_fp16=False
num_epochs=100

generator_path=/home/sam/experiments/app_transfer/26_07_TEST/models/5.pt
discriminator_path=/home/sam/experiments/app_transfer/26_07_64x64_dis_pretrain/models/final.pt

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/app_transfer/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz app_transfer/* ${EXP_DIR}/code
rsync --quiet -avhz joint_train.sh ${EXP_DIR}

python -m app_transfer.joint_train \
    --task_path='/home/sam/experiments/app_transfer' \
    --data_dir='/home/sam/data/asos/1307_clean/train' \
    --exp_name=$exp_name \
    --batch_size=$batch_size \
    --generator_lr=$generator_lr \
    --discriminator_lr=$discriminator_lr \
    --over_train=$over_train \
    --use_fp16=$use_fp16 \
    --generator_path=$generator_path \
    --discriminator_path=$discriminator_path \
    --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
    #--load_checkpoint=$load_checkpoint \
