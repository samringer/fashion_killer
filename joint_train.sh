#! /bin/zsh
set -eu

# Note
# lr data augmentation bug fixed
experiment=128x128_joint_fix_data_aug
generator_lr=2e-5
discriminator_lr=2e-4
batch_size=32
over_train=False
use_fp16=False
num_epochs=200

generator_path=/home/sam/experiments/app_transfer/18_08_fix_lr_dependency_data_aug/models/final.pt
discriminator_path=/home/sam/experiments/app_transfer/21_08_128x128_fix_data_aug_disc_pretrain/models/final.pt

#checkpoint=/home/sam/experiments/app_transfer/01_08_128x128_continue/models/190000.chk

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
    --gen_lr=$generator_lr \
    --disc_lr=$discriminator_lr \
    --over_train=$over_train \
    --use_fp16=$use_fp16 \
    --gen_path=$generator_path \
    --disc_path=$discriminator_path \
    --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
    #--load_checkpoint=$checkpoint \
