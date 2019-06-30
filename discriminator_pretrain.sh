#! /bin/zsh
set -eu

experiment=BIG_discriminator_pretrain_no_sigmoid
learning_rate=1e-4
batch_size=128
over_train=False
use_fp16=False
num_epochs=500
generator_path=/home/sam/experiments/DeformGAN/26_06_Attn_all_the_way_plus_one_more_conv/models/195.pt
#load_checkpoint=/home/sam/experiments/DeformGAN/04_06_BIG_longer_higher_lr_wider_step/models/final.chk

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/DeformGAN/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz DeformGAN/* ${EXP_DIR}/code
rsync --quiet -avhz discriminator_pretrain.sh ${EXP_DIR}/

python -m DeformGAN.pretrain_discriminator \
    --task_path='/home/sam/experiments/DeformGAN' \
    --data_dir='/home/sam/data/asos/2604_clean/train' \
    --exp_name=$exp_name \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --over_train=$over_train \
    --use_fp16=$use_fp16 \
    --generator_path=$generator_path \
    --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
    #--load_checkpoint=$load_checkpoint \
