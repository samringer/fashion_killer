#! /bin/zsh
set -eu

experiment=TEST_partial_gen_training
generator_lr=5e-6
discriminator_lr=2e-4
batch_size=8
over_train=False
use_fp16=False
num_epochs=100

generator_path=/home/sam/experiments/DeformGAN/06_07_BIG_from_checkpoint/models/10.pt
discriminator_path=/home/sam/experiments/DeformGAN/07_07_to_convergence/models/final.pt

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/DeformGAN/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz DeformGAN/* ${EXP_DIR}/code
rsync --quiet -avhz joint_train.sh ${EXP_DIR}

python -m DeformGAN.joint_train \
    --task_path='/home/sam/experiments/DeformGAN' \
    --data_dir='/home/sam/data/asos/0107_clean/train' \
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
