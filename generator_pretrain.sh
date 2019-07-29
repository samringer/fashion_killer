#! /bin/zsh
set -eu

# NOTES
# Made the learning rate half the size and batch size 4 times smaller
experiment=256x256_g_pretrain
learning_rate=5e-5
batch_size=8
over_train=False
use_fp16=False
num_epochs=1000
#load_checkpoint=/home/sam/experiments/DeformGAN/11_07_BIG_flipped_no_softmax/models/50000.chk

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/app_transfer/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz app_transfer/* ${EXP_DIR}/code
rsync --quiet -avhz generator_pretrain.sh ${EXP_DIR}/

python -m app_transfer.generator_pretrain \
    --task_path='/home/sam/experiments/app_transfer' \
    --data_dir='/home/sam/data/asos/1307_clean/train' \
    --exp_name=$exp_name \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --over_train=$over_train \
    --use_fp16=$use_fp16 \
    --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
    #--load_checkpoint=$load_checkpoint \

#./score_experiment.sh $EXP_DIR
