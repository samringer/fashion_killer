#! /bin/zsh
set -eu

# NOTES
experiment=DDP_acc_batches_2
learning_rate=1e-4
batch_size=4
over_train=False
use_fp16=True
num_epochs=20
train_data='/home/sam/data/asos/1307_clean/train'
val_data='/home/sam/data/asos/1307_clean/val'

#load_checkpoint=/home/sam/experiments/DeformGAN/11_07_BIG_flipped_no_softmax/models/50000.chk

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/app_transfer/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz app_transfer/* ${EXP_DIR}/code
rsync --quiet -avhz generator_pretrain.sh ${EXP_DIR}/

python -m app_transfer.lightning_model \
    '/home/sam/experiments/app_transfer' \
    $exp_name \
    $train_data \
    $val_data \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
    #--load_checkpoint=$load_checkpoint \

#./score_experiment.sh $EXP_DIR
