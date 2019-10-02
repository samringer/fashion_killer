#! /bin/zsh
set -eu

# Note
# First lightning joint
experiment=lightning_joint
generator_lr=2e-100
discriminator_lr=2e-4
batch_size=4
overtrain=False
num_epochs=400
train_data_dir='/home/sam/data/asos/1307_clean/train'
val_data_dir='/home/sam/data/asos/1307_clean/val'

generator_path=/home/sam/experiments/app_transfer/30_09_DDP_acc_batches_2/weights/2.pt
discriminator_path=/home/sam/experiments/app_transfer/01_10_torch_lightning_discriminator/models/final.pt

#checkpoint=/home/sam/experiments/app_transfer/01_08_128x128_continue/models/190000.chk

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/app_transfer/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz app_transfer/* ${EXP_DIR}/code
rsync --quiet -avhz joint_train.sh ${EXP_DIR}

python -m app_transfer.lightning_joint \
    '/home/sam/experiments/app_transfer' \
    $exp_name \
    $train_data_dir \
    $val_data_dir \
    $generator_path \
    $discriminator_path \
    --batch_size=$batch_size \
    --gen_lr=$generator_lr \
    --disc_lr=$discriminator_lr \
    --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
    #--load_checkpoint=$checkpoint \
