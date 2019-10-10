#! /bin/zsh
set -eu

# Note
# Trying to diagnose why its running out of memory
experiment=DDP_smaller_vgg
generator_lr=2e-5
discriminator_lr=2e-4
batch_size=8
num_epochs=500

generator_path=/home/sam/experiments/app_transfer/30_09_DDP_acc_batches_2/weights/2.pt
discriminator_path=/home/sam/experiments/app_transfer/01_10_torch_lightning_discriminator/models/final.pt

#checkpoint=/home/sam/experiments/app_transfer/01_08_128x128_continue/models/190000.chk

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/app_transfer/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz app_transfer/* ${EXP_DIR}/code
rsync --quiet -avhz joint_train.sh ${EXP_DIR}

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    -m app_transfer.joint_train \
        '/home/sam/experiments/app_transfer' \
        $exp_name \
        $generator_path \
        $discriminator_path \
        --data_dir='/home/sam/data/asos/1307_clean/train' \
        --batch_size=$batch_size \
        --gen_lr=$generator_lr \
        --disc_lr=$discriminator_lr \
        --use_fp16 \
        --distributed \
        --num_epochs=$num_epochs |& tee -a ${EXP_DIR}/train.log
        #--load_checkpoint=$checkpoint \
