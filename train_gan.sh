set -e

experiment=BIG_inc_l1_loss
load_checkpoint=/home/sam/experiments/GAN/07_05_BIG_much_smaller_lrs_2/models/25000.chk
learning_rate=1e-5
discriminator_lr=5e-5
batch_size=8
over_train=False
use_fp16=True
num_epochs=100

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/GAN/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz GAN/* ${EXP_DIR}/code
rsync --quiet -avhz train_gan.sh ${EXP_DIR}/code/

python -m GAN.train \
    --task_path='/home/sam/experiments/GAN' \
    --data_dir='/home/sam/data/asos/2604_clean/train' \
    --exp_name=$exp_name \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --discriminator_lr=$discriminator_lr \
    --over_train=${over_train} \
    --use_fp16=$use_fp16 \
    --num_epochs=$num_epochs \
    --load_checkpoint=$load_checkpoint \
