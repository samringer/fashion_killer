set -e

experiment=BIG_fix_shape_bug
learning_rate=5e-5
discriminator_lr=2e-4
batch_size=16
over_train=False
use_fp16=False

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
    --num_epochs=500 \
    --load_checkpoint=$checkpoint_path
