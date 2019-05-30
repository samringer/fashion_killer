set -e

experiment=TEST
learning_rate=2e-5
batch_size=128
over_train=False
use_fp16=True
num_epochs=30

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/DeformGAN/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz DeformGAN/* ${EXP_DIR}/code
rsync --quiet -avhz train_deform_gan.sh ${EXP_DIR}/code/

python -m DeformGAN.train \
    --task_path='/home/sam/experiments/DeformGAN' \
    --data_dir='/home/sam/data/asos/2604_clean/train' \
    --exp_name=$exp_name \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --over_train=${over_train} \
    --use_fp16=$use_fp16 \
    --num_epochs=$num_epochs \
    #--load_checkpoint=$load_checkpoint \
