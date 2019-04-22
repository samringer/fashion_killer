set -e

experiment=BIG_from_checkpoint_b
checkpoint_path=/home/sam/experiments/AsosNet/21_04_BIG_no_fp16_lr_5e-5_b/models/30000.chk
learning_rate=5e-5
over_train=False
use_fp16=False

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/AsosNet/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz asos_net/* ${EXP_DIR}/code
rsync --quiet -avhz train_asos_net.sh ${EXP_DIR}/code/

python -m asos_net.train \
    --task_path='/home/sam/experiments/AsosNet' \
    --data_dir='/home/sam/data/asos/train_clean' \
    --exp_name=$exp_name \
    --batch_size=8 \
    --learning_rate=$learning_rate \
    --over_train=${over_train} \
    --use_fp16=$use_fp16 \
    --num_epochs=500 \
    --load_checkpoint=$checkpoint_path
