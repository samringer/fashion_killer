set -e

experiment=
learning_rate=1e-4
over_train=False
use_fp16=True

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/V_U_Net/${exp_name}

mkdir -p $EXP_DIR
mkdir -p ${EXP_DIR}/code

rsync --quiet -avhz v_u_net ${EXP_DIR}/code/pose_detector
rsync --quiet -avhz pose_drawer ${EXP_DIR}/code/pose_drawer
rsync --quiet -avhz utils.py ${EXP_DIR}/code/
rsync --quiet -avhz train_v_u_net.sh ${EXP_DIR}/code/

python -m v_u_net.train \
    --task_path='/home/sam/experiments/V_U_Net' \
    --data_dir='/home/sam/data/deepfashion' \
    --exp_name=$exp_name \
    --batch_size=8 \
    --learning_rate=$learning_rate \
    --over_train=${over_train} \
    --use_fp16=$use_fp16 \
    --num_epochs=500
