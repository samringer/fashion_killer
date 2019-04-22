set -e

experiment=BIG_7_train_joints_0.01_null_pred_penalty
learning_rate=2e-4
over_train=False
use_fp16=True
min_joints_to_train_on=7

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/Pose_Detector/${exp_name}

mkdir -p $EXP_DIR
mkdir -p ${EXP_DIR}/code

rsync --quiet -avhz pose_detector ${EXP_DIR}/code/pose_detector
rsync --quiet -avhz pose_drawer ${EXP_DIR}/code/pose_drawer
rsync --quiet -avhz utils.py ${EXP_DIR}/code/
rsync --quiet -avhz train_pose_detector.sh ${EXP_DIR}/code/

python -m pose_detector.train \
    --exp_name=$exp_name \
    --batch_size=4 \
    --learning_rate=$learning_rate \
    --over_train=${over_train} \
    --use_fp16=$use_fp16 \
    --min_joints_to_train_on=$min_joints_to_train_on \
    --num_epochs=500
