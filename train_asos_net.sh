set -e

experiment=BIG
learning_rate=1e-4
over_train=False

exp_name=$(date +"%d_%m")_${experiment}
EXP_DIR=/home/sam/experiments/AsosNet/${exp_name}

mkdir -p $EXP_DIR

rsync --quiet -avhz asos_net/* ${EXP_DIR}/code
rsync --quiet -avhz train_asos_net.sh ${EXP_DIR}/code/

python -m asos_net.train \
    --exp_name=$exp_name \
    --batch_size=8 \
    --learning_rate=$learning_rate \
    --over_train=${over_train}
