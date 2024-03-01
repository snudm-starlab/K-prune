GPU=0
TASK=mrpc
EXP_NAME=test_run/$TASK
MODEL=bert-base-uncased
CKPT_DIR=../ckpts/$MODEL/$TASK/

SAMPLE_SIZE=100000
SEED=0
CONS=0.5
LAM_PRED=1.
LAM_REP=2.5e-4
T=2
MU=64

python src/main.py --model_name $MODEL --ckpt_dir $CKPT_DIR --exp_name $EXP_NAME \
                   --task_name $TASK --gpu $GPU --seed $SEED --num_tokens $SAMPLE_SIZE \
                   --constraint $CONS --lam_pred $LAM_PRED --lam_rep $LAM_REP --T $T --mu $MU \
                   --sublayerwise_tuning
