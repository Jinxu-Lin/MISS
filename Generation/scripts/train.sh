gpu_ids=$1
main_process_port=$2
num_processes=$3
seed=$4
dataset=$5

echo "gpu_ids: $gpu_ids"
echo "main_process_port: $main_process_port"
echo "num_processes: $num_processes"
echo "seed: $seed"
echo "dataset: $dataset"

if [ "$dataset" = "cifar10" ] || [ "$dataset" = "cifar2" ]; then
    ori_dataset="CIFAR10"
fi

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME="~/codes/.cache/huggingface"

accelerate launch --gpu_ids $gpu_ids --main_process_port=$main_process_port --num_processes=$num_processes train.py \
    --seed=$seed \
    --load-dataset \
    --dataset $dataset \
    --dataset-dir ../Dataset/$ori_dataset \
    --batch-size 128 \
    --dataloader-num-workers 8 \
    --resolution 32 \
    --model-config "config.json" \
    --learning-rate 1e-4 \
    --adam-weight-decay 1e-6 \
    --num-epochs 200 \
    --checkpointing-steps -1 \
    --gradient-accumulation-steps 1 \
    --logger "tensorboard" \
    --train-index-path ./data/$dataset/idx-train.pkl \
    --save-dir=./saved/models/$dataset/origin/seed-$seed \