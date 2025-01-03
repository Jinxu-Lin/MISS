gpu_ids=$1
main_process_port=$2
seed=$3
dataset=$4

echo "gpu_ids: $gpu_ids"
echo "main_process_port: $main_process_port"
echo "seed: $seed"
echo "dataset: $dataset"

if [ "$dataset" = "cifar10" ] || [ "$dataset" = "cifar2" ]; then
    ori_dataset="CIFAR10"
fi

export HF_HOME="~/codes/.cache/huggingface"

accelerate launch --gpu_ids $1 --main_process_port=$2 train.py \
    --seed=$seed \
    --load-dataset \
    --dataset $dataset \
    --dataset-dir ../Dataset/$ori_dataset \
    --batch_size 128 \
    --dataloader_num_workers 8 \
    --resolution 32 \
    --model_config_name_or_path "config.json" \
    --learning_rate 1e-4 \
    --adam_weight_decay 1e-6 \
    --num_epochs 200 \
    --checkpointing_steps -1 \
    --gradient_accumulation_steps 1 \
    --logger "tensorboard" \
    --train-index-path ./data/$dataset/idx-train.pkl \
    --save-dir=./saved/models/$dataset/origin/seed-$seed \