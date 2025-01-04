gpu_ids=$1
seed=$2
dataset=$3
batch_size=$4
lr=$5

echo "gpu_ids: $gpu_ids"
echo "seed: $seed"
echo "dataset: $dataset"
echo "batch_size: $batch_size"
echo "lr: $lr"

if [ "$dataset" = "cifar10" ] || [ "$dataset" = "cifar2" ]; then
    model="resnet9"
    ori_dataset="CIFAR10"
elif [ "$dataset" = "imagenet" ]; then
    model="resnet18"
    ori_dataset="IMAGENET"
fi

CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --seed $seed \
    --load-dataset \
    --dataset $dataset \
    --dataset-dir ../Dataset/$ori_dataset \
    --train-index-path ./data/$dataset/idx-train.pkl \
    --test-index-path ./data/$dataset/idx-test.pkl \
    --batch-size $batch_size \
    --model $model \
    --learning-rate $lr \
    --save-dir ./saved/models/$dataset/origin/seed-$seed \
    --save-interval 10