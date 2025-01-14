gpu_ids=$1
seed=$2
dataset=$3
dataset_split=$4
batch_size=$5

echo "gpu_ids: $gpu_ids"
echo "seed: $seed"
echo "dataset: $dataset"
echo "dataset_split: $dataset_split"
echo "batch_size: $batch_size"

if [ "$dataset" = "cifar10" ] || [ "$dataset" = "cifar2" ]; then
    model="resnet9"
    ori_dataset="CIFAR10"
elif [ "$dataset" = "imagenet" ]; then
    model="resnet18"
    ori_dataset="IMAGENET"
fi

CUDA_VISIBLE_DEVICES=$gpu_ids python 3_error.py \
    --seed $seed \
    --load-dataset \
    --dataset-dir ../Dataset/$ori_dataset \
    --dataset $dataset \
    --dataset-split $dataset_split \
    --train-index-path ./data/$dataset/idx-train.pkl \
    --test-index-path ./data/$dataset/idx-test.pkl \
    --batch-size $batch_size \
    --model $model \
    --model-dir ./saved/models/$dataset/origin/seed-$seed \
    --model-name model_23.pth \
    --save-dir ./saved/error/$dataset/seed-$seed