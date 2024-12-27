echo "gpu_ids: $1"
echo "seed: $2"
echo "dataset: $3"
echo "dataset_split: $4"

if [ "$3" = "cifar10" ] || [ "$3" = "cifar2" ]; then
    model="resnet9"
    ori_dataset="CIFAR10"
elif [ "$3" = "imagenet" ]; then
    model="resnet18"
    ori_dataset="IMAGENET"
fi

CUDA_VISIBLE_DEVICES=$1 python grad.py \
    --load-dataset \
    --dataset-dir ../Dataset/$ori_dataset \
    --dataset $3 \
    --dataset-split $4 \
    --train-index-path ./data/$3/idx-train.pkl \
    --test-index-path ./data/$3/idx-test.pkl \
    --model $model \
    --model-dir ./saved/models/$3/origin/seed-$2 \
    --model-name model_23.pth \
    --save-dir ./saved/grad/$3/