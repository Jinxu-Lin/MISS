echo "gpu_ids: $1"
echo "seed: $2"
echo "dataset: $3"
echo "batch_size: $4"
echo "lr: $5"

if [ "$3" = "cifar10" ] || [ "$3" = "cifar2" ]; then
    model="resnet9"
    ori_dataset="CIFAR10"
elif [ "$3" = "imagenet" ]; then
    model="resnet18"
    ori_dataset="IMAGENET"
fi

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --seed $2 \
    --load-dataset \
    --dataset-dir ../Dataset/$ori_dataset \
    --dataset $3 \
    --train-index-path ./data/$3/idx-train.pkl \
    --test-index-path ./data/$3/idx-test.pkl \
    --batch-size $4 \
    --model $model \
    --learning-rate $5 \
    --save-dir ./saved/models/$3/origin/seed-$2 \
    --save-interval 10