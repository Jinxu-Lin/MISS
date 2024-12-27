echo "gpu_ids: $1"
echo "seed: $2"
echo "dataset: $3"
echo "batch_size: $4"
echo "lr: $5"

if [ "$4" = "cifar10" ] || [ "$4" = "cifar2" ]; then
    model="resnet9"
    ori_dataset="CIFAR10"
elif [ "$4" = "imagenet" ]; then
    model="resnet18"
    ori_dataset="IMAGENET"
fi

CUDA_VISIBLE_DEVICES=$1 python train.py \
    --seed $2 \
    --load-dataset \
    --dataset-dir ../Dataset/$ori_dataset \
    --dataset $4 \
    --train-index-path ./data/$4/idx-train.pkl \
    --test-index-path ./data/$4/idx-test.pkl \
    --batch-size $4 \
    --model $model \
    --learning-rate $5 \
    --save-dir ./saved/models/$3/origin \
    --save-interval 10