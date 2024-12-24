echo "gpu_ids: $1"
echo "start: $2"
echo "end: $3"
echo "dataset: $4"
echo "batch_size: $5"
echo "lr: $6"

if [ "$4" = "cifar10" ] || [ "$4" = "cifar2" ]; then
    model="resnet9"
    ori_dataset="CIFAR10"
elif [ "$4" = "imagenet" ]; then
    model="resnet18"
    ori_dataset="IMAGENET"
fi

for seed in `seq 0 2`
do
    echo "seed: $seed"
    for index in `seq $2 $3`
    do
        echo "index: $index"
        CUDA_VISIBLE_DEVICES=$1 python train.py \
            --seed $seed \
            --load-dataset \
            --dataset-dir ../Dataset/$ori_dataset \
            --dataset $4 \
            --train-index-path ./data/$4/lds_val/sub-idx-$index.pkl \
            --test-index-path ./data/$4/idx-test.pkl \
            --batch-size $5 \
            --model $model \
            --learning-rate $6 \
            --save-dir ./saved/models/$4/lds-val/index-$index-seed-$seed \
            --save-interval 24
    done
done