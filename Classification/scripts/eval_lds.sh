echo "gpu_ids: $1"
echo "start: $2"
echo "end: $3"
echo "dataset: $4"
echo "batch_size: $5"
echo "dataset_split: $6"

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
        CUDA_VISIBLE_DEVICES=$1 python eval.py \
            --seed $seed \
            --load-dataset \
            --dataset-dir ../Dataset/$ori_dataset \
            --dataset $4 \
            --dataset-split $6 \
            --test-index-path ./data/$4/idx-test.pkl \
            --batch-size $5 \
            --model $model \
            --model-dir ./saved/models/$4/lds-val/index-$index-seed-$seed \
            --model-name model_23.pth \
            --save-dir ./saved/models/$4/lds-val/index-$index-seed-$seed
    done
done