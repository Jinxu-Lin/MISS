gpu_ids=$1
start=$2
end=$3
dataset=$4
batch_size=$5
lr=$6

echo "gpu_ids: $gpu_ids"
echo "start: $start"
echo "end: $end"
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

for seed in `seq 0 2`
do
    echo "seed: $seed"
    for index in `seq $start $end`
    do
        echo "index: $index"
        CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
            --seed $seed \
            --load-dataset \
            --dataset-dir ../Dataset/$ori_dataset \
            --dataset $dataset \
            --train-index-path ./data/$dataset/lds-val/sub-idx-$index.pkl \
            --test-index-path ./data/$dataset/idx-test.pkl \
            --batch-size $batch_size \
            --model $model \
            --learning-rate $lr \
            --save-dir ./saved/models/$dataset/lds-val/index-$index-seed-$seed \
            --save-interval 24
    done
done