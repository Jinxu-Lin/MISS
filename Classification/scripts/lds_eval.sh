gpu_ids=$1
start=$2
end=$3
dataset=$4
dataset_split=$5
batch_size=$6

echo "gpu_ids: $gpu_ids"
echo "start: $start"
echo "end: $end"
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

for seed in `seq 0 2`
do
    echo "seed: $seed"
    for index in `seq $start $end`
    do
        echo "index: $index"
        CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
            --seed $seed \
            --load-dataset \
            --dataset-dir ../Dataset/$ori_dataset \
            --dataset $dataset \
            --dataset-split $dataset_split \
            --test-index-path ./data/$dataset/idx-test.pkl \
            --batch-size $batch_size \
            --model $model \
            --model-dir ./saved/models/$dataset/lds-val/index-$index-seed-$seed \
            --model-name model_23.pth \
            --save-dir ./saved/models/$dataset/lds-val/index-$index-seed-$seed
    done
done