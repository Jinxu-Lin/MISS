cd Classification
bash scripts/grad.sh 0 1 cifar10 train 16
bash scripts/grad.sh 0 2 cifar10 train 16
bash scripts/grad.sh 0 0 cifar10 test 16
bash scripts/grad.sh 0 1 cifar10 test 16
bash scripts/grad.sh 0 2 cifar10 test 16