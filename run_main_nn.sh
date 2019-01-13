#!/bin/bash
exp=exp
model=fnn # cnn, fnn
nonlinear=relu # relu, sigmoid, tanh
lr=0.001
optim=adam
l2=1e-3
dropout=0.5 # 0
batchSize=64
testbatchSize=64
max_norm=5
max_epoch=50
deviceId=0
    # --channel 1024 256 64 --kernel_size 7 5 3 --maxpool_kernel_size 4 4 4 \
python main_nn.py --experiment $exp --model $model --nonlinear $nonlinear --split_ratio 0.1\
    --affine_layers 512 128 \
    --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --max_epoch $max_epoch --max_norm $max_norm \
    --optim $optim --deviceId $deviceId --test_batchSize $testbatchSize #--batchnorm
