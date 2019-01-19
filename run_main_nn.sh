#!/bin/bash
exp=exp
model=cnn # cnn, fnn
normalize="z-score" # min-max, z-score, none
nonlinear=relu # relu, sigmoid, tanh
lr=0.001
optim=adam
l2=1e-3
dropout=0.5 # 0 in CNN, we seldom use dropout, set it =0
batchSize=64
testbatchSize=64
max_norm=5
max_epoch=50
deviceId=0
    # --affine_layers 512 128 \
python main_nn.py --experiment $exp --model $model --nonlinear $nonlinear --split_ratio 0.2\
    --channel 256 64 --kernel_size 5 5 --maxpool_kernel_size 4 4 --batchnorm\
    --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --max_epoch $max_epoch --max_norm $max_norm \
    --optim $optim --deviceId $deviceId --test_batchSize $testbatchSize --normalize $normalize 
