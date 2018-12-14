#!/bin/bash
exp=exp
model=fnn # cnn
nonlinear=relu
lr=0.001
optim=adam
l2=1e-3
dropout=0.5
batchSize=64
testbatchSize=64
max_norm=5
max_epoch=50
deviceId=0
# --channel 512 128 32 --kernel_size 5 3 3 --maxpool_kernel_size 4 2 2 
python main_nn.py --experiment $exp --model $model --nonlinear $nonlinear \
    --affine_layers 1024 256 64\
    --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --max_epoch $max_epoch --max_norm $max_norm \
    --optim $optim --deviceId $deviceId --test_batchSize $testbatchSize #--batchnorm
