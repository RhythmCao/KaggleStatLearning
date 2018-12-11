#!/bin/bash
exp=exp
model=fnn
nonlinear=relu
lr=0.001
optim=adam
l2=1e-3
dropout=0.5
batchSize=64
max_norm=5
max_epoch=100
deviceId=0
python main_nn.py --experiment $exp --model $model --nonlinear $nonlinear \
    --affine_layers 1024 256 64 --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --max_epoch $max_epoch --max_norm $max_norm \
    --optim $optim --deviceId $deviceId
