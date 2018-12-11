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
max_epoch=64
deviceId=0
python main.py --experiment $exp --model $model --nonlinear $nonlinear \
    --affine_layers 512 128 --lr $lr --l2 $l2 --dropout $dropout --batchSize $batchSize --max_epoch $max_epoch --max_norm $max_norm \
    --optim $optim --deviceId $deviceId
