#!/bin/bash
exp=exp
split=0

model=knn # logistic, ridge, knn

if [ $model == 'logistic' ] ;then
    penalty=l2 # l1, l2
    C="0.5 1.0 1.5 2 2.5 3 3.5 4 4.5 5" # smaller values specify stronger regularization
    solver=liblinear # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    cv=5
    python main_classifier.py --experiment $exp --model $model --split_ratio $split \
                --penalty $penalty --C $C --solver $solver --cv $cv
elif [ $model == 'ridge' ] ;then
    alpha="0.5 1.0 1.5 2 2.5 3 3.5 4 4.5 5" # larger values specify stronger regularization
    cv=5
    python main_classifier.py --experiment $exp --model $model --split_ratio $split \
                --alpha $alpha --cv $cv
elif [ $model == 'knn' ] ;then
    k=9
    weights=uniform # 'uniform', 'distance'
    python main_classifier.py --experiment $exp --model $model --split_ratio $split \
                --k $k --weights $weights
else
    echo "Unknown model type ... ..."
fi