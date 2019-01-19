#!/bin/bash
exp=exp
normalize="z-score" # min-max, z-score, none
model=ridge # logistic, ridge, knn

if [ $model == 'logistic' ] ;then
    penalty=l2 # l1, l2
    C="0.001 0.002 0.005 0.01 0.02 0.05 0.1" # smaller values specify stronger regularization
    solver=sag # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    cv=5
    python main_classifier.py --experiment $exp --model $model --cv $cv \
                --penalty $penalty --C $C --solver $solver --normalize $normalize
elif [ $model == 'ridge' ] ;then
    alpha="100000 1000000" # larger values specify stronger regularization
    cv=5
    python main_classifier.py --experiment $exp --model $model --cv $cv \
                --alpha $alpha --normalize $normalize --cv_score
elif [ $model == 'knn' ] ;then
    k=9
    weights=uniform # 'uniform', 'distance'
    python main_classifier.py --experiment $exp --model $model --cv $cv \
                --k $k --weights $weights --normalize $normalize
else
    echo "Unknown model type ... ..."
fi