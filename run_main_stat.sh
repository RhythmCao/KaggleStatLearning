#!/bin/bash
exp=exp
split=0

# SVM or LDA Model
model=svm # svm, lda
type=nusvc # svc, nusvc, linearsvc, lda, qda

if [ $type == 'svc' ] ;then
    C=1
    kernel=rbf # rbf, linear, poly, sigmoid
    degree=3 # degree of the poly kernel function
    gamma=auto # auto, scale
    coef0=0.0 # only significant in poly and sigmoid
    decision_function_shape=ovr # ovo, ovr
    python main_stat.py --experiment $exp --model $model --type $type --split_ratio $split \
                --kernel $kernel --gamma $gamma --C $C --decision_function_shape $decision_function_shape \
                --degree $degree --coef0 $coef0 
elif [ $type == 'nusvc' ] ;then
    nu=0.5 # upper bound on fraction of training errors and lower bound of fractions upport vectors
    kernel=rbf # rbf, linear, poly, sigmoid
    degree=3 # degree of the poly kernel function
    gamma=scale # auto, scale
    coef0=0.0 # only significant in poly and sigmoid
    decision_function_shape=ovr # ovo, ovr
    python main_stat.py --experiment $exp --model $model --type $type --split_ratio $split \
                --kernel $kernel --gamma $gamma --nu $nu --decision_function_shape $decision_function_shape \
                --degree $degree --coef0 $coef0 
elif [ $type == 'linearsvc' ] ;then
    C=1
    loss=hinge # hinge, squared_hinge
    penalty=l2 # l1, l2
    decision_function_shape=ovr # ovr, crammer_singer
    python main_stat.py --experiment $exp --model $model --type $type --split_ratio $split \
                --C $C --decision_function_shape $decision_function_shape \
                --penalty $penalty --loss $loss
elif [ $type == 'lda' ] ;then
    solver=eigen # svd, lsqr, eigen
    shrinkage=0.5 # float
    python main_stat.py --experiment $exp --model $model --type $type --split_ratio $split \
                --solver $solver --shrinkage $shrinkage
elif [ $type == 'qda' ] ;then
    reg_param=0.5
    python main_stat.py --experiment $exp --model $model --type $type --split_ratio $split \
                --reg_param $reg_param
else
    echo "Unknown model type ... ..."
fi