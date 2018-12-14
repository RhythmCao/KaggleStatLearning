#!/bin/bash
exp=exp
model=svm # lda
type=svc # nusvc, linearsvc, lda, qda
split=0

gamma=scale # atuo
C=1
nu=0.5
kernel=rbf # linear, poly, sigmoid
decision_function_shape=ovr # ovo, crammer_singer
degree=3
coef0=0.0
penalty=l2 # l1
loss=squared_hinge # hinge

reg_param=0.5
solver=lsqr # svd, lsqr, eigen
shrinkage=0.5 # float

python main_stat.py --experiment $exp --model $model --type $type --split_ratio $split \
            --kernel $kernel --gamma $gamma --C $C --nu $nu --decision_function_shape $decision_function_shape \
            --degree $degree --coef0 $coef0 --penalty $penalty --loss $loss --dual
            # --reg_param $shrinkage --solver $solver --shrinkage $shrinkage