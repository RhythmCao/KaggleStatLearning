#coding=utf8
import argparse, random, os, sys, time
import numpy as np
import logging
import gc

from utils.data_utils import load_train_data, load_test_data, write_csv_result
import utils.util as util

from models.svm import SVMModel, SVCModel, NuSVCModel, LinearSVCModel
from models.lda import LDAModel, LinearDAModel, QuadraticDAModel

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='exp', help='Where to store samples and models')
parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')

parser.add_argument('--model', choices=['svm','lda'], default='svm')
parser.add_argument('--type', choices=['svc','nusvc','linearsvc','lda','qda'], default='svc')
parser.add_argument('--normalize', choices=['min-max', 'z-score', 'none'], required=True)

svm_paras = parser.add_argument_group('SVM model parameters')
svm_paras.add_argument('--gamma', choices=['scale','auto'], default='scale')
svm_paras.add_argument('--C', type=float, default=1)
svm_paras.add_argument('--nu', type=float, default=0.5)
svm_paras.add_argument('--kernel', choices=['linear','rbf','poly','sigmoid'], default='rbf')
svm_paras.add_argument('--decision_function_shape', choices=['ovo','ovr','crammer_singer'], default='ovr')
svm_paras.add_argument('--degree', type=int, default=3)
svm_paras.add_argument('--coef0', type=float, default=0)
svm_paras.add_argument('--penalty', choices=['l1','l2'], default='l2')
svm_paras.add_argument('--loss', choices=['hinge','squared_hinge'], default='squared_hinge')
svm_paras.add_argument('--dual', action='store_true')

lda_paras = parser.add_argument_group('LDA model parameters')
lda_paras.add_argument('--reg_param', type=float, default=0.0)
lda_paras.add_argument('--solver', choices=['svd','lsqr','eigen'], default='svd')
lda_paras.add_argument('--shrinkage', type=float, default=0, help='shrinkage param used in LDA, if <0, shrinkage=None, if >1, shrinkage=auto')

parser.add_argument('--cv', type=int, default=5, help='K-fold cross validation, default k=5')
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')

opt = parser.parse_args()
if opt.shrinkage > 1:
    opt.shrinkage = 'auto'
if opt.solver == 'svd' or opt.shrinkage < 0:
    opt.shrinkage = None

exp_path = util.hyperparam_string_stat(opt)
if opt.normalize != 'none':
    exp_path += '__norm_minmax' if opt.normalize == 'min-max' else '__norm_zscore'
exp_path = os.path.join(opt.experiment, exp_path)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

logFormatter = logging.Formatter('%(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w') # override written
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
if not opt.noStdout:
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
logger.info("Parameters:"+str(opt))
logger.info("Experiment path: %s" % (exp_path))
logger.info(time.asctime(time.localtime(time.time())))

random.seed(opt.random_seed)
np.random.seed(opt.random_seed)

# load dataset
start_time = time.time()
train_data, train_label, paras = load_train_data(split_ratio=0, normalize=opt.normalize)
test_data = load_test_data(normalize=opt.normalize, paras=paras)
logger.info("Prepare data ... cost %.4fs" % (time.time()-start_time))

if opt.model == 'svm':
    if opt.type == 'svc':
        train_model = SVCModel(kernel=opt.kernel, C=opt.C, degree=opt.degree, gamma=opt.gamma, coef0=opt.coef0,
                        tol=opt.tol, decision_function_shape=opt.decision_function_shape, random_state=opt.random_seed)
    elif opt.type == 'nusvc':
        train_model = NuSVCModel(kernel=opt.kernel, nu=opt.nu, degree=opt.degree, gamma=opt.gamma, coef0=opt.coef0,
                        tol=opt.tol, decision_function_shape=opt.decision_function_shape, random_state=opt.random_seed)
    elif opt.type == 'linearsvc':
        train_model = LinearSVCModel(penalty=opt.penalty, loss=opt.loss, dual=opt.dual, tol=opt.tol, 
                        C=opt.C, multi_class=opt.decision_function_shape, random_state=opt.random_seed)
    else:
        raise ValueError('[Error]: unknown svm type!')
elif opt.model == 'lda':
    if opt.type == 'lda':
        train_model = LinearDAModel(solver=opt.solver, shrinkage=opt.shrinkage, tol=opt.tol)
    elif opt.type == 'qda':
        train_model = QuadraticDAModel(reg_param=opt.reg_param, tol=opt.tol)
    else:
        raise ValueError('[Error]: unknown lda type!')

score = train_model.train(train_data, train_label)
train_model.save_model(os.path.join(exp_path, 'train.model'))
_, train_acc = train_model(train_data, train_label)
logger.info('Training acc is %.4f' % (train_acc))
if opt.cv > 0:
    scores = train_model.get_cv_accuracy(train_data, train_label, cv=opt.cv)
    logger.info("Cross validation accuracy is %.4f" % (scores))
logger.info('Start predicting labels on Test set ...')
result = train_model(test_data)
logger.info('Start writing results into file %s' % (os.path.join(exp_path,'result.csv')))
write_csv_result(result, outfile=os.path.join(exp_path,'result.csv'))