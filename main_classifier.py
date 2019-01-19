#coding=utf8
import argparse, random, os, sys, time
import numpy as np
import logging
import gc

from utils.data_utils import load_train_data, load_test_data, write_csv_result
import utils.util as util
from models.classifier import *

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='exp', help='Where to store samples and models')
parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')
parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
parser.add_argument('--read_model', required=False, help='Only test: read model from this file')
parser.add_argument('--out_path', required=False, help='Only test: out_path')

parser.add_argument('--model', choices=['logistic','ridge','knn'], default='logistic')
parser.add_argument('--normalize', choices=['min-max','z-score','none'], required=True)

logistic_paras = parser.add_argument_group('Logistic params')
logistic_paras.add_argument('--penalty', choices=['l1','l2'], default='l2')
logistic_paras.add_argument('--C', type=float, nargs='+', help='smaller values specify stronger regularization')
logistic_paras.add_argument('--solver', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], default='liblinear')

ridge_paras = parser.add_argument_group('Ridge params')
ridge_paras.add_argument('--alpha', type=float, nargs='+', help='larger values specify stronger regularization')

knn_paras = parser.add_argument_group('KNN params')
knn_paras.add_argument('--k', type=int, default=9, help='number of neighbours')
knn_paras.add_argument('--weights', choices=['uniform', 'distance'], default='uniform')

parser.add_argument('--cv_score', action='store_true', help='whether need the cv accuracy and models for each para')
parser.add_argument('--cv', type=int, default=5, help='K-fold, default k=5')
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')
opt = parser.parse_args()

opt.cv_score = opt.cv_score if opt.model != 'knn' else False

exp_path = util.hyperparam_string_class(opt)
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

if opt.model == 'logistic':
    if not opt.cv_score:
        train_model = LogisticModel(penalty=opt.penalty, C=opt.C, solver=opt.solver, 
                            cv=opt.cv, tol=opt.tol, random_state=opt.random_seed)
    else:
        train_model = []
        for each in opt.C:
            train_model.append(LogisticModel(penalty=opt.penalty, C=opt.C, solver=opt.solver, \
                            cv=each, tol=opt.tol, random_state=opt.random_seed))
elif opt.model == 'ridge':
    if not opt.cv_score:
        train_model = RidgeModel(alpha=opt.alpha, cv=opt.cv)
    else:
        train_model = []
        for each in opt.alpha:
            train_model.append(RidgeModel(alpha=each, cv=opt.cv))
elif opt.model == 'knn':
    train_model = KNNModel(k=opt.k, weights=opt.weights)
else:
    raise ValueError('Unknown classifier name!')

if not opt.cv_score: # do not need to save each model
    train_acc = train_model.train(train_data, train_label)
    train_model.save_model(os.path.join(exp_path, 'train.model'))
    logger.info('Training acc is %.4f' % (train_acc))
    logger.info('Start predicting labels on Test set ...')
    result = train_model(test_data)
    logger.info('Start writing predictions to file %s' % (os.path.join(exp_path,'result.csv')))
    write_csv_result(result, outfile=os.path.join(exp_path,'result.csv'))
    if opt.model in ['ridge', 'logistic'] and opt.cv_score:
        logger.info('Best paras is %s' % (train_model.get_best_paras()))
    if opt.model in ['knn']:
        logger.info('Cross validation score is %.4f' % (train_model.get_cv_accuracy(train_data, train_label, cv=opt.cv)))
else:
    paras_list = opt.alpha if opt.model == 'ridge' else opt.C
    for idx, each_model in enumerate(train_model):
        cv_score = each_model.get_cv_accuracy(train_data, train_label, cv=opt.cv)
        logger.info("CV accuracy for hyperpara=%.4f is %.4f" % (paras_list[idx], cv_score))
        train_acc = each_model.train(train_data, train_label)
        each_model.save_model(os.path.join(exp_path, 'train_%s.model' % (paras_list[idx])))
        logger.info('Training acc for hyperpara=%.4f is %.4f' % (paras_list[idx], train_acc))
        result = each_model(test_data)
        logger.info('Start writing predictions to file %s' % (os.path.join(exp_path,'result_%s.csv' % (paras_list[idx]))))
        write_csv_result(result, outfile=os.path.join(exp_path,'result_%s.csv' % (paras_list[idx])))
