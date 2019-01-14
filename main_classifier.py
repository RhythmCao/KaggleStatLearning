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
parser.add_argument('--split_ratio', type=float, help='split train data into dev data and train data')

logistic_paras = parser.add_argument_group('Logistic params')
logistic_paras.add_argument('--penalty', choices=['l1','l2'], default='l2')
logistic_paras.add_argument('--C', type=float, nargs='+', help='smaller values specify stronger regularization')
logistic_paras.add_argument('--solver', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], default='liblinear')

ridge_paras = parser.add_argument_group('Ridge params')
ridge_paras.add_argument('--alpha', type=float, nargs='+', help='larger values specify stronger regularization')

knn_paras = parser.add_argument_group('KNN params')
knn_paras.add_argument('--k', type=int, default=9, help='number of neighbours')
knn_paras.add_argument('--weights', choices=['uniform', 'distance'], default='uniform')

parser.add_argument('--cv', type=int, default=0)
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')
opt = parser.parse_args()

opt.cv = None if opt.cv <= 0 else opt.cv
if not opt.testing:
    exp_path = util.hyperparam_string_class(opt)
    exp_path = os.path.join(opt.experiment, exp_path)
else:
    exp_path = opt.out_path
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

logFormatter = logging.Formatter('%(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
if opt.testing:
    fileHandler = logging.FileHandler('%s/log_test_%s.txt' % (exp_path, opt.split_ratio), mode='w')
else:
    fileHandler = logging.FileHandler('%s/log_train_%s.txt' % (exp_path, opt.split_ratio), mode='w') # override written
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
if not opt.testing:
    if opt.split_ratio > 0:
        train_data, train_label, dev_data, dev_label = load_train_data(split_ratio=opt.split_ratio)
    else:
        train_data, train_label = load_train_data(split_ratio=0)
        dev_data = None
test_data = load_test_data()
logger.info("Prepare data ... cost %.4fs" % (time.time()-start_time))

if opt.model == 'logistic':
    train_model = LogisticModel(penalty=opt.penalty, C=opt.C, solver=opt.solver, 
                            cv=opt.cv, tol=opt.tol, random_state=opt.random_seed)
elif opt.model == 'ridge':
    train_model = RidgeModel(alpha=opt.alpha, cv=opt.cv)
elif opt.model == 'knn':
    train_model = KNNModel(k=opt.k, weights=opt.weights)
else:
    raise ValueError('Unknown classifier name!')

if not opt.testing:
    train_model.train(train_data, train_label)
    train_model.save_model(os.path.join(exp_path, 'train_%s.model' % (opt.split_ratio)))
    _, train_acc = train_model(train_data, train_label)
    logger.info('Training acc is %.4f' % (train_acc))
    if dev_data is not None:
        _, dev_acc = train_model(dev_data, dev_label)
        logger.info('Dev acc is %.4f' % (dev_acc))
    result = train_model(test_data)
    write_csv_result(result, outfile=os.path.join(exp_path,'result_%s.csv' % (opt.split_ratio)))
else:
    train_model.load_model(opt.read_model+'.model')
    result = train_model(test_data)
    write_csv_result(result, os.path.join(exp_path,'result.csv'))