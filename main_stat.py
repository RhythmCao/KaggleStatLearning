#coding=utf8
import argparse, random, os, sys, time
import numpy as np
import logging
import gc

from utils.data_utils import load_train_data, load_test_data, write_csv_result
import utils.util as util

from models.svm import SVMModel, SVCModel, NuSVCModel, LinearSVCModel
from models.lda import LDAModel

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='exp', help='Where to store samples and models')
parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')
parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
parser.add_argument('--read_model', required=False, help='Only test: read model from this file')
parser.add_argument('--out_path', required=False, help='Only test: out_path')

parser.add_argument('--model', choices=['svm','lda'], default='svm')
parser.add_argument('--split_ratio', type=float, help='split train data into dev data and train data')

svm_paras = parser.add_argument_group('SVM model parameters')
svm_paras.add_argument('--type', choices=['svc','nusvc','linearsvc'], default='svc')
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
lda_paras.add_argument()
lda_paras.add_argument()
lda_paras.add_argument()

parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')

opt = parser.parse_args()
if not opt.testing:
    exp_path = util.hyperparam_string_stat(opt)
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

if opt.model == 'svm':
    if opt.type == 'svc':
        train_model = SVCModel(kernel=opt.kernel, C=opt.C, degree=opt.degree, gamma=opt.gamma, coef0=opt.coef0,
                        tol=opt.tol, decision_function_shape=opt.decision_function_shape, random_state=opt.random_seed)
    elif opt.type == 'nusvc':
        train_model = NuSVCModel(kernel=opt.kernel, nu=opt.nu, degree=opt.degree, gamma=opt.gamma, coef0=opt.coef0,
                        tol=opt.tol, decision_function_shape=opt.decision_function_shape, random_state=opt.random_seed)
    else:
        train_model = LinearSVCModel(penalty=opt.penalty, loss=opt.loss, dual=opt.dual, tol=opt.tol, 
                        C=opt.C, multi_class=opt.decision_function_shape, random_state=opt.random_seed)
elif opt.model == 'lda':
    pass

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