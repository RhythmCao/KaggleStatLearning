#coding=utf8
import argparse, random, os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import gc

from utils.data_iter import ImageDataset, collate_fn_for_data
from utils.data_utils import load_train_data, load_test_data, write_csv_result
import utils.util as util
from models.fnn import FNNModel

FEATURE_SIZE = 4096
NUM_CLASSES = 12

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='exp', help='Where to store samples and models')
parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')
parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
parser.add_argument('--read_model', required=False, help='Only test: read model from this file')
parser.add_argument('--out_path', required=False, help='Only test: out_path')

parser.add_argument('--model', choices=['fnn','cnn','svm','lda'], default='fnn')
parser.add_argument('--nonlinear', choices=['relu','tanh','sigmoid'], default='relu')
parser.add_argument('--affine_layers', type=int, nargs='+')

parser.add_argument('--deviceId', type=int, default=-1, help='train model on ith gpu. -1:cpu, 0:auto_select')
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.05, help='decay of learning rate')
parser.add_argument('--l2', type=float, default=0, help='weight decay (L2 penalty)')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate at each non-recurrent layer')
parser.add_argument('--batchnorm', action='store_true', help='whether use batch normalization')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=0, help='input batch size in decoding')
parser.add_argument('--init_weight', type=float, default=0.2, help='all weights will be set to [-init_weight, init_weight] during initialization')
parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
parser.add_argument('--optim', default='sgd', choices=['adadelta', 'sgd', 'adam', 'rmsprop'], help='choose an optimizer')

opt = parser.parse_args()

if opt.test_batchSize == 0:
    opt.test_batchSize = opt.batchSize

if not opt.testing:
    exp_path = util.hyperparam_string(opt)
    exp_path = os.path.join(opt.experiment, exp_path)
else:
    exp_path = opt.out_path
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

logFormatter = logging.Formatter('%(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
if opt.testing:
    fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
else:
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

if opt.deviceId >= 0:
    import utils.gpu_selection as gpu_selection
    if opt.deviceId > 0:
        opt.deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu(assigned_gpu_id=opt.deviceId - 1)
    elif opt.deviceId == 0:
        opt.deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu()
    logger.info("Valid GPU list: %s ; GPU %d (%s) is auto selected." % (valid_gpus, opt.deviceId, gpu_name))
    torch.cuda.set_device(opt.deviceId)
    opt.device = torch.device("cuda") # is equivalent to torch.device('cuda:X') where X is the result of torch.cuda.current_device()
else:
    logger.info("CPU is used.")
    opt.device = torch.device("cpu")

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
if torch.cuda.is_available():
    if opt.device.type != 'cuda':
        logger.info("WARNING: You have a CUDA device, so you should probably run with --deviceId [1|2|3]")
    else:
        torch.cuda.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)

# load dataset
start_time = time.time()
if not opt.testing:
    train_data, train_label, dev_data, dev_label = load_train_data(split_ratio=0.2)
    train_dataset = ImageDataset(train_data, train_label)
    dev_dataset = ImageDataset(dev_data, dev_label)
    train_iter = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, collate_fn=collate_fn_for_data, num_workers=4)
    dev_iter = DataLoader(dev_dataset, batch_size=opt.test_batchSize, shuffle=False, collate_fn=collate_fn_for_data, num_workers=4)
test_data = load_test_data()
test_dataset = ImageDataset(test_data)
test_iter = DataLoader(test_dataset, batch_size=opt.test_batchSize, shuffle=False, collate_fn=collate_fn_for_data, num_workers=4)
logger.info("Prepare train, dev and test data ... cost %.4fs" % (time.time()-start_time))

if opt.model == 'fnn':
    train_model = FNNModel(FEATURE_SIZE, NUM_CLASSES, layers=opt.affine_layers, dropout=opt.dropout, nonlinear=opt.nonlinear, batchnorm=opt.batchnorm, device=opt.device)
    train_model = train_model.to(opt.device)
    if not opt.testing:
        train_model.init_weight(opt.init_weight)
        logger.info("Train model init weights ... ...")
    if opt.read_model:
        train_model.load_model(opt.read_model+'.model')
        logger.info("Load model from %s ..." % (opt.read_model))

# loss function
loss_function = nn.NLLLoss(reduction='sum')

# optimizer
params = []
for model in (train_model, ):
    params += list(model.parameters())
if opt.optim.lower() == 'sgd':
    optimizer = optim.SGD(params, lr=opt.lr, weight_decay=opt.l2, momentum=0.5)
elif opt.optim.lower() == 'adam':
    optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.l2) # (beta1, beta2)
elif opt.optim.lower() == 'adadelta':
    optimizer = optim.Adadelta(params, rho=0.95, lr=1.0, weight_decay=opt.l2)
elif opt.optim.lower() == 'rmsprop':
    optimizer = optim.RMSprop(params, lr=opt.lr, weight_decay=opt.l2)

#################################################################
################## Training and Decoding ########################
#################################################################

def decode(data_iter, eval_model, write_result=None, add_loss=False):
    count, eval_loss, has_label = 0, [], True
    result = []
    for j, data in enumerate(data_iter):
        if type(data) == tuple and len(data) == 2:
            data, label = data
            label = label.to(opt.device)
        else:
            label = None
            has_label = False
        data = data.to(opt.device)
        scores = eval_model(data)
        pred = scores.argmax(dim=1)
        if label is not None:
            count += torch.sum(pred==label).item()
            if add_loss:
                eval_loss.append(loss_function(scores, label))
        result.append(pred.cpu().numpy())
    if write_result is not None:
        write_csv_result(np.concatenate(pred, axis=0), outfile=write_result)
    if has_label:
        eval_acc = count*100.0/len(data_iter)
        if add_loss:
            eval_loss = np.sum(eval_loss, axis=0)
            return eval_acc, eval_loss
        return eval_acc
    else:
        return

if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    best_result = {"epoch_loss":[], "best_dev_acc":0, "best_test_acc":0, "best_dev_loss":1e8}
    for i in range(opt.max_epoch):
        start_time = time.time()
        losses, count = [], 0
        train_model.train()
        if opt.optim.lower() == 'sgd':
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr / (1 + opt.lr_decay * i)
        nsentences = len(train_data)
        piece_sentences = int(nsentences * 0.1 / opt.batchSize) * opt.batchSize
        for j, (data, label) in enumerate(train_iter):
            data, label = data.to(opt.device), label.to(opt.device)
            # print(data.dtype,label.dtype,str(data.shape),str(label.shape))
            optimizer.zero_grad()
            scores = train_model(data)
            loss = loss_function(scores, label)
            loss.backward()
            if opt.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), opt.max_norm, norm_type=2)
            optimizer.step()
            losses.append(loss.item())
            count += torch.sum(scores.argmax(dim=1)==label).item()
            if j % piece_sentences == 0:
                print('[learning] epoch %i >> %2.2f%%'%(i,(j+opt.batchSize)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-start_time), end='')
                sys.stdout.flush()
        print('[learning] epoch %i >> %3.2f%%'%(i,100),'completed in %.2f (sec) <<'%(time.time()-start_time))
        epoch_loss = np.sum(losses, axis=0)
        train_acc = count*100.0/len(train_data)
        logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f \t Acc: %.2f%%' % (i, time.time() - start_time, epoch_loss, train_acc))
        best_result['epoch_loss'].append((epoch_loss, train_acc))
        gc.collect()

        # Evaluation phase
        train_model.eval()
        start_time = time.time()
        accuracy_v, loss_val = decode(dev_iter, train_model, add_loss=True)
        logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tAcc : %.4f' % (i, time.time() - start_time, loss_val, accuracy_v))
        start_time = time.time()
        decode(test_iter, train_model, write_result=os.path.join(exp_path, 'test.iter'+str(i)))
        logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tAcc : %.4f' % (i, time.time() - start_time, accuracy_t))
        if accuracy_v > best_result['best_dev_acc'] or ( accuracy_v == best_result['best_dev_acc'] and loss_val < best_result['best_dev_loss'] ):
            train_model.save_model(os.path.join(exp_path, 'train.model'))
            best_result['epoch'] = i
            best_result['best_dev_acc'], best_result['best_dev_loss'] = accuracy_v, loss_val
            logger.info('NEW BEST:\tEpoch : %d\tBest Valid Acc : %.4f' % (i, accuracy_v))
    logger.info('BEST RESULT: \tEpoch : %d\tBest Valid (Loss: %.5f Acc : %.4f)' % (best_result['epoch'], best_result['best_dev_loss'], best_result['best_dev_acc']))
else:    
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    test_model = train_model
    test_model.eval()
    start_time = time.time()
    decode(test_iter, train_model, write_result=os.path.join(exp_path,'result.csv'))
    logger.info('Evaluation:\tTime : %.4fs\tAcc : %.4f' % (time.time() - start_time, accuracy))