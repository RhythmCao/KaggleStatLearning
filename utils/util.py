#coding=utf8

''' Construct exp directory according to hyperparameters '''
import os

def hyperparam_string(options, tf=False):
    """Hyerparam string."""
    model_path = 'model_%s' % (options.model)
    
    exp_name = ''
    if options.model == 'fnn':
        exp_name += 'layers_%s__' % (str(options.affine_layers).strip('[]').replace(',','_').replace(' ',''))
    elif options.model == 'cnn':
        exp_name += 'channel_%s__' % (str(options.channel).strip('[]').replace(',','_').replace(' ',''))
        exp_name += 'kernel_%s__' % (str(options.kernel_size).strip('[]').replace(',','_').replace(' ',''))
        if options.maxpool_kernel_size != [] and options.maxpool_kernel_size is not None:
            exp_name += 'maxpoolkernel_%s__' % (str(options.maxpool_kernel_size).strip('[]').replace(',','_').replace(' ',''))
    exp_name += 'nonlinear_%s__' % (options.nonlinear)
    exp_name += 'bs_%s__' % (options.batchSize)
    exp_name += 'dropout_%s__' % (options.dropout)
    exp_name += 'batchnorm__' if options.batchnorm else ''
    exp_name += 'optim_%s__' % (options.optim)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'me_%s' % (options.max_epoch)

    if tf:
        return os.path.join('tf_' + model_path, exp_name)
    else:
        return os.path.join(model_path, exp_name)

def hyperparam_string_stat(options):
    """Hyerparam string."""
    model_path = 'model_%s' % (options.model)
    exp_name = ''
    if options.model == 'svm':
        exp_name += 'type_%s__' % (options.type)
        if options.type == 'svc':
            exp_name += 'kernel_%s__' % (options.kernel)
            exp_name += 'C_%s__' % (options.C)
            exp_name += 'gamma_%s__' % (options.gamma)
            exp_name += 'degree_%s__' % (options.degree)
            exp_name += 'coef0_%s__' % (options.coef0)
            exp_name += '%s' % (options.decision_function_shape)
        elif options.type == 'nusvc':
            exp_name += 'kernel_%s__' % (options.kernel)
            exp_name += 'nu_%s__' % (options.nu)
            exp_name += 'gamma_%s__' % (options.gamma)
            exp_name += 'degree_%s__' % (options.degree)
            exp_name += 'coef0_%s__' % (options.coef0)
            exp_name += '%s' % (options.decision_function_shape)
        else:
            exp_name += 'penalty_%s__' % (options.penalty)
            exp_name += 'loss_%s__' % (options.loss)
            exp_name += 'dual_%s__' % (options.dual)
            exp_name += 'C_%s__' % (options.C)
            exp_name += '%s' % (options.decision_function_shape)
    elif options.model == 'lda':
        exp_name += 'type_%s__' % (options.type)
        if options.type == 'lda':
            exp_name += 'solver_%s' % (options.solver)
            exp_name += 'shrinkage_%s' % (options.shrinkage)
        else:
            exp_name += 'reg_%s' % (options.reg_param)
    
    return os.path.join(model_path, exp_name)