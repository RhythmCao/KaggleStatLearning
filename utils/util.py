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