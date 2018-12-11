#coding=utf8
import csv
import os,sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
DATAROOT='data/'

def load_train_data(split_ratio=0.2):
    train_file = os.path.join(DATAROOT,'train.csv')
    data = np.loadtxt(open(train_file,'rb'), delimiter=',', dtype=np.float, skiprows=1)[:,1:-1]
    label = np.loadtxt(open(train_file,'rb'), delimiter=',', dtype=np.int, skiprows=1, usecols=-1)
    if split_ratio > 0:
        # split according to labels, make sure the number of different labels are the same
        z_x_y = list(zip(data,label))
        label_data_dict, train_dict, dev_dict = dict(), dict(), dict()
        for x, y in z_x_y:
            if y not in label_data_dict:
                label_data_dict[y] = list()
            label_data_dict[y].append(x)
        train_data, train_label, dev_data, dev_label = list(), list(), list(), list()
        for y in label_data_dict:
            x = label_data_dict[y]
            data_index = np.arange(len(x))
            np.random.shuffle(data_index)
            dev_size = int(len(x)*split_ratio)
            train_data.extend([x[idx] for idx in data_index[dev_size:]])
            train_label.extend([y]*(len(x)-dev_size))
            dev_data.extend([x[idx] for idx in data_index[:dev_size]])
            dev_label.extend([y]*dev_size)
        return np.vstack(train_data), np.array(train_label), np.vstack(dev_data), np.array(dev_label)
    return data, label

def load_test_data():
    test_file = os.path.join(DATAROOT,'test.csv')
    data = np.loadtxt(open(test_file,'rb'), delimiter=',', dtype=np.float, skiprows=1)[:,1:]
    return data

def write_csv_result(result, outfile=None):
    if outfile is None:
        outfile = os.path.join(DATAROOT, outfile)
    if len(result.shape) == 1:
        result = np.hstack([np.arange(len(result))[:,np.newaxis], result[:, np.newaxis]])
    np.savetxt(outfile, result, delimiter=',', header='id,categories', comments='',fmt="%d,%d")


if __name__ == '__main__':

    train_data, train_label, dev_data, dev_label = load_train_data(split_ratio=0.2)
    print(str(train_data.shape),str(train_label.shape),str(dev_data.shape),str(dev_label.shape))
    data = load_test_data()
    print(str(data.shape))
