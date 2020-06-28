import numpy as np
import scipy.io as spio
import sys
import matplotlib.pyplot as plt
sys.path.append('../NCKHECG')
def read_mitbih(filename, max_time=100, classes= ['F', 'N', 'S', 'V', 'Q'], max_nlabel=100, trainset=1):
    def normalize(data):
        data = np.nan_to_num(data)  # removing NaNs and Infs
        data = data - np.mean(data)
        data = data / np.std(data)
        return data

    # read data
    data = []
    samples = spio.loadmat(filename + ".mat")
    if trainset == 1: #DS1
        samples = samples['s2s_mitbih_DS1']

    else: # DS2
        samples = samples['s2s_mitbih_DS2']

    values = samples[0]['seg_values']
    labels = samples[0]['seg_labels']
    items_len = len(labels)
    num_annots = sum([item.shape[0] for item in values])

    n_seqs = num_annots / max_time
    #  add all segments(beats) together
    l_data = 0
    for i, item in enumerate(values):
        l = item.shape[0]
        for itm in item:
            if l_data == n_seqs * max_time:
                break
            data.append(itm[0])
            l_data = l_data + 1

    #  add all labels together
    l_lables  = 0
    t_lables = []
    for i, item in enumerate(labels):
        if len(t_lables)==n_seqs*max_time:
            break
        item= item[0]
        for lebel in item:
            if l_lables == n_seqs * max_time:
                break
            t_lables.append(str(lebel))
            l_lables = l_lables + 1

    del values
    data = np.asarray(data)
    shape_v = data.shape
    data = np.reshape(data, [shape_v[0], -1])
    t_lables = np.array(t_lables)
    _data  = np.asarray([],dtype=np.float64).reshape(0,shape_v[1])
    _labels = np.asarray([],dtype=np.dtype('|S1')).reshape(0,)
    for cl in classes:
        _label = np.where(t_lables == cl)
        permute = np.random.permutation(len(_label[0]))
        _label = _label[0][permute[:max_nlabel]]
        # _label = _label[0][:max_nlabel]
        # permute = np.random.permutation(len(_label))
        # _label = _label[permute]
        _data = np.concatenate((_data, data[_label]))
        _labels = np.concatenate((_labels, t_lables[_label]))

    data = _data[:(len(_data)// max_time) * max_time, :]
    _labels = _labels[:(len(_data) // max_time) * max_time]

    # data = _data
    #  split data into sublist of 100=se_len values
    data = [data[i:i + max_time] for i in range(0, len(data), max_time)]
    labels = [_labels[i:i + max_time] for i in range(0, len(_labels), max_time)]
    # shuffle
    permute = np.random.permutation(len(labels))
    data = np.asarray(data)
    labels = np.asarray(labels)
    data= data[permute]
    labels = labels[permute]

    print('Records processed!')

    return data, labels
if __name__=='__main__':
    X_train, y_train = read_mitbih(filename='Data/s2s_mitbih_aami_DS1DS2', max_time=10, classes=['N', 'S','V','F'], max_nlabel=50000,trainset=1)
    X_train=np.reshape(X_train,(X_train.shape[0]*X_train.shape[1],1,X_train.shape[2]))
    y_train=np.reshape(y_train,(y_train.shape[0]*y_train.shape[1],))
    print(len(np.where(y_train=='N')[0]))
    print(len(np.where(y_train=='S')[0]))
    print(len(np.where(y_train=='V')[0]))
    print(len(np.where(y_train=='F')[0]))