import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def load_dat_htk(filename):
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH",spam)
    veclen = int(sampSize/4)
    fh.seek(12,0)
    dat = np.fromfile(fh, dtype = np.float32)
    dat = dat.reshape(int(len(dat)/veclen),veclen)
    dat = dat.byteswap()
    fh.close()

    return dat

def onehot(x, classes):
    result = np.zeros((len(x), classes), dtype = 'float32')

    for i in range(len(x)):
        result[i, x[i]] = 1.0

    return result

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('Conv1d')!=-1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM')!=-1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

def sort_pad(xs, ts=[], ts_onehot=[], ts_onehot_LS=[], lengths, ts_lengths=[], train =
             False):

