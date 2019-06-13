import copy
import codecs
from operator import itemgetter, attrgetter
import itertools
from struct import unpack, pack
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

HIDDEN_SIZE = 320
NUM_LAYERS = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
NUM_CLASSES = int(sys.argv[3])
EOS_ID = int(sys.argv[4])
#NUM_CLASSES_ACT = int(sys.argv[4])
script_file = sys.argv[5]
model_file = sys.argv[6]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wordid = []
with codecs.open('/n/rd32/mimura/e2e/data/script/aps/word.id', 'r',
                 'euc-jp') as f:
    for line in f:
        sp_line = line.split(' ')[0].split('+')
        wordid.append(sp_line[0])

def load_dat(filename):
    fh = open(filename, "rb")
    spam = fh.read(12)
    nSamples, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)
    veclen = int(sampSize/4)
    fh.seek(12, 0)
    dat = np.fromfile(fh, dtype=np.float32)
    dat = dat.reshape(int(len(dat)/veclen), veclen)
    dat = dat.byteswap()
    fh.close()
    return dat

def onehot(x, classes): 
    result = np.zeros((len(x),classes),dtype = 'float32')

    for i in range(len(x)):
        result[i,x[i]] = 1

    return result

def sort_pad(xs, lengths):
    arg_lengths = np.argsort(np.array(lengths))[::-1].tolist()
    maxlen = max(lengths)
    xs_tensor = torch.zeros((BATCH_SIZE, maxlen, 120),dtype = torch.float32,requires_grad=True).to(DEVICE)

    for i, i_sort in enumerate(arg_lengths):
        xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
        
    return xs_tensor

def load_model(model_file):
    if torch.cuda.is_available():
        model_state = torch.load(model_file)
    else:
        model_state = torch.load(model_file, map_location = 'cpu')
    is_multi_loading = True if torch.cuda.device_count()>1 else False

    is_multi_loaded = True if 'module' in list(model_state.keys())[0] else False

    if is_multi_loaded is is_multi_loading:
        return model_state

    elif is_multi_loaded is False and is_multi_loading is True:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state['module.'+key]=model.state[key]

        return new_model_state

    elif is_multi_loaded is True and is_multi_loading is False:
        new_model_state = {}
        for key in model_state.keys():
            new_model_state[key[7:]] = model_state[key]

        return new_model_state

    else:
        print('ERROR in load model')
        sys.exit(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #encoder is BiRnn
        self.bilstm = nn.LSTM(input_size=120, dropout = 0.2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True, bidirectional=True)  # batch,seq,feature
        # attention is w(ax+by+c) output is batch*seqlen*2hidden
        self.attw = nn.Linear(HIDDEN_SIZE*2, 1, bias=False)
        self.atts = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE*2,bias=False)
        self.atth = nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE*2)

        self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)
        self.fe = nn.Linear(10, HIDDEN_SIZE * 2, bias=False)

        #decoder is RNN
        self.sy = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE,bias=False)
        self.gy = nn.Linear(HIDDEN_SIZE*2,HIDDEN_SIZE)
        self.yy = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        
        "next state of s in attention model"
        self.ss = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE*4,bias=False)
        self.gs = nn.Linear(HIDDEN_SIZE*2,HIDDEN_SIZE*4)
        self.ys = nn.Linear(NUM_CLASSES,HIDDEN_SIZE*4)

        "word level encoder"
        #self.sn = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE,bias=False)
        #self.nn = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        #self.wn = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)

        "segment level decoder"
        #self.segseg = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        #self.segy = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        #self.yres = nn.Linear(HIDDEN_SIZE,NUM_CLASSES_ACT)

    def forward(self, data, length):
        total_length = data.size(1)
        datapack = torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first = True)

        h, _ = self.bilstm(datapack)
        hpack, flength = torch.nn.utils.rnn.pad_packed_sequence(h,total_length=total_length,batch_first=True)

        batchs = hpack.size(0)
        features = hpack.size(1)

        e_mask = torch.ones((batchs, features, 1), device=DEVICE, requires_grad=False)
        
        token_beam_sel = [([], [], 0.0, (torch.zeros((BATCH_SIZE,
                                                      HIDDEN_SIZE),
                                                     device=DEVICE,
                                                     requires_grad=False),
                                         torch.zeros((BATCH_SIZE, HIDDEN_SIZE),
                                                     device=DEVICE,
                                                     requires_grad=False)))]

        for i, tmp in enumerate(lengths):
            if tmp < features:
                e_mask.data[i,tmp:] = 0.0
                

        BEAM_WIDTH = 4
        for num_labels in range(200):
            #tmpconv = self.F_conv1d(alpha)
            #tmpconv = tmpconv.transpose(1, 2)[:,:features,:]
            #tmpconv = self.fe(tmpconv)
            token_beam_all = []


            for current_token in token_beam_sel:
                cur_seq, cur_bottle, cur_seq_score, (c,s) = current_token
                
                if len(cur_bottle)!=0:
                    tmp_bottle = copy.deepcopy(cur_bottle)
                else:
                    tmp_bottle = cur_bottle

                e = torch.tanh(self.atts(s).unsqueeze(1) +self.atth(hpack))
                e = self.attw(e)
    
                e_nonlin = (e - e.max(1)[0].unsqueeze(1)).exp()
                e_nonlin = e_nonlin * e_mask
    
                alpha = e_nonlin/e_nonlin.sum(dim=1,keepdim = True)
    
                g = (alpha * hpack).sum(dim = 1)
                y = self.yy(torch.tanh(self.sy(s) + self.gy(g)))
                bottle_feats = torch.tanh(self.sy(s)+self.gy(g)).detach().cpu().numpy()
                tmp_bottle.append(bottle_feats)

                y = F.log_softmax(y, dim=1)
                tmpy = y.clone()

                for k in range(BEAM_WIDTH):
                    best, bestidx = tmpy.data.max(1)
                    print(wordid[bestidx.item()])
                    bestidx = bestidx.item()

                    tmpseq = cur_seq.copy()
                    tmpseq.append(bestidx)

                    tmpscore = cur_seq_score + tmpy.data[0][bestidx]
                    tmpy.data[0][bestidx] = -10000000000.0
                    target_for_t_estimated = torch.zeros((1,NUM_CLASSES),
                                                         device = DEVICE,
                                                         requires_grad = False)
                    target_for_t_estimated.data[0][bestidx] = 1.0

                    "update the state of s"
                    initial = torch.tanh(self.ss(s) + self.gs(g) + self.ys(y))
                    batch = initial.size(0)
                    ingate, forgetgate, cellgate, outgate = initial.chunk(4, 1)
                    half = 0.5
                    ingate = torch.tanh(ingate * half) * half + half
                    forgetgate = torch.tanh(forgetgate * half) * half + half
                    cellgate = torch.tanh(cellgate)
                    outgate = torch.tanh(outgate * half) * half + half
                    c_next = (forgetgate * c) + (ingate * cellgate)
                    h = outgate * torch.tanh(c_next)
    
                    tmps = h
                    tmpc = c_next

                    token_beam_all.append((tmpseq,tmp_bottle,tmpscore,(tmpc,tmps)))
            
            sorted_token_beam_all = sorted(token_beam_all, key=itemgetter(2),
                                          reverse = True)
            token_beam_sel = sorted_token_beam_all[:BEAM_WIDTH]
            if token_beam_sel[0][0][-1] == EOS_ID:
                for character in token_beam_sel[0][0]:
                    print(wordid[character], end=" ")

            "update the state in dialog act word level"
            #n = torch.tanh(self.sn(s) + self.nn(n))

            "update the state in dialog act segment level"
            #if gt[labels] == "/":
            #    seg = torch.tanh(self.segseg(seg) + self.nseg(n))
            #    result = self.yres(torch.tanh(self.segy(seg))
            #    Youtput[:step] = result

        print()
        sys.stdout.flush()
        return token_beam_sel

net = Net()
net.to(DEVICE)
criterion = nn.CrossEntropyLoss()
net.eval()

model_state = load_model(model_file)
net.load_state_dict(model_state)

file_name = []
with open(script_file) as f:
    for line in f:
        file_name.append(line)

num_mb = len(file_name) // BATCH_SIZE
maxlen = 0
for i in range(num_mb):
    xs = []
    lengths = []
    for j in range(BATCH_SIZE):
        s = file_name[i*BATCH_SIZE+j].strip()
        read_file = s
        if '.htk' in read_file:
            cpudat = load_dat(read_file)
            cpudat = cpudat[:,:40]
        elif '.npy' in read_file:
            cpudat = np.load(read_file)

        newlen = int(cpudat.shape[0]/3)
        cpudat = cpudat[:3*newlen,:]
        cpudat = np.reshape(cpudat, (newlen,3,40))
        cpudat = np.reshape(cpudat, (newlen, 120)).astype(np.float32)
            
#            print(newlen)
        lengths.append(newlen)
        xs.append(cpudat)
    xs = sort_pad(xs,lengths)
    hs = net(xs,lengths)
