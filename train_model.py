import copy
import itertools
from struct import unpack, pack
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

HIDDEN_SIZE = 256
NUM_LAYERS = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
NUM_CLASSES = int(sys.argv[3])
NUM_CLASSES_ACT = int(sys.argv[4])
script_file = sys.argv[5]
save_dir = sys.argv[6]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dat(filename):
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
        result[i, int(x[i])] = 1.0 
    
    return result

def onehot_da(x, classes):
    result = np.zeros(classes, dtype = 'float32')
    result[int(x)] = 1.0

    return result

def sort_pad(xs, ts, da, da_onehot, da_onehot_LS, ts_onehot, ts_onehot_LS, lengths, ts_lengths):
    arg_lengths = np.argsort(np.array(lengths))[::-1].tolist()
    maxlen = max(lengths)
    ts_maxlen = max(ts_lengths)
    xs_tensor = torch.zeros((BATCH_SIZE, maxlen, 120),dtype = torch.float32,requires_grad=True).to(DEVICE)
    ts_onehot_tensor = torch.zeros((BATCH_SIZE, ts_maxlen, NUM_CLASSES),dtype=torch.float32,requires_grad=True).to(DEVICE)
    ts_onehot_LS_tensor = torch.zeros((BATCH_SIZE, ts_maxlen, NUM_CLASSES),dtype=torch.float32,requires_grad=True).to(DEVICE)
    lengths_tensor = torch.zeros((BATCH_SIZE),dtype=torch.float32).to(DEVICE)
    ts_result = []
    da_result = []
    da_onehot_tensor = torch.zeros((BATCH_SIZE,NUM_CLASSES_ACT),dtype=torch.float32,requires_grad=True).to(DEVICE)
    da_onehot_LS_tensor = torch.zeros((BATCH_SIZE,NUM_CLASSES_ACT),dtype=torch.float32, requires_grad=True).to(DEVICE)

    for i, i_sort in enumerate(arg_lengths):
        xs_tensor.data[i, 0:lengths[i_sort]] = torch.from_numpy(xs[i_sort])
        ts_onehot_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot[i_sort])
        ts_onehot_LS_tensor.data[i, 0:ts_lengths[i_sort]] = torch.from_numpy(ts_onehot_LS[i_sort])
        ts_result.append(Variable(torch.from_numpy(ts[i_sort]).to(DEVICE).long()))
        lengths_tensor.data[i] = lengths[i_sort]
        da_onehot_tensor.data[i] = torch.from_numpy(da_onehot[i_sort].squeeze())
        da_onehot_LS_tensor.data[i] =torch.from_numpy(da_onehot_LS[i_sort])
        #da_result.append(Variable(torch.from_numpy(np.array(da[i_sort]).astype(np.int32)).to(DEVICE).long()))

        
    return xs_tensor, ts_result, da, da_onehot_tensor,da_onehot_LS_tensor, ts_onehot_tensor, ts_onehot_LS_tensor, lengths_tensor

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Linear')!=-1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM')!=-1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #encoder is BiRnn
        self.bilstm = nn.LSTM(input_size=120, dropout = 0.2,
                              hidden_size=HIDDEN_SIZE, num_layers=3, 
                              batch_first=True, bidirectional=True)  # batch,seq,feature
        # attention is w(ax+by+c) output is batch*seqlen*2hidden
        self.attw = nn.Linear(HIDDEN_SIZE*2, 1, bias=False)
        self.atts = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE*2,bias=False)
        self.atth = nn.Linear(HIDDEN_SIZE*2, HIDDEN_SIZE*2)

        #self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)
        #self.fe = nn.Linear(10, HIDDEN_SIZE * 2, bias=False)

        #decoder is RNN
        self.sy = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE,bias=False)
        self.gy = nn.Linear(HIDDEN_SIZE*2,HIDDEN_SIZE)
        self.yy = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        
        "next state of s in attention model"
        self.ss = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE*4,bias=False)
        self.gs = nn.Linear(HIDDEN_SIZE*2,HIDDEN_SIZE*4)
        self.ys = nn.Linear(NUM_CLASSES,HIDDEN_SIZE*4)

        "word level decoder of DA classification"
        self.sn = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE,bias=False)
        self.nn = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.wn = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.dan = nn.Linear(NUM_CLASSES_ACT, HIDDEN_SIZE)
        self.nda = nn.Linear(HIDDEN_SIZE, NUM_CLASSES_ACT)

        #"segment level decoder"
        #self.segseg = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        #self.segy = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        #self.yres = nn.Linear(HIDDEN_SIZE,NUM_CLASSES_ACT)

    def forward(self, data, length, gt, da):
        total_length = data.size(1)
        datapack = torch.nn.utils.rnn.pack_padded_sequence(data, length, batch_first = True)

        h, _ = self.bilstm(datapack)
        hpack, flength = torch.nn.utils.rnn.pad_packed_sequence(h,total_length=total_length,batch_first=True)

        batchs = hpack.size(0)
        features = hpack.size(1)
        labels = gt.size(1)
        labels_da = da.size(1)
        #print(da.size(0), labels_da)
        #print(batchs)
        #print(features)

        e_mask = torch.ones((batchs, features, 1), device=DEVICE, requires_grad=False)
        s = torch.cuda.FloatTensor(batchs,HIDDEN_SIZE).zero_()
        c = torch.cuda.FloatTensor(batchs,HIDDEN_SIZE).zero_()
        n = torch.zeros((batchs, HIDDEN_SIZE), device = DEVICE, requires_grad=False)
        youtput = torch.cuda.FloatTensor(batchs,labels,NUM_CLASSES).zero_()
        #alpha = torch.zeros((batchs, 1, features), device=DEVICE, requires_grad=False)
        Youtput = torch.zeros((batchs, labels, NUM_CLASSES_ACT),
                              device=DEVICE, requires_grad = False)

        for i, tmp in enumerate(flength):
            if tmp < features:
                e_mask.data[i,tmp:] = 0.0

        "attention calculation"
        for i in range(labels):
            #tmpconv = self.F_conv1d(alpha)
            #tmpconv = tmpconv.transpose(1, 2)[:,:features,:]
            #tmpconv = self.fe(tmpconv)
            e = torch.tanh(self.atts(s).unsqueeze(1) +self.atth(hpack))
            e = self.attw(e)

            e_nonlin = (e - e.max(1)[0].unsqueeze(1)).exp()
            e_nonlin = e_nonlin * e_mask

            alpha = e_nonlin/e_nonlin.sum(dim=1,keepdim = True)

            g = (alpha * hpack).sum(dim = 1)
            y = self.yy(torch.tanh(self.sy(s) + self.gy(g)))
            youtput[:,i] = y
            Youtput[:,i] = self.nda(n)
            n = torch.tanh(self.sn(s)+self.nn(n)+self.dan(da))
            n = self.wn(n)

            "update the state of s"
            initial = self.ss(s) + self.gs(g) + self.ys(gt[:,i,:])
            batch = initial.size(0)
            ingate, forgetgate, cellgate, outgate = initial.chunk(4, 1)
            half = 0.5
            ingate = torch.tanh(ingate * half) * half + half
            forgetgate = torch.tanh(forgetgate * half) * half + half
            cellgate = torch.tanh(cellgate)
            outgate = torch.tanh(outgate * half) * half + half
            c_next = (forgetgate * c) + (ingate * cellgate)
            h = outgate * torch.tanh(c_next)
            s = h
            c = c_next

           # "encode the state in dialog act word level"
           # n = torch.tanh(self.sn(s) + self.nn(n))

           # "update the state in dialog act segment level"
           # if gt[i] == 27287:
           #     seg = torch.tanh(self.segseg(da[i]) + self.nseg(n))
           #     daresult = self.yres(torch.tanh(self.segy(seg)))
           #     Youtput[:,i] = daresult
           # else:
           #     Youtput[:,i] = -1

        return youtput,Youtput

net = Net()
net.apply(init_weight)
net.to(DEVICE)
criterion = nn.CrossEntropyLoss()
net.train()
optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

os.makedirs(save_dir, exist_ok=True)

for epoch in range(0,40):
    file_name = []
    linenum = -1
    with open(script_file) as f:
        for line in f:
            linenum+=1
            if linenum != 0:
                file_name.append(line)

    num_mb = len(file_name) // BATCH_SIZE
    maxlen = 0
    scheduler.step(epoch)

    for i in range(num_mb):
        xs = []
        ts = []
        da = []
        da_onehot = []
        da_onehot_LS = []
        ts_onehot = []
        ts_onehot_LS = []
        lengths = []
        ts_lengths = []
        # make batch
        for j in range(BATCH_SIZE):
            s = file_name[i*BATCH_SIZE+j].strip()
            _, read_file, _, _, _, _, dialog_act, _, target, _ = s.split('\t')
            if '.htk' in read_file:
                cpudat = load_dat(read_file)
                cpudat = cpudat[:,:40]
            elif '.npy' in read_file:
                cpudat = np.load(read_file)

            if '.htk' in read_file:
                newlen = int(cpudat.shape[0]/3)
                cpudat = cpudat[:3*newlen,:]
                cpudat = np.reshape(cpudat, (newlen,3,40))
                cpudat = np.reshape(cpudat, (newlen, 120)).astype(np.float32)
            
#            print(newlen)
            lengths.append(int(cpudat.shape[0]))
            xs.append(cpudat)
            cpulab = np.array([int(i) for i in target.split(' ')],
                              dtype=np.int32)
            cpulab_onehot = onehot(cpulab, NUM_CLASSES)
            dialog_act_onehot = onehot_da(dialog_act, NUM_CLASSES_ACT)
            da_onehot_LS.append(0.9*dialog_act_onehot+0.1*1.0/NUM_CLASSES_ACT)
            da.append(dialog_act)
            da_onehot.append(dialog_act_onehot)
            ts.append(cpulab)
            ts_onehot.append(cpulab_onehot)
            ts_lengths.append(len(cpulab))
            ts_onehot_LS.append(0.9*onehot(cpulab, NUM_CLASSES)+0.1*1.0/NUM_CLASSES)

        xs, ts, da, da_onehot, da_onehot_LS, ts_onehot, ts_onehot_LS, lengths = sort_pad(xs, ts, da, da_onehot, da_onehot_LS,ts_onehot, ts_onehot_LS, lengths, ts_lengths)

        youtputV, YoutputV = net(xs, lengths, ts_onehot, da_onehot)
        loss = 0.0
        loss1 = 0.0
        loss2 = 0.0
        for k in range(BATCH_SIZE):
            num_labels = ts[k].size(0)
            #print(YoutputV[k].size())
            loss1 += -(F.log_softmax(youtputV[k][:num_labels], dim=1) *
                      ts_onehot_LS[k][:num_labels]).sum() / num_labels *0.5
            loss2 += -(F.log_softmax(YoutputV[k][num_labels-1],dim=0) * da_onehot_LS[k]).sum()*0.5
            
        loss = loss1+loss2
        #print("loss1 =", loss1.item())
        #print("loss2 =", loss2.item())
        #print(i)
        print("EPOCH {} loss =".format(epoch), loss.item())
        sys.stdout.flush()
        optimizer.zero_grad()
        # backward
        loss.backward()
        clip = 5.0
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        # optimizer update
        optimizer.step()
        loss.detach()
        torch.cuda.empty_cache()
    torch.save(net.state_dict(), save_dir+"/network.epoch{}".format(epoch+1))
    torch.save(optimizer.state_dict(), save_dir+"/network.optimizer.epoch{}".format(epoch+1))
    print("EPOCH {}".format(epoch))


