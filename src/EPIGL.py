#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable

dist_norm = 500000

def num2list(x):
    if type(x) is not list:
        x = [x]
    return x

class local_sLSTM_module(nn.Module):
    """ Enhancer-promoter separated LSTM module """
    def __init__(self, e_channel, p_channel, cnn_size, cnn_num, cnn_pool, lstm_size, lstm_layer=2, lstm_droprate=0, da=128, r=64):
        super(local_sLSTM_module, self).__init__()
        # cnn 
        self.e_cnn, self.p_cnn = nn.ModuleList(), nn.ModuleList()
        e_cnn_num, p_cnn_num = [e_channel] + cnn_num, [p_channel] + cnn_num
        for i in range(len(cnn_size)):
            self.e_cnn.append(
                    nn.Sequential(
                        nn.Conv1d(e_cnn_num[i], e_cnn_num[i + 1], cnn_size[i], padding=(cnn_size[i] // 2)),
                        nn.BatchNorm1d(e_cnn_num[i + 1]),
                        nn.ReLU(),
                        nn.MaxPool1d(cnn_pool[i])
                        )
                    )
            self.p_cnn.append(
                    nn.Sequential(
                        nn.Conv1d(p_cnn_num[i], p_cnn_num[i + 1], cnn_size[i], padding=(cnn_size[i] // 2)),
                        nn.BatchNorm1d(p_cnn_num[i + 1]),
                        nn.ReLU(),
                        nn.MaxPool1d(cnn_pool[i])
                        )
                    )

        # lstm
        self.lstm_size = lstm_size
        self.lstm_layer = lstm_layer
        self.e_lstm = nn.LSTM(cnn_num[-1], lstm_size, num_layers=lstm_layer, bidirectional=True, dropout=lstm_droprate)
        self.p_lstm = nn.LSTM(cnn_num[-1], lstm_size, num_layers=lstm_layer, bidirectional=True, dropout=lstm_droprate)

        # attention
        self.r = r
        self.linear_first = nn.Linear(self.lstm_size * 2, da)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = nn.Linear(da, r)
        self.linear_second.bias.data.fill_(0)

    def forward(self, enhancer, promoter):
        batch_size = enhancer.size(0)
        device = enhancer.device
        for cnn in self.e_cnn:
            enhancer = cnn(enhancer)
        for cnn in self.p_cnn:
            promoter = cnn(promoter)

        enhancer = enhancer.transpose(0, 1).transpose(0, 2)
        h0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        c0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        enhancer, (hn, cn) = self.e_lstm(enhancer, (h0, c0))
        del h0, c0, hn, cn

        promoter = promoter.transpose(0, 1).transpose(0, 2)
        h0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        c0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        promoter, (hn, cn) = self.p_lstm(promoter, (h0, c0))

        out = torch.cat((enhancer, promoter), dim=0).transpose(0, 1)
        x = torch.tanh(self.linear_first(out))
        x = self.linear_second(x)
        x = F.softmax(x, 1)
        att = x.transpose(1, 2)
        del x
        seq_embeddings = torch.matmul(att, out)
        avg_seq_embeddings = torch.sum(seq_embeddings, 1) / self.r
        del seq_embeddings
        avg_seq_embeddings = avg_seq_embeddings.view(batch_size, -1)
        return avg_seq_embeddings, att

    def freeze_cnn(self):
        for param in self.e_cnn.parameters():
            param.requires_grad = False
        for param in self.p_cnn.parameters():
            param.requires_grad = False
    def unfreeze_cnn(self):
        for param in self.e_cnn.parameters():
            param.requires_grad = True
        for param in self.p_cnn.parameters():
            param.requires_grad = True
    def freeze_lstm(self):
        for param in self.e_lstm.parameters():
            param.requires_grad = False
        for param in self.p_lstm.parameters():
            param.requires_grad = False
    def unfreeze_lstm(self):
        for param in self.e_lstm.parameters():
            param.requires_grad = True
        for param in self.p_lstm.parameters():
            param.requires_grad = True
    def freeze_att(self):
        for param in self.linear_first.parameters():
            param.requires_grad = False
        for param in self.linear_second.parameters():
            param.requires_grad = False
    def unfreeze_att(self):
        for param in self.linear_first.parameters():
            param.requires_grad = True
        for param in self.linear_second.parameters():
            param.requires_grad = True




class local_uLSTM_module(nn.Module):
    """ Enhancer-promoter unified LSTM module """
    def __init__(self, e_channel, p_channel, cnn_size, cnn_num, cnn_pool, lstm_size, lstm_layer=2, lstm_droprate=0, da=128, r=64):
        super(local_uLSTM_module, self).__init__()
        # cnn 
        self.e_cnn, self.p_cnn = nn.ModuleList(), nn.ModuleList()
        e_cnn_num, p_cnn_num = [e_channel] + cnn_num, [p_channel] + cnn_num
        for i in range(len(cnn_size)):
            self.e_cnn.append(
                    nn.Sequential(
                        nn.Conv1d(e_cnn_num[i], e_cnn_num[i + 1], cnn_size[i], padding=(cnn_size[i] // 2)),
                        nn.BatchNorm1d(e_cnn_num[i + 1]),
                        nn.ReLU(),
                        nn.MaxPool1d(cnn_pool[i])
                        )
                    )
            self.p_cnn.append(
                    nn.Sequential(
                        nn.Conv1d(p_cnn_num[i], p_cnn_num[i + 1], cnn_size[i], padding=(cnn_size[i] // 2)),
                        nn.BatchNorm1d(p_cnn_num[i + 1]),
                        nn.ReLU(),
                        nn.MaxPool1d(cnn_pool[i])
                        )
                    )
        # lstm
        self.lstm_size = lstm_size
        self.lstm_layer = lstm_layer
        self.lstm = nn.LSTM(cnn_num[-1], lstm_size, num_layers=lstm_layer, bidirectional=True, dropout=lstm_droprate)

        # attention
        self.r = r
        self.linear_first = nn.Linear(self.lstm_size * 2, da)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = nn.Linear(da, r)
        self.linear_second.bias.data.fill_(0)

    def forward(self, enhancer, promoter):
        batch_size = enhancer.size(0)
        device = enhancer.device
        for cnn in self.e_cnn:
            enhancer = cnn(enhancer)
        for cnn in self.p_cnn:
            promoter = cnn(promoter)
        signal_dim = enhancer.size(1)
        space = torch.zeros(batch_size, signal_dim, 1).to(device)
        x = torch.cat((enhancer, space, promoter), dim=2)
        x = x.transpose(0, 1).transpose(0, 2)
        h0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        c0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        del h0, c0, hn, cn
        out = out.transpose(0, 1)
        x = torch.tanh(self.linear_first(out))
        x = self.linear_second(x)
        x = F.softmax(x, 1)
        att = x.transpose(1, 2)
        del x
        seq_embeddings = torch.matmul(att, out)
        avg_seq_embeddings = torch.sum(seq_embeddings, 1) / self.r
        del seq_embeddings
        avg_seq_embeddings = avg_seq_embeddings.view(batch_size, -1)
        return avg_seq_embeddings, att

    def freeze_cnn(self):
        for param in self.e_cnn.parameters():
            param.requires_grad = False
        for param in self.p_cnn.parameters():
            param.requires_grad = False
    def unfreeze_cnn(self):
        for param in self.e_cnn.parameters():
            param.requires_grad = True
        for param in self.p_cnn.parameters():
            param.requires_grad = True
    def freeze_lstm(self):
        for param in self.lstm.parameters():
            param.requires_grad = False
    def unfreeze_lstm(self):
        for param in self.lstm.parameters():
            param.requires_grad = True
    def freeze_att(self):
        for param in self.linear_first.parameters():
            param.requires_grad = False
        for param in self.linear_second.parameters():
            param.requires_grad = False
    def unfreeze_att(self):
        for param in self.linear_first.parameters():
            param.requires_grad = True
        for param in self.linear_second.parameters():
            param.requires_grad = True


class global_module(nn.Module):
    def __init__(self, in_channel, cnn_size, cnn_num, cnn_pool, lstm_size, lstm_layer=2, lstm_droprate=0, da=128, r=64):
        super(global_module, self).__init__()
        # cnn
        self.cnn = nn.ModuleList()
        cnn_num = [in_channel] + cnn_num
        for i in range(len(cnn_size)):
            self.cnn.append(
                    nn.Sequential(
                        nn.Conv1d(cnn_num[i], cnn_num[i + 1], cnn_size[i], padding=(cnn_size[i] // 2)),
                        nn.BatchNorm1d(cnn_num[i + 1]),
                        nn.ReLU(),
                        nn.MaxPool1d(cnn_pool[i])
                        )
                    )

        # lstm
        self.lstm_size = lstm_size
        self.lstm_layer = lstm_layer
        self.lstm = nn.LSTM(cnn_num[-1], lstm_size, num_layers=lstm_layer, bidirectional=True, dropout=lstm_droprate)

        # attention
        self.r = r
        self.linear_first = nn.Linear(lstm_size * 2, da)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = nn.Linear(da, r)
        self.linear_second.bias.data.fill_(0)

    def forward(self, global_signal):
        batch_size = global_signal.size(0)
        device = global_signal.device
        for i, cnn in enumerate(self.cnn):
            global_signal = cnn(global_signal)
        global_signal = global_signal.transpose(0, 1).transpose(0, 2)
        h0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        c0 = Variable(torch.zeros(self.lstm_layer * 2, batch_size, self.lstm_size)).to(device)
        global_signal, (hn, cn) = self.lstm(global_signal, (h0, c0))
        del h0, c0
        global_signal = global_signal.transpose(0, 1)
        out = torch.tanh(self.linear_first(global_signal))
        out = F.softmax(self.linear_second(out), 1)
        att = out.transpose(1, 2)
        del out
        seq_embeddings = torch.matmul(att, global_signal)
        avg_seq_embeddings = torch.sum(seq_embeddings, 1) / self.r
        del seq_embeddings
        avg_seq_embeddings = avg_seq_embeddings.view(batch_size, -1)
        return avg_seq_embeddings, att
        
    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False
    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
    def freeze_lstm(self):
        for param in self.lstm.parameters():
            param.requires_grad = False
    def unfreeze_lstm(self):
        for param in self.lstm.parameters():
            param.requires_grad = True
    def freeze_att(self):
        for param in self.linear_first.parameters():
            param.requires_grad = False
        for param in self.linear_second.parameters():
            param.requires_grad = False
    def unfreeze_att(self):
        for param in self.linear_first.parameters():
            param.requires_grad = True
        for param in self.linear_second.parameters():
            param.requires_grad = True



class EPIGL(nn.Module):
    def __init__(self, \
            use_dist=True,
            use_local=True, use_global=True, \
            e_channel=19, p_channel=19, \
            ep_cnn_size=[41], ep_cnn_num=[32], ep_cnn_pool=[20], \
            ep_lstm_size=32, ep_lstm_layer=2, ep_droprate=0, \
            ep_da=128, ep_r=64, \
            mode='unified', \
            s_channel=9, \
            s_cnn_size=[5], s_cnn_num=[32], s_cnn_pool=[5], \
            s_lstm_size=32, s_lstm_layer=2, s_droprate=0, \
            s_da=64, s_r=32,
            fc_size=[64], fc_dr=0.3, trained_epoch=0):
        super(EPIGL, self).__init__()

        self.use_global = use_global
        self.use_local = use_local
        self.trained_epoch = trained_epoch
        self.use_dist = use_dist

        # local module
        if self.use_local:
            ep_cnn_size, ep_cnn_num, ep_cnn_pool = \
                    num2list(ep_cnn_size), num2list(ep_cnn_num), num2list(ep_cnn_pool)
            if mode == 'unified':
                local_module = local_uLSTM_module
            else:
                local_module = local_sLSTM_module
            self.local_module = local_module(e_channel, p_channel, \
                    cnn_size=ep_cnn_size, cnn_num=ep_cnn_num, cnn_pool=ep_cnn_pool, \
                    lstm_size=ep_lstm_size, lstm_layer=ep_lstm_layer, \
                    lstm_droprate=ep_droprate, \
                    da=ep_da, r=ep_r)
        else:
            self.local_module = None
            ep_lstm_size = 0

        if self.use_global:
            # global module
            s_cnn_size, s_cnn_num, s_cnn_pool = \
                    num2list(s_cnn_size), num2list(s_cnn_num), num2list(s_cnn_pool)
            self.global_module = global_module(s_channel, \
                    cnn_size=s_cnn_size, cnn_num=s_cnn_num, cnn_pool=s_cnn_pool, \
                    lstm_size=s_lstm_size, lstm_layer=s_lstm_layer, \
                    lstm_droprate=s_droprate, \
                    da=s_da, r=s_r)
        else:
            self.global_module = None
            s_lstm_size = 0
        
        # combine module

        fc_size = num2list(fc_size)
        self.fc = nn.ModuleList()
        if fc_size == 0 or fc_size[0] == 0:
            fc_size = [ep_lstm_size * 2 + s_lstm_size * 2 + (1 if self.use_dist else 0)]
        else:
            fc_size = [ep_lstm_size * 2 + s_lstm_size * 2 + (1 if self.use_dist else 0)] + fc_size
            for i in range(len(fc_size) - 1):
                self.fc.append(
                        nn.Sequential(
                            nn.Linear(fc_size[i], fc_size[i + 1]),
                            nn.ReLU()
                        )
                    )
        self.drouput =  nn.Dropout(fc_dr)
        self.final = nn.Linear(fc_size[-1], 1)
        
    def forward(self, e=None, p=None, g=None, dist=None, return_embed=False):
        if self.use_local:
            ep_embedding, ep_att = self.local_module(e, p)
        else:
            ep_embedding = torch.Tensor(g.size(0), 0).to(g.device)
            ep_att = None
        if self.use_global:
            s_embedding, s_att = self.global_module(g)
        else:
            s_embedding = torch.Tensor(e.size(0), 0).to(e.device)
            s_att = None
        x = torch.cat((ep_embedding, s_embedding), dim=1)
        if self.use_dist:
            dist = torch.log2(1 + dist_norm / dist)
            x = torch.cat((x, dist), dim=1)
        for fc in self.fc:
            x = fc(x)
        if return_embed:
            return x
        else:
            x = self.drouput(x)
            x = torch.sigmoid(self.final(x))
            return (x, ep_att, s_att)

    def l2_matrix_norm(self, m):
        return torch.sum(torch.sum(torch.sum(m**2, 1), 1)**0.5).type(torch.cuda.DoubleTensor)

    def freeze_CNN(self):
        self.freeze_global(cnn=True, lstm=False, att=False)
        self.freeze_local(cnn=True, lstm=False, att=False)
    def freeze_LSTM(self):
        self.freeze_global(cnn=False, lstm=True, att=False)
        self.freeze_local(cnn=False, lstm=True, att=False)
    def freeze_ATT(self):
        self.freeze_global(cnn=False, lstm=False, att=True)
        self.freeze_local(cnn=False, lstm=False, att=True)

    def freeze_global(self, cnn=True, lstm=True, att=True):
        if self.global_module is not None:
            if cnn:
                self.global_module.freeze_cnn()
            if lstm:
                self.global_module.freeze_lstm()
            if att:
                self.global_module.freeze_att()
    def unfreeze_global(self, cnn=True, lstm=True, att=True):
        if self.global_module is not None:
            if cnn:
                self.global_module.unfreeze_cnn()
            if lstm:
                self.global_module.unfreeze_lstm()
            if att:
                self.global_module.unfreeze_att()
    def freeze_local(self, cnn=True, lstm=True, att=True):
        if self.local_module is not None:
            if cnn:
                self.local_module.freeze_cnn()
            if lstm:
                self.local_module.freeze_lstm()
            if att:
                self.local_module.freeze_att()
    def unfreeze_local(self, cnn=True, lstm=True, att=True):
        if self.local_module is not None:
            if cnn:
                self.local_module.unfreeze_cnn()
            if lstm:
                self.local_module.unfreeze_lstm()
            if att:
                self.local_module.unfreeze_att()

class EPIGL_FC(nn.Module):
    def __init__(self, in_size, fc_size, fc_dr=0, trained_epoch=0, use_dist=True):
        super(EPIGL_FC, self).__init__()
        self.use_dist = use_dist
        self.trained_epoch = trained_epoch
        
        if type(fc_size) is int:
            if fc_size > 0:
                fc_size = [fc_size]
            else:
                fc_size = []
        fc_size = [in_size] + fc_size
        self.fc_dr = fc_dr

        self.fc = nn.ModuleList()
        for i in range(len(fc_size) - 1):
            self.fc.append(
                    nn.Sequential(
                        nn.Linear(fc_size[i], fc_size[i + 1], bias=False),
                        nn.ReLU()
                        )
                    )
        if self.fc_dr > 0:
            self.dropout = nn.Dropout(fc_dr)
        else:
            self.dropout = None

        self.lr = nn.Sequential(
                nn.Linear(fc_size[-1], 1, bias=False),
                nn.Sigmoid()
                )

    def forward(self, embeddings, dist=None):
        if embeddings.shape[1] == 1:
            return embeddings
        if self.use_dist:
            dist = torch.log2(1 + dist_norm / dist)
            embeddings = torch.cat((embeddings, dist), dim=1)
        for fc in self.fc:
            embeddings = fc(embeddings)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        out = self.lr(embeddings)
        return out
