import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import pdb

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    # print("ip: ", ip)
    w1 = torch.norm(x1, 2, dim)
    # print("w1: ", w1)
    w2 = torch.norm(x2, 2, dim)
    # print("w2: ", w2)
    # print("ger: ", torch.ger(w1,w2))
    cosine = ip / torch.ger(w1,w2).clamp(min=eps)             #torch.ger(w1,w2)dengjiayu w1.T*w2
    return cosine


class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=8.0, m=0.05):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):

        NormOfFeature = torch.norm(input, 2, 1)                                       # anzhao meihang qiu xiangliang de 2fanshu
        cosine = cosine_sim(input, self.weight)
        cosine = cosine.cuda()
        # pdb.set_trace()
        cosine1 = F.linear(F.normalize(input), F.normalize(self.weight))             #F.normalize(c)   daibiao chuyi meihang de 2fanshu
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        # print("label: ", label)
        one_hot = torch.zeros_like(cosine1)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # one_hot = one_hot.cuda()
        # print("one: ", one_hot)
        # one_hot = label
        # pdb.set_trace()
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (cosine1 - one_hot * self.m) * NormOfFeature.view(-1, 1)
        # out = NormOfFeature.view(-1, 1) * (cosine - one_hot * self.m)
        # pdb.set_trace()
        # output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

if __name__ == '__main__':
    pass