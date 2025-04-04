from torch import nn


class AdaIN(nn.Module):

    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, x):
        # 计算均值和方差
        mu = x.mean(dim=-1, keepdim=True)
        sigma = (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        # 归一化
        x_normalized = (x - mu) / sigma
        return x_normalized
