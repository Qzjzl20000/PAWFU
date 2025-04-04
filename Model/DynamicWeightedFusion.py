import torch
import torch.nn as nn


class DynamicWeightedFusion(nn.Module):

    def __init__(self):
        super(DynamicWeightedFusion, self).__init__()
        self.weightedFC = nn.Sequential(nn.Linear(3, 16), nn.Tanh(),
                                        nn.Linear(16, 8), nn.Tanh(),
                                        nn.Linear(8, 8), nn.Tanh(),
                                        nn.Linear(8, 3))
        #   ,nn.Softmax(dim=-1))

    def forward(self, U, R1, R2):
        # U = self.standardize(U)
        # R1 = self.standardize(R1)
        # R2 = self.standardize(R2)

        x = torch.concat([U, R1, R2], dim=-1)
        # weights = self.weightedFC(x)+1
        weights = x * 0 + 1
        w1, w2, w3 = torch.chunk(weights, chunks=3, dim=-1)
        return w1, w2, w3

    def standardize(self, tensor):
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        standardized_tensor = (tensor - mean) / (std + 1e-8)
        return standardized_tensor
