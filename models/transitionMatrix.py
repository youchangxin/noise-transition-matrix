import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransitionMatrix(nn.Module):
    def __init__(self, num_classes, device):
        super(TransitionMatrix, self).__init__()
        self.register_parameter(name='w', param=nn.parameter.Parameter(-2 * torch.ones(num_classes, num_classes)))

        self.w.to(device)

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.to(device)
        self.identity = torch.eye(num_classes).to(device)

    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig * self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T
        """  
        if num_classes == 10:
            init = -2
        else:
            init = -4.5
        super(TransitionMatrix, self).__init__()
        T_w= torch.ones([num_classes, num_classes]) * init
        self.register_parameter(name="T_w", param=nn.parameter.Parameter(T_w))
        self.T_w.to(device)

        self.identity = torch.eye(num_classes)
        self.identity = self.identity.to(device)

        self.coeff = torch.ones([num_classes, num_classes])
        coeff_diag = torch.diag_embed(self.coeff)[0]
        self.coeff = self.coeff - coeff_diag
        self.coeff = self.coeff.to(device)

    def forward(self):
        sig = torch.sigmoid(self.T_w)
        T = self.identity.detach() + sig * self.coeff.detach()
        T = F.normalize(T, p=1, dim=1)
        return T
"""
