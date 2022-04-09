import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionMatrix(nn.Module):
    def __init__(self, num_classes, ):
        if num_classes == 10:
            init = -2
        else:
            init = -4.5
        super(TransitionMatrix, self).__init__()
        T_w= torch.ones([num_classes, num_classes]) * init
        self.register_parameter(name="T_w", param=nn.parameter.Parameter(T_w))

        self.identity = torch.eye(num_classes)

        self.coeff = torch.ones([num_classes, num_classes])
        coeff_diag = torch.diag_embed(self.coeff)[0]
        self.coeff = self.coeff - coeff_diag

    def forward(self):
        sig = torch.sigmoid(self.T_w)
        T = self.identity.detach() + sig * self.coeff.detach()
        T = F.normalize(T, p=1, dim=1)
        return T
