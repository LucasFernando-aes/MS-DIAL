import torch
from torch import nn


class DIALBatchNorm1d(nn.Module):
    __constants__ = ['num_domains']

    def __init__(self, num_domains, num_features, eps=1e-05, momentum=0.1, track_running_stats=True):
        super(DIALBatchNorm1d, self).__init__()
        
        self.num_domains = num_domains
        
        self.bn = [nn.BatchNorm1d(num_features, 
                                   eps=eps, 
                                   momentum=momentum, 
                                   affine=False, 
                                   track_running_stats=track_running_stats) for i in range(num_domains)]
        
        self.bn = nn.ModuleList(self.bn)

        self.gamma = nn.Parameter(torch.Tensor((num_features)))
        self.beta = nn.Parameter(torch.Tensor((num_features)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def extra_repr(self):
        s = ('num_domains={num_domains}')
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:

            x = x.view((-1, self.num_domains) + x.size()[1:]).transpose(0, 1)

            norm = []
            for i in range(self.num_domains):
                norm.append(self.bn[i](x[i]))

            x = torch.stack(norm, dim=1).view((-1, ) + x.size()[2:])
            x = x * self.gamma + self.beta

        else:
            x = self.bn[0](x)
            x = x * self.gamma + self.beta

        return x


class DIALBatchNorm2d(nn.Module):
    __constants__ = ['num_domains']

    def __init__(self, num_domains, num_features, eps=1e-05, momentum=0.1, track_running_stats=True):
        super(DIALBatchNorm2d, self).__init__()
        
        self.num_domains = num_domains
        
        self.bn = [nn.BatchNorm2d(num_features, 
                                   eps=eps, 
                                   momentum=momentum, 
                                   affine=False, 
                                   track_running_stats=track_running_stats) for i in range(num_domains)]
        
        self.bn = nn.ModuleList(self.bn)

        self.gamma = nn.Parameter(torch.Tensor((num_features)))
        self.beta = nn.Parameter(torch.Tensor((num_features)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def extra_repr(self):
        s = ('num_domains={num_domains}')
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:

            x = x.view((-1, self.num_domains) + x.size()[1:]).transpose(0, 1)

            norm = []
            for i in range(self.num_domains):
                norm.append(self.bn[i](x[i]))

            x = torch.stack(norm, dim=1).view((-1, ) + x.size()[-3:])     
            x = x * self.gamma[:, None, None] + self.beta[:, None, None]
        else:
            x = self.bn[0](x)
            x = x * self.gamma[:, None, None] + self.beta[:, None, None]

        return x


class DIALBatchNorm3d(nn.Module):
    __constants__ = ['num_domains']

    def __init__(self, num_domains, num_features, eps=1e-05, momentum=0.1, track_running_stats=True):
        super(DIALBatchNorm3d, self).__init__()
        
        self.num_domains = num_domains
        
        self.bn = [nn.BatchNorm3d(num_features, 
                                   eps=eps, 
                                   momentum=momentum, 
                                   affine=False, 
                                   track_running_stats=track_running_stats) for i in range(num_domains)]
        
        self.bn = nn.ModuleList(self.bn)

        self.gamma = nn.Parameter(torch.Tensor((num_features)))
        self.beta = nn.Parameter(torch.Tensor((num_features)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def extra_repr(self):
        s = ('num_domains={num_domains}')
        return s.format(**self.__dict__)

    def forward(self, x):
        if self.training:

            x = x.view((-1, self.num_domains) + x.size()[1:]).transpose(0, 1)
            norm = []
            
            for i in range(self.num_domains):
                norm.append(self.bn[i](x[i]))

            x = torch.stack(norm, dim=1).view((-1, ) + x.size()[-4:])
            x = x * self.gamma[:, None, None, None] + self.beta[:, None, None, None]
        else:

            x = self.bn[0](x)
            x = x * self.gamma[:, None, None, None] + self.beta[:, None, None, None]

        return x
