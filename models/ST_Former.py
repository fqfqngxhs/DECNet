import torch
from torch import nn
from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer



class DECNet(nn.Module):   #多任务+特征层融合   #最终版本
    def __init__(self, s_former_depth=1, t_former_depth=3, nf=32, Incep_depth=6):
        super().__init__()
        self.s_former_v = spatial_transformer(depth=s_former_depth)
        self.t_former_v = temporal_transformer(depth=t_former_depth)
        self.fc_v = nn.Linear(512, 7)
        self.incetionblock = InceptionBlock(ni=8, nf=nf, depth=Incep_depth)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool1d(output_size=1), nn.Flatten())
        self.fc_db = nn.Linear(nf * 4, 7)

        self.fc = nn.Linear(512 + nf*4, 7)
        

    def forward(self, x):
        (x1, x2) = x   #x1:video x2:DB
        x1 = self.s_former_v(x1)
        x1 = self.t_former_v(x1)
        o_v = self.fc_v(x1)

        x2 = self.incetionblock(x2)
        x2 = self.gap(x2)
        
        x = torch.cat((x1, x2), dim=1)   #shape = [batchzise, 680]

        o_db = self.fc_db(x2)

        x = self.fc(x)
        return o_v, o_db, x
    
    

    


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super().__init__()
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([nn.Conv1d(nf if bottleneck else ni, nf, k, padding=k//2, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False)])
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = torch.cat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)], dim=1)
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=32, residual=True, depth=6, **kwargs):  #ni:输入特征通道数，nf:输出特征通道数
        super().__init__()
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * 4, nf))
            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else nn.Sequential(nn.Conv1d(n_in, n_out, 1, bias=False), nn.BatchNorm1d(n_out)))
        self.act = nn.ReLU()
        # self.weight_decay = 1e-4

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                res = x = self.act(torch.add(x, self.shortcut[d//3](res)))
        return x

class InceptionTime(nn.Module):
    def __init__(self, c_in, c_out, nf=32, depth=6):
        super().__init__()
        self.incetionblock = InceptionBlock(c_in, nf, depth=depth)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool1d(output_size=1), nn.Flatten())
        self.fc = nn.Linear(nf * 4, c_out)

    def forward(self, x):
        x = self.incetionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return x