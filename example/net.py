import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform, uniform
import torch.nn.functional as F

def mini_weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(uniform(m.weight.data, -3e-3, 3e-3))
        #norm = m.weight.data.pow(2).sum(dim=0, keepdim=True).sqrt()
        #m.weight.data.mul_(0.001/norm)
        m.bias.data.fill_(0)

def weight_init(m):
    if m.__class__.__name__ == 'Linear':
        m.weight.data.copy_(kaiming_uniform(m.weight.data))
        #norm = m.weight.data.pow(2).sum(dim=0, keepdim=True).sqrt()
        #m.weight.data.mul_(1/norm)
        m.bias.data.fill_(0)

class PolNet(nn.Module):
    def __init__(self, ob_space, ac_space):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(ob_space.shape[0], 200)
        self.fc2 = nn.Linear(200, 100)
        self.mean_layer = nn.Linear(100, ac_space.shape[0])
        #self.log_std_layer = nn.Linear(100, ac_space.shape[0])
        self.log_std_param = nn.Parameter(torch.randn(ac_space.shape[0])*1e-10 - 1)

        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.mean_layer.apply(mini_weight_init)
        #self.log_std_layer.apply(mini_weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        mean = F.tanh(self.mean_layer(h))
        #log_std = self.log_std_layer(h)
        return mean, self.log_std_param





















































class DeterministicPolNet(nn.Module):
    def __init__(self, ob_space, ac_space, hidden_layer1, hidden_layer2):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(ob_space.shape[0], hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1,hidden_layer2)
        self.mean_layer = nn.Linear(hidden_layer2, ac_space.shape[0])

        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.mean_layer.apply(mini_weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        mean = F.tanh(self.mean_layer(h))
        return mean



class VNet(nn.Module):
    def __init__(self, ob_space):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(ob_space.shape[0], 200)
        self.fc2 = nn.Linear(200, 100)
        self.output_layer = nn.Linear(100, 1)
        self.apply(weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1(ob))
        h = F.relu(self.fc2(h))
        return self.output_layer(h)

class QNet(nn.Module):
    def __init__(self, ob_space, ac_space, hidden_layer1, hidden_layer2):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(ob_space.shape[0], hidden_layer1)
        self.fc2 = nn.Linear(ac_space.shape[0] + hidden_layer1, hidden_layer2)
        self.output_layer = nn.Linear(hidden_layer2, 1)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(mini_weight_init)

    def forward(self, ob, ac):
        h = F.relu(self.fc1(ob))
        h = torch.cat([h, ac], dim=1)
        h = F.relu(self.fc2(h))
        return self.output_layer(h)

class PolNetBN(nn.Module):
    def __init__(self, ob_space, ac_space):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(ob_space.shape[0], 400)
        self.fc1_bn=nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.fc2_bn=nn.BatchNorm1d(300)
        self.mean_layer = nn.Linear(300, ac_space.shape[0])
        #self.log_std_layer = nn.Linear(100, ac_space.shape[0])
        self.log_std_param = nn.Parameter(torch.randn(ac_space.shape[0])*1e-10 - 1)

        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.mean_layer.apply(mini_weight_init)
        #self.log_std_layer.apply(mini_weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1_bn(self.fc1(ob)))
        h = F.relu(self.fc2_bn(self.fc2(h)))
        mean = F.tanh(self.mean_layer(h))
        #log_std = self.log_std_layer(h)
        return mean, self.log_std_param

class DeterministicPolNetBN(nn.Module):
    def __init__(self, ob_space, ac_space, hidden_layer1, hidden_layer2):
        nn.Module.__init__(self)
        self.input = nn.BatchNorm1d(ob_space.shape[0])
        self.fc1 = nn.Linear(ob_space.shape[0], hidden_layer1)
        self.fc1_bn=nn.BatchNorm1d(hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc2_bn=nn.BatchNorm1d(hidden_layer2)
        self.mean_layer = nn.Linear(hidden_layer2, ac_space.shape[0])

        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.mean_layer.apply(mini_weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1_bn(self.fc1(ob)))
        h = F.relu(self.fc2_bn(self.fc2(h)))
        mean = F.tanh(self.mean_layer(h))
        return mean

class VNetBN(nn.Module):
    def __init__(self, ob_space):
        nn.Module.__init__(self)
        self.input_bn = nn.BatchNorm1d(ob_space.shape[0])
        self.fc1 = nn.Linear(ob_space.shape[0], 200)
        self.fc1_bn=nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.fc2_bn=nn.BatchNorm1d(100)
        self.output_layer = nn.Linear(100, 1)
        self.apply(weight_init)

    def forward(self, ob):
        h = F.relu(self.fc1_bn(self.fc1(self.input_bn(ob))))
        h = F.relu(self.fc2_bn(self.fc2(h)))
        return self.output_layer(h)

class QNetBN(nn.Module):
    def __init__(self, ob_space, ac_space, hidden_layer1, hidden_layer2):
        nn.Module.__init__(self)
        self.input_bn = nn.BatchNorm1d(ob_space.shape[0])
        self.fc1 = nn.Linear(ob_space.shape[0], hidden_layer1)
        self.fc1_bn=nn.BatchNorm1d(hidden_layer1)
        self.fc2 = nn.Linear(ac_space.shape[0] + hidden_layer1, hidden_layer2)
        self.output_layer = nn.Linear(hidden_layer2, 1)
        self.fc1.apply(weight_init)
        self.fc2.apply(weight_init)
        self.output_layer.apply(mini_weight_init)

    def forward(self, ob, ac):
        h = F.relu(self.fc1_bn(self.fc1(self.input_bn(ob))))
        h = torch.cat([h, ac], dim=1)
        h = F.relu(self.fc2(h))
        return self.output_layer(h)
