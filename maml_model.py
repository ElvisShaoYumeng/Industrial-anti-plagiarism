import torch.nn.modules as nn
import torch


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)  #均匀分布初始化
    torch.nn.init.constant_(module.bias.data, 0.0)               #常数初始化
    return module


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool_factor=1.0):
        super().__init__()
        stride = (int(2 * max_pool_factor))
        self.max_pool = nn.MaxPool1d(kernel_size=stride, stride=stride, ceil_mode=False)
        self.normalize = nn.BatchNorm1d(out_channels, affine=True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):
    def __init__(self, hidden=64, channels=1, layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, 3, max_pool_factor)]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, 3, max_pool_factor))
        super(ConvBase, self).__init__(*core)


class CNN4Backbone(ConvBase):
    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class Net4CNN(torch.nn.Module):
    def __init__(self, output_size, hidden_size, layers, channels, embedding_size):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers, max_pool_factor=4 // layers)
        self.classifier = torch.nn.Linear(embedding_size, output_size, bias=True)
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Net4CNN_add(torch.nn.Module):
    def __init__(self, output_size, hidden_size, layers, channels, embedding_size):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers, max_pool_factor=4 // layers)
        self.classifier1 = torch.nn.Linear(embedding_size, 256, bias=True)
        self.classifier2 = torch.nn.Linear(256, output_size, bias=True)
        maml_init_(self.classifier1)
        maml_init_(self.classifier2)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x
    

class Net4CNN_PU(torch.nn.Module):
    def __init__(self, output_size, hidden_size, layers, channels, embedding_size):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers, max_pool_factor=4 // layers)
        self.classifier1 = torch.nn.Linear(embedding_size, 256, bias=True)
        self.classifier2 = torch.nn.Linear(256, output_size, bias=True)
        maml_init_(self.classifier1)
        maml_init_(self.classifier2)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x
    
class BP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Linear(1024, 5, bias=True)
        self.classifier = torch.nn.Linear(5, 5, bias=True)
    
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        x = x.view(x.shape[0], x.shape[-1])
        return x