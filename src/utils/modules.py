import torch
from torch import nn
from itertools import permutations
from torch.autograd import Function
from .misc import calc_coeff, grl_hook


class AvgPoolAggregation(nn.Module):
    """Performs Average Pooling aggregation in the clip dimension."""

    def forward(self, x):
        return torch.mean(x, dim=1)


class AverageAggregation(nn.Module):
    def forward(self, x1, x2):
        x = torch.stack((x1, x2), dim=0)
        x = torch.mean(x, dim=0)
        return x


class MLPAggregation(nn.Module):
    def __init__(self, feature_size, n_clips, layers=1, add_bn=False, dropout=0.0):
        super().__init__()

        self.n_clips = n_clips

        self.mlp = []
        self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(nn.Linear(self.n_clips * feature_size, feature_size))
        if add_bn:
            self.mlp.append(nn.BatchNorm1d(feature_size))
        self.mlp.append(nn.ReLU(inplace=True))

        for i in range(layers - 2):
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.Linear(feature_size, feature_size))
            if add_bn:
                self.mlp.append(nn.BatchNorm1d(feature_size))
            self.mlp.append(nn.ReLU(inplace=True))

        self.mlp.append(nn.Linear(feature_size, feature_size))

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
        x = self.mlp(x)
        return x


class MLPAggregationWeights(nn.Module):
    def __init__(self, feature_size, n_clips, layers=1, add_bn=False, dropout=0.0):
        super().__init__()

        self.n_clips = n_clips
        self.mlp = []
        self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(nn.Linear(self.n_clips * feature_size, feature_size))
        if add_bn:
            self.mlp.append(nn.BatchNorm1d(feature_size))
        self.mlp.append(nn.ReLU(inplace=True))

        for i in range(layers - 2):
            self.mlp.append(nn.Dropout(dropout))
            self.mlp.append(nn.Linear(feature_size, feature_size))
            if add_bn:
                self.mlp.append(nn.BatchNorm1d(feature_size))
            self.mlp.append(nn.ReLU(inplace=True))

        self.mlp.append(nn.Linear(feature_size, n_clips))
        self.mlp.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        b = x.size(0)
        x_unrolled = x.view(b, -1)
        weights = self.mlp(x_unrolled)
        x = x * weights.unsqueeze(2)
        return torch.mean(x, dim=1)


class LSTMAggregation(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=feature_size, hidden_size=feature_size, batch_first=True
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        return x

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class LSTMAggregationWeights(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=feature_size, hidden_size=feature_size, batch_first=True
        )
        self.linear = nn.Linear(feature_size, 4)

    def forward(self, x):
        h, _ = self.rnn(x)
        h = h[:, -1, :]
        w = F.softmax(self.linear(h), dim=1)
        x = x * w.unsqueeze(2)
        x = torch.mean(x, dim=1)
        return x

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class Classifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dim=512):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x


# internal block for cop
class COPInternalBlock(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, hidden_dim)
        self.relu = nn.ReLU(hidden_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class COPModule(nn.Module):
    def __init__(self, device, n_clips=3, in_dim=1024, hidden_dim=512, dropout_p=0.5):
        super().__init__()
        self.n_clips = n_clips
        self.device = device
        self.in_dim = in_dim
        self.block = COPInternalBlock(in_dim=in_dim, hidden_dim=hidden_dim)
        self.permutations = list(permutations(range(n_clips)))
        self.dropout = nn.Dropout(dropout_p)
        self.pair_num = int(n_clips * (n_clips - 1) / 2)
        self.fc = nn.Linear(hidden_dim * self.pair_num, len(self.permutations))

    def forward(self, x):
        device = x.device
        samples = []
        labels = []
        for i in range(x.size(0)):
            indices = torch.randperm(x.size(1))
            indices_tuple = tuple(indices.tolist())
            label = self.permutations.index(indices_tuple)
            sample = x[i, indices]
            samples.append(sample)
            labels.append(label)
        x = torch.stack(samples, dim=0).cuda(device=device)
        labels = torch.tensor(labels).cuda(device=device)
        cats = []
        for i in range(self.n_clips):
            for j in range(i + 1, self.n_clips):
                cat = torch.cat((x[:, i, :], x[:, j, :]), dim=1)
                cats.append(cat)
        feats = []
        for cat in cats:
            feat = self.block(cat)
            feats.append(feat)
        feat = torch.cat(feats, dim=1)
        feat = self.dropout(feat)
        out = self.fc(feat)

        return out, labels


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class SimCLRProjector_old(nn.Module):
    def __init__(self, input_dim, proj_hidden_dim, proj_output_dim):
        super(SimCLRProjector_old, self).__init__()
        self.fc1 = nn.Linear(input_dim, proj_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimCLRProjector(nn.Module):
    def __init__(self, input_dim, proj_hidden_dim, proj_output_dim, add_bn, n_layers):
        super(SimCLRProjector, self).__init__()
        self.mlp = []
        self.mlp.append(nn.Linear(input_dim, proj_hidden_dim))
        if add_bn:
            self.mlp.append(nn.BatchNorm1d(proj_hidden_dim))
        self.mlp.append(nn.ReLU())

        for i in range(n_layers - 2):
            self.mlp.append(nn.Linear(proj_hidden_dim, proj_hidden_dim))
            if add_bn:
                self.mlp.append(nn.BatchNorm1d(proj_hidden_dim))
            self.mlp.append(nn.ReLU())

        self.mlp.append(nn.Linear(proj_hidden_dim, proj_output_dim))

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        x = self.mlp(x)
        return x


class KMeansProjector(nn.Module):
    def __init__(self, input_dim, proj_hidden_dim, proj_output_dim):
        super(KMeansProjector, self).__init__()
        self.fc1 = nn.Linear(input_dim, proj_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=100, max_iter=3000, alpha=10):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)
        self.iter_num = 0
        self.alpha = alpha
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x, reverse=True):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(
            self.iter_num, self.high, self.low, self.alpha, self.max_iter
        )
        x = x * 1.0
        if reverse:
            x.register_hook(grl_hook(coeff))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x, coeff
