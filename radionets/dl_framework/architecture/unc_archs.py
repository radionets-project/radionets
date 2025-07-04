import torch
from torch import nn

from radionets.dl_framework.architecture import SRResNet_34
from radionets.dl_framework.model import GeneralELU, LocallyConnected2d


class Uncertainty(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, 9, stride=1, padding=4, groups=2),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 9, stride=1, padding=4, groups=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            LocallyConnected2d(
                64,
                2,
                [img_size // 2 + 1, img_size],
                1,
                stride=1,
                bias=False,
            )
        )

        self.elu = GeneralELU(add=+(1 + 1e-7))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.final(x)

        return self.elu(x)


class UncertaintyWrapper(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.pred = SRResNet_34()

        self.uncertainty = Uncertainty(img_size)

    def forward(self, x):
        inp = x.clone()

        pred = self.pred(x)

        # x = torch.abs(pred - inp)
        x = torch.cat([pred, inp], dim=1)

        unc = self.uncertainty(x)

        val = torch.cat(
            [
                pred[:, 0].unsqueeze(1),
                unc[:, 0].unsqueeze(1),
                pred[:, 1].unsqueeze(1),
                unc[:, 1].unsqueeze(1),
            ],
            dim=1,
        )

        return val
