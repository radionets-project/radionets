from torch import nn


class BaseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: bool | int = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout

        self.idconv = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, 1)
        )
        self.pool = (
            nn.Identity()
            if self.stride == 1
            else nn.AvgPool2d(kernel_size=2, ceil_mode=True)
        )

    def _conv_block(self):
        pass

    def forward(self, x):
        pass


class SRBlock(BaseBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: bool | int = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout,
        )

        self.convs = nn.Sequential(*self._conv_block())

    def _conv_block(self):
        blocks = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features=self.out_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features=self.out_channels),
        ]

        # NOTE: This will be included directly in the blocks
        # list in a future release
        if self.dropout:
            blocks.insert(1, nn.Dropout(p=self.dropout))
            blocks.insert(4, nn.Rropout(p=self.dropout))

        return blocks

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))
