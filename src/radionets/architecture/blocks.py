from torch import nn

from radionets.architecture.layers import (
    ComplexConv2d,
    ComplexInstanceNorm2d,
    ComplexPReLU,
)

__all__ = [
    "NNBlock",
    "SRBlock",
    "ComplexSRBlock",
    "BottleneckResBlock",
    "Encoder",
    "Decoder",
]


class NNBlock(nn.Module):
    """Base block for neural networks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolution kernel. Default: 3
    stride : int or tuple, optional
        Stride for the cross-correlation. Default: 1
    padding : int, optional
        The amount of padding applied to the input.
        Default: 0
    groups : int, optional
        Controls the behavior of input and output groups.
        See :class:`~torch.nn.Conv2d`. Default: 1
    dropout : bool or float, optional
        Whether to apply dropout. If float > 0 this is
        the dropout percentage. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int | tuple = 1,
        padding: int = 0,
        groups: int = 1,
        dropout: bool | float = False,
    ):
        """Base block for neural networks.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolution kernel. Default: 3
        stride : int or tuple, optional
            Stride for the cross-correlation. Default: 1
        padding : int, optional
            The amount of padding applied to the input.
            Default: 0
        groups : int, optional
            Controls the behavior of input and output groups.
            See :class:`~torch.nn.Conv2d`. Default: 1
        dropout : bool or float, optional
            Whether to apply dropout. If float > 0 this is
            the dropout percentage. Default: False
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
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


class SRBlock(NNBlock):
    """Default SRResNet building block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolution kernel. Default: 3
    stride : int or tuple, optional
        Stride for the cross-correlation. Default: 1
    padding : int, optional
        The amount of padding applied to the input.
        Default: 0
    groups : int, optional
        Controls the behavior of input and output groups.
        See :class:`~torch.nn.Conv2d`. Default: 1
    dropout : bool or float, optional
        Whether to apply dropout. If float > 0 this is
        the dropout percentage. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dropout: bool | int = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dropout=dropout,
        )
        """Default SRResNet building block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolution kernel. Default: 3
        stride : int or tuple, optional
            Stride for the cross-correlation. Default: 1
        padding : int, optional
            The amount of padding applied to the input.
            Default: 0
        groups : int, optional
            Controls the behavior of input and output groups.
            See :class:`~torch.nn.Conv2d`. Default: 1
        dropout : bool or float, optional
            Whether to apply dropout. If float > 0 this is
            the dropout percentage. Default: False
        """

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
            nn.Dropout(p=self.dropout),
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
            nn.Dropout(p=self.dropout),
            nn.InstanceNorm2d(num_features=self.out_channels),
        ]

        return blocks

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))


class ComplexSRBlock(NNBlock):
    """Complex SRResNet building block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolution kernel. Default: 3
    stride : int or tuple, optional
        Stride for the cross-correlation. Default: 1
    padding : int, optional
        The amount of padding applied to the input.
        Default: 0
    groups : int, optional
        Controls the behavior of input and output groups.
        See :class:`~torch.nn.Conv2d`. Default: 1
    dropout : bool or float, optional
        Whether to apply dropout. If float > 0 this is
        the dropout percentage. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dropout: int = 0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dropout=dropout,
        )
        """Complex SRResNet building block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolution kernel. Default: 3
        stride : int or tuple, optional
            Stride for the cross-correlation. Default: 1
        padding : int, optional
            The amount of padding applied to the input.
            Default: 0
        groups : int, optional
            Controls the behavior of input and output groups.
            See :class:`~torch.nn.Conv2d`. Default: 1
        dropout : bool or float, optional
            Whether to apply dropout. If float > 0 this is
            the dropout percentage. Default: False
        """

        self.convs = nn.Sequential(*self._conv_block())

    def _conv_block(self):
        blocks = [
            ComplexConv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=1,
                bias=False,
            ),
            nn.Dropout(p=self.dropout),
            ComplexInstanceNorm2d(num_features=self.out_channels),
            ComplexPReLU(num_parameters=2),
            ComplexConv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=1,
                bias=False,
            ),
            nn.Dropout(p=self.dropout),
            ComplexInstanceNorm2d(num_features=self.out_channels),
        ]

        return blocks

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))


class BottleneckResBlock(NNBlock):
    """Three-convolution layer deep residual neural network
    building block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolution kernel. Default: 3
    stride : int or tuple, optional
        Stride for the cross-correlation. Default: 1
    padding : int, optional
        The amount of padding applied to the input.
        Default: 0
    groups : int, optional
        Controls the behavior of input and output groups.
        See :class:`~torch.nn.Conv2d`. Default: 1
    dropout : bool or float, optional
        Whether to apply dropout. If float > 0 this is
        the dropout percentage. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dropout: bool | int = False,
        downsample: bool = False,
    ):
        """Three-convolution layer deep residual neural network
        building block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolution kernel. Default: 3
        stride : int or tuple, optional
            Stride for the cross-correlation. Default: 1
        padding : int, optional
            The amount of padding applied to the input.
            Default: 0
        groups : int, optional
            Controls the behavior of input and output groups.
            See :class:`~torch.nn.Conv2d`. Default: 1
        dropout : bool or float, optional
            Whether to apply dropout. If float > 0 this is
            the dropout percentage. Default: False
        """

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dropout=dropout,
        )

        self.prelu = nn.PReLU()
        self.convs = self._conv_block()

    def _conv_block(self):
        block = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels // 4,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.Dropout(p=self.dropout),
            nn.BatchNorm2d(self.out_channels // 4),
            self.prelu,
            nn.Conv2d(
                self.out_channels // 4,
                self.out_channels // 4,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=False,
            ),
            nn.Dropout(p=self.dropout),
            nn.BatchNorm2d(self.out_channels // 4),
            self.prelu,
            nn.Conv2d(
                self.out_channels // 4,
                self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.Dropout(p=self.dropout),
            nn.BatchNorm2d(self.out_channels),
        )

        return block

    def forward(self, x):
        x0 = x

        x = self.convs(x)

        x += x0
        x = self.prelu(x)

        return x


class Encoder(NNBlock):
    """Encoder block for UNets.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolution kernel. Default: 3
    stride : int or tuple, optional
        Stride for the cross-correlation. Default: 1
    padding : int, optional
        The amount of padding applied to the input.
        Default: 0
    groups : int, optional
        Controls the behavior of input and output groups.
        See :class:`~torch.nn.Conv2d`. Default: 1
    bias : bool
        Whether to apply bias. Default: False
    dropout : bool or float, optional
        Whether to apply dropout. If float > 0 this is
        the dropout percentage. Default: False
    batchnorm : bool, optional
        If ``True``, add a batchnorm layer to the
        encoder block. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
        dropout: bool | int = False,
        batchnorm: bool = False,
    ):
        """Encoder block for UNets.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolution kernel. Default: 3
        stride : int or tuple, optional
            Stride for the cross-correlation. Default: 1
        padding : int, optional
            The amount of padding applied to the input.
            Default: 0
        groups : int, optional
            Controls the behavior of input and output groups.
            See :class:`~torch.nn.Conv2d`. Default: 1
        bias : bool
            Whether to apply bias. Default: False
        dropout : bool or float, optional
            Whether to apply dropout. If float > 0 this is
            the dropout percentage. Default: False
        batchnorm : bool, optional
            If ``True``, add a batchnorm layer to the
            encoder block. Default: False
        """

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dropout=dropout,
        )

        self.bias = bias
        self.batchnorm = batchnorm

        self.encoder_block = nn.Sequential(self.__encoder_block())

    def __encoder_block(self):
        block = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.PReLU(),
        ]

        if self.batchnorm:
            block.insert(1, nn.BatchNorm2d(self.out_channels))

        return block

    def forward(self, x):
        return self.encoder_block(x)


class Decoder(NNBlock):
    """Decoder block for UNets.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Size of the convolution kernel. Default: 3
    stride : int or tuple, optional
        Stride for the cross-correlation. Default: 1
    padding : int, optional
        The amount of padding applied to the input.
        Default: 0
    output_padding : int, optional
        Controls the padding applied to the output.
        See :class:`~torch.nn.ConvTranspose2d`. Default: 0
    groups : int, optional
        Controls the behavior of input and output groups.
        See :class:`~torch.nn.Conv2d`. Default: 1
    bias : bool
        Whether to apply bias. Default: False
    dropout : bool or float, optional
        Whether to apply dropout. If float > 0 this is
        the dropout percentage. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        dropout: bool | int = False,
    ):
        """Decoder block for UNets.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolution kernel. Default: 3
        stride : int or tuple, optional
            Stride for the cross-correlation. Default: 1
        padding : int, optional
            The amount of padding applied to the input.
            Default: 0
        output_padding : int, optional
            Controls the padding applied to the output.
            See :class:`~torch.nn.ConvTranspose2d`. Default: 0
        groups : int, optional
            Controls the behavior of input and output groups.
            See :class:`~torch.nn.Conv2d`. Default: 1
        bias : bool
            Whether to apply bias. Default: False
        dropout : bool or float, optional
            Whether to apply dropout. If float > 0 this is
            the dropout percentage. Default: False
        """

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dropout=dropout,
        )

        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias

        self.decoder_block = nn.Sequential(self.__decoder_block())

    def __decoder_block(self):
        block = [
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                bias=self.bias,
            ),
            nn.PReLU(),
        ]

        return block

    def forward(self, x):
        return self.decoder_block(x)
