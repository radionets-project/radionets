import torch
from torch import nn
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    """A 2D locally connected layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    output_size : tuple[int, int]
        Output spatial size as ``(out_height, out_width)``.
    kernel_size : int | tuple[int, int]
        Kernel size.
    stride : int | tuple[int, int]
        Stride.
    bias : bool, default=False
        If ``True``, add learnable bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_size: tuple[int, int],
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = _pair(output_size)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        kh, kw = self.kernel_size
        oh, ow = self.output_size

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, oh, ow, kh * kw)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, oh, ow))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply locally connected transform.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(batch, in_channels, height, width)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(batch, out_channels, out_height, out_width)``.
        """
        kh, kw = self.kernel_size
        sh, sw = self.stride

        # (n, c, oh, ow, kh, kw)
        x_unf = x.unfold(2, kh, sh).unfold(3, kw, sw)
        # (n, c, oh, ow, kh*kw)
        x_unf = x_unf.contiguous().view(*x_unf.shape[:-2], kh * kw)

        oh_cfg, ow_cfg = self.output_size
        oh, ow = x_unf.shape[2], x_unf.shape[3]

        if oh < oh_cfg or ow < ow_cfg:
            raise ValueError(
                f"Input too small for configured output_size {self.output_size}; "
                f"computed {(oh, ow)}."
            )

        x_unf = x_unf[:, :, :oh_cfg, :ow_cfg, :]

        # Broadcast over out_channels:
        # x_unf.unsqueeze(1): (n, 1, c, oh, ow, kh*kw)
        # weight:             (1, out, c, oh, ow, kh*kw)
        out = (x_unf.unsqueeze(1) * self.weight).sum(dim=(2, 5))

        if self.bias is not None:
            out = out + self.bias

        return out


class ComplexConv2d(nn.Module):
    """
    2D convolution layer for complex-valued tensors.

    This layer performs 2D convolution on complex-valued inputs by decomposing
    the operation into separate real and imaginary components. It implements
    the mathematical formula for complex multiplication:
    (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple of int
        Size of the convolving kernel. If int, the same value is used for
        both height and width dimensions.
    stride : int or tuple of int, optional
        Stride of the convolution. If int, the same value is used for both
        height and width dimensions. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.

    Attributes
    ----------
    conv_real : torch.nn.Conv2d
        Convolution layer for processing real components of the complex
        multiplication formula.
    conv_imag : torch.nn.Conv2d
        Convolution layer for processing imaginary components of the complex
        multiplication formula.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        bias=True,
    ):
        """
        Initialize the ComplexConv2d layer.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int or tuple of int
            Size of the convolving kernel.
        stride : int or tuple of int, optional
            Stride of the convolution. Default is 1.
        bias : bool, optional
            If True, adds a learnable bias to the output. Default is True.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize real component convolution layer
        self.conv_real = nn.Conv2d(
            self.in_channels // 2,
            self.out_channels // 2,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Initialize imaginary component convolution layer
        self.conv_imag = nn.Conv2d(
            self.in_channels // 2,
            self.out_channels // 2,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass of the complex convolution layer.

        Performs complex-valued 2D convolution by applying separate
        convolutions for real and imaginary values, and combining
        results according to complex multiplication rules.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width) with
            dtype (torch.float32 or torch.float64). Expected channels are equally
            split into real and imag channels, e.g., num channels is 2 for first
            network layer.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        real, imag = x.chunk(2, dim=1)

        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        return torch.cat([real_out, imag_out], dim=1)


class ComplexInstanceNorm2d(nn.Module):
    """
    2D instance normalization layer for complex-valued tensors.
    This layer performs instance normalization on complex-valued inputs by
    treating real and imaginary parts separately.

    The normalization is applied as:
    normalized = (x - mean) / sqrt(variance + eps)

    If affine=True, learnable scale and shift parameters are applied:
    output = normalized * weight + bias

    Parameters
    ----------
    num_features : int
        Number of channels in the input tensor. For complex inputs, this
        represents the number of complex channels (input will have 2*num_features
        channels representing real and imaginary parts).
    eps : float, optional
        A small value added to the denominator for numerical stability.
        Default is 1e-5.
    affine : bool, optional
        If True, adds learnable affine parameters (scale and shift).
        Default is True.

    Attributes
    ----------
    num_features : int
        Number of complex channels which get split into real and imaginary
        part equally. (num_channels // 2 for real and imag)
    eps : float
        Epsilon value for numerical stability.
    affine : bool
        Whether affine transformation is enabled.
    weight_real : torch.nn.Parameter or None
        Learnable scale parameter for real part. Shape: (num_features // 2,).
        Only exists if affine=True.
    weight_imag : torch.nn.Parameter or None
        Learnable scale parameter for imaginary part. Shape: (num_features // 2,).
        Only exists if affine=True.
    bias_real : torch.nn.Parameter or None
        Learnable shift parameter for real part. Shape: (num_features // 2,).
        Only exists if affine=True.
    bias_imag : torch.nn.Parameter or None
        Learnable shift parameter for imaginary part. Shape: (num_features // 2,).
        Only exists if affine=True.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        Initialize the ComplexInstanceNorm2d layer.

        Parameters
        ----------
        num_features : int
            Number of channels in the input tensor. Num_features will be
            equally split into channels for real and imag.
        eps : float, optional
            Small value added to variance for numerical stability.
            Must be positive. Default is 1e-5.
        affine : bool, optional
            If True, creates learnable affine parameters (weights and biases)
            for both real and imaginary components. Default is True.
        """
        super().__init__()

        # Store configuration
        self.num_features = num_features // 2  # Divide by 2 for equal real and imag
        self.eps = eps
        self.affine = affine

        if self.affine:
            # Separate parameters for real and imaginary parts
            # Initialize weights to ones for identity scaling
            self.weight_real = nn.Parameter(torch.ones(self.num_features))
            self.weight_imag = nn.Parameter(torch.ones(self.num_features))

            # Initialize biases to zeros for no initial shift
            self.bias_real = nn.Parameter(torch.zeros(self.num_features))
            self.bias_imag = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x):
        """
        Forward pass of the complex instance normalization layer.

        Performs instance normalization on complex-valued input by processing
        real and imaginary components separately. Computes statistics across
        spatial dimensions for each sample and channel independently.

        Complex values have to be passed in separate channels as torch.float,
        e.g., one channel for real and one channel for imag, leading to a shape
        [bs, 2, h, w].

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_features, height, width)
            where the first half of channels represents real parts and the
            second half represents imaginary parts of complex values.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as input
            (batch_size, num_features, height, width).

        """
        # Split complex input into real and imaginary components
        real, imag = x.chunk(2, dim=1)

        # Instance normalization for each part separately
        # Calc mean and variance across dimensions (H, W) for each sample and channel
        # unbiased=False: uses N denominator instead of N-1 for variance
        real_mean = real.mean(dim=[2, 3], keepdim=True)
        imag_mean = imag.mean(dim=[2, 3], keepdim=True)

        real_var = real.var(dim=[2, 3], keepdim=True, unbiased=False)
        imag_var = imag.var(dim=[2, 3], keepdim=True, unbiased=False)

        # Normalize
        real_norm = (real - real_mean) / torch.sqrt(real_var + self.eps)
        imag_norm = (imag - imag_mean) / torch.sqrt(imag_var + self.eps)

        if self.affine:
            # Apply learnable affine transformation
            real_norm = real_norm * self.weight_real.view(1, -1, 1, 1).to(
                real_norm.device
            ) + self.bias_real.view(1, -1, 1, 1).to(real_norm.device)

            imag_norm = imag_norm * self.weight_imag.view(1, -1, 1, 1).to(
                real_norm.device
            ) + self.bias_imag.view(1, -1, 1, 1).to(real_norm.device)

        return torch.cat([real_norm, imag_norm], dim=1)


class ComplexPReLU(nn.Module):
    """
    Parametric ReLU activation function for complex-valued tensors.

    This layer applies Parametric ReLU activation to complex-valued inputs by
    treating real and imaginary parts separately. PReLU allows the negative
    slope to be learned during training, providing more flexibility than
    standard ReLU activation.

    The activation is applied as:
    - For positive values: f(x) = x
    - For negative values: f(x) = a * x

    where 'a' is the learnable negative slope parameter.

    Parameters
    ----------
    num_parameters : int, optional
        Number of learnable parameters. Can be:
        - 1: Single shared parameter for all channels (default)
        - num_channels: Per-channel parameters for fine-grained control
        Default is 1.
    init : float, optional
        Initial value for the negative slope parameter(s).
        Should be a small positive value. Default is 0.25.

    Attributes
    ----------
    num_parameters : int
        Number of learnable parameters (1 for shared, num_channels for per-channel).
    weight_real : torch.nn.Parameter
        Learnable negative slope parameter(s) for real channel(s).
        Shape: (num_parameters // 2,)
    weight_imag : torch.nn.Parameter
        Learnable negative slope parameter(s) for imaginary channel(s).
        Shape: (num_parameters // 2,)
    """

    def __init__(self, num_parameters=1, init=0.25):
        """
        Initialize the ComplexPReLU activation layer.

        Parameters
        ----------
        num_parameters : int, optional
            Number of learnable parameters. Options:
            - 1: Single parameter shared across all channels (default)
            - num_channels: Individual parameter per channel
            Must be positive integer. Default is 1.
        init : float, optional
            Initial value for the negative slope parameter(s).
            Typically a small positive value (e.g., 0.01 to 0.25).
            Must be finite and typically in range [0, 1].
            Default is 0.25.
        """
        super().__init__()

        # Store configuration
        self.num_parameters = num_parameters

        # Create separate learnable parameters for real and imaginary parts
        n_params = self.num_parameters // 2 if self.num_parameters >= 2 else 1

        self.weight_real = nn.Parameter(torch.full((n_params,), init))
        self.weight_imag = nn.Parameter(torch.full((n_params,), init))

    def forward(self, x):
        """
        Forward pass of the complex PReLU activation function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_channels, height, width)
            where the first half of channels represents real parts and the
            second half represents imaginary parts of complex values.

        Returns
        -------
        torch.Tensor
            Activated tensor of the same shape and dtype as input
            (batch_size, num_channels, height, width).
        """
        # Split channels into real and imaginary components
        real, imag = x.chunk(2, dim=1)

        if self.num_parameters == 1:
            # Shared parameter across all channels
            real_out = torch.where(real >= 0, real, self.weight_real * real)
            imag_out = torch.where(imag >= 0, imag, self.weight_imag * imag)
        else:
            # Per-channel parameters
            weight_real = self.weight_real.view(1, -1, 1, 1)
            weight_imag = self.weight_imag.view(1, -1, 1, 1)

            real_out = torch.where(real >= 0, real, weight_real * real)
            imag_out = torch.where(imag >= 0, imag, weight_imag * imag)

        return torch.cat([real_out, imag_out], dim=1)
