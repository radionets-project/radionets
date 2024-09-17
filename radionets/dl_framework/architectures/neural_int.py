import torch
from einops import rearrange
from torch import einsum, nn


class ImplicitAstro_Trans(nn.Module):
    def __init__(self):
        super().__init__()

        self.cond_mlp = PosEncodedMLP_FiLM(
            context_dim=128,
            input_size=2,
            output_size=2,
            hidden_dims=[128, 128, 128],
            L_embed=5,  # 5
            embed_type="fourier",
            activation=nn.PReLU,
            sigma=2.5,  # 2.5
            context_type="Transformer",
        )

        self.context_encoder = VisibilityTransformer(
            input_dim=2,  # value dim
            # PE dim for MLP, we are going to use the same PE as the MLP
            pe_dim=self.cond_mlp.input_size,
            dim=128,
            depth=4,
            heads=16,
            dim_head=128 // 16,
            output_dim=128,  # latent_dim,
            dropout=0.1,
            emb_dropout=0.0,
            mlp_dim=128,
            output_tokens=4,  # args.mlp_layers,
            has_global_token=False,
        )

    def forward(self, x):
        uv_coords, uv_dense, vis_sparse, _, _ = x
        pe_encoder = self.cond_mlp.embed_fun
        pe_uv = pe_encoder(uv_coords)
        inputs_encoder = torch.cat([pe_uv, vis_sparse], dim=-1)
        z = self.context_encoder(inputs_encoder)

        halfspace = get_halfspace(uv_dense)
        uv_coords_flipped = flip_uv(uv_dense, halfspace)
        pred_visibilities = self.cond_mlp(uv_coords_flipped, context=z)
        return pred_visibilities


def calcB(m=1024, d=2, sigma=1.0):
    B = torch.randn(m, d) * sigma
    return B


def fourierfeat_enc(x, B):
    feat = torch.cat(
        [
            x,
            torch.cos(2 * 3.14159265 * (x @ B.T)),
            torch.sin(2 * 3.14159265 * (x @ B.T)),
        ],
        -1,
    )
    return feat


class PE_Module(torch.nn.Module):
    def __init__(self, type, embed_L):
        super(PE_Module, self).__init__()

        self.embed_L = embed_L
        self.type = type

    def forward(self, x):
        return fourierfeat_enc(x, B=self.embed_L)


class FiLMLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        context_dim=64,
        residual=False,
    ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation1 = nn.LeakyReLU()
        # self.activation2 = nn.LeakyReLU()
        self.film1 = nn.Linear(context_dim, out_dim)
        self.film2 = nn.Linear(context_dim, out_dim)
        self.residual = residual

    def forward(self, x, shape_context):
        out = self.linear(x)
        gamma = self.film1(shape_context)
        beta = self.film2(shape_context)

        out = gamma * out + beta
        out = self.activation1(out)
        return out


class PosEncodedMLP_FiLM(nn.Module):
    def __init__(
        self,
        context_dim=64,
        input_size=2,
        output_size=2,
        hidden_dims=[256, 256],
        L_embed=5,
        embed_type="fourier",
        activation=nn.ReLU,
        sigma=2.5,
        context_type="Transformer",
    ):
        super().__init__()

        self.context_type = context_type
        layer = FiLMLinear
        self.context_dim = context_dim
        self.L_embed = L_embed

        self.B = nn.Parameter(calcB(m=L_embed, d=2, sigma=sigma), requires_grad=False)
        self.input_size = L_embed * 2 + 2

        # positional embed function
        self.embed_fun = PE_Module(type="fourier", embed_L=self.B)

        self.layers = []
        self.activations = []
        dim_prev = self.input_size
        for h_dim in hidden_dims:
            self.layers.append(layer(dim_prev, h_dim, context_dim=self.context_dim))
            self.activations.append(activation())
            dim_prev = h_dim

        self.layers = nn.ModuleList(self.layers)
        self.activations = nn.ModuleList(self.activations)
        self.final_layer = layer(
            hidden_dims[-1], output_size, context_dim=self.context_dim
        )

    def forward(self, x_in, context):
        """
        B x L x ndim for Transformer (assuming L layers in MLP)
        """
        x_embed = self.embed_fun(x_in)

        x_tmp = x_embed
        for ilayer, layer in enumerate(self.layers):
            x = layer(x_tmp, context[:, ilayer, :].unsqueeze(1))
            x = self.activations[ilayer](x)
            x_tmp = x

        x = self.final_layer(x_tmp, context[:, -1, :].unsqueeze(1))
        return x


class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        _, _, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0,
        has_global_token=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class VisibilityTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        pe_dim,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
        dropout,
        emb_dropout,
        output_dim,
        output_tokens,
        has_global_token,
    ):
        super().__init__()
        """
        dim_value_embedding = dim - pe_dim
        """

        assert output_tokens > 0
        assert dim > pe_dim

        self.input_dim = (
            input_dim  # input value dimension (e.g. =2 visibility map - real and imag)
        )
        self.pe_dim = pe_dim  # postional-encoding dim
        self.dim = dim  # feature embedding dim

        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.output_tokens = output_tokens

        self.global_token = None
        self.has_global_token = has_global_token

        if self.has_global_token:
            self.global_token = nn.Parameter(torch.randn(1, 1, dim))

        self.feat_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.dim - self.pe_dim)
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

        self.output_token_heads = [
            nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.output_dim),
            )
            for _ in range(int(self.output_tokens))
        ]

        self.output_token_heads = nn.ModuleList(self.output_token_heads)

    def forward(self, tokens):
        """
        inputs
        tokens - b x n_token x dim_token_feature (input_dim),
        input : [pose_embed, values]

        outputs
        output_tokens - b x n_out_tokens x dim_out_tokens
        """

        emb_val = self.feat_embedding(
            tokens[..., self.pe_dim :]
        )  # b x n_token x self.dim - self.pe_dim
        emb_token = torch.cat(
            [tokens[..., : self.pe_dim], emb_val], dim=-1
        )  # b x n_token x self.dim

        b, n_token, _ = emb_token.shape

        if self.has_global_token:
            emb_token = torch.cat([self.global_token.repeat(b, 1, 1), emb_token], dim=1)

        emb_token = self.emb_dropout(emb_token)
        transformed_token = self.transformer(emb_token)

        transformed_token_reduced = transformed_token[:, : self.output_tokens, ...]

        out_tokens = []
        for idx_token in range(self.output_tokens):
            out_tokens.append(
                self.output_token_heads[idx_token](
                    transformed_token_reduced[:, idx_token, ...].unsqueeze(1)
                )
            )
        output_tokens = torch.cat(out_tokens, dim=1)

        return output_tokens


def get_halfspace(uv_coords):
    left_halfspace = torch.logical_and(
        torch.logical_or(
            uv_coords[:, :, 0] < 0,
            torch.logical_and(uv_coords[:, :, 0] == 0, uv_coords[:, :, 1] > 0),
        ),
        ~torch.logical_and(uv_coords[:, :, 0] == -0.5, uv_coords[:, :, 1] > 0),
    )
    return left_halfspace


def flip_uv(uv_coords, halfspace):
    halfspace_2d = torch.stack((halfspace, halfspace), axis=-1)
    uv_coords_flipped = torch.where(halfspace_2d, f(-uv_coords), uv_coords)
    return uv_coords_flipped


def f(x):
    return ((x + 0.5) % 1) - 0.5


def conjugate_vis(vis, halfspace):
    # take complex conjugate if flipped uv coords
    # so network doesn't receive confusing gradient information
    vis[halfspace] = torch.conj(vis[halfspace])
    return vis
