import torch
from torch import nn
from einops import rearrange


def empty(tensor):
    return tensor.numel() == 0


class FastAttention(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, q, k, v):
        device = q.device

        out = linear_attention(q, k, v)
        return out
    
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


class Attention(nn.Module):
    def __init__( 
        self,
        c_in,
        c_out,
        dim_h,
        dim_w,
        dropout = 0.,
        heads=1,
        qkv_bias = False,
        attn_out_bias = True,
    ):
        super().__init__()
        self.dim = dim_h * dim_w
        dim_head = self.dim
        inner_dim = dim_head * heads
        self.inner_dim = dim_head * heads
        self.fast_attention = FastAttention()
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1, dropout=0.5, batch_first=True)
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(self.dim, inner_dim, bias = qkv_bias), nn.BatchNorm1d(2))
        self.to_k = nn.Sequential(nn.Linear(self.dim, inner_dim, bias = qkv_bias), nn.BatchNorm1d(2))
        self.to_v = nn.Sequential(nn.Linear(self.dim, inner_dim, bias = qkv_bias), nn.BatchNorm1d(2))
        self.to_out = nn.Linear(inner_dim, self.dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(2)

    def forward(self, x, **kwargs):
        b, n, H, W, h = *x.shape, self.heads
        q, k, v = self.to_q(x.reshape(b, n, -1)), self.to_k(x.reshape(b, n, -1)), self.to_v(x.reshape(b, n, -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        attn_outs = []
        if not empty(q):
            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.batch_norm(self.dropout(out).reshape(b, n, H, W))


class LAG(nn.Module):
    def __init__(self, c_in, c_out, dim_h, dim_w):
        super().__init__()
        
        self.lin_attention = Attention(c_in, c_out, dim_h, dim_w)
        self.gating = nn.Sequential(
                nn.Conv2d(c_in, c_in, 1, stride=1, bias=False), nn.GELU(), nn.BatchNorm2d(2))
        self.conv = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=1, bias=False), nn.BatchNorm2d(2))

    def forward(self, x):
        x = x + self.conv(torch.mul(self.lin_attention(x), self.gating(x)))
        return x


class FFN(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        
        self.upper = nn.Sequential(
            nn.Conv2d(c_in, c_in, 1, stride=1, bias=False),
             nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False),
        )
        self.lower = nn.Sequential(
            nn.Conv2d(c_in, c_in, 1, stride=1, bias=False),
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False),
            nn.GELU(),
        )
        self.conv = nn.Conv2d(c_in, c_out, 1, stride=1, bias=False)

    def forward(self, x):
        x = x + self.conv(torch.mul(self.upper(x), self.lower(x)))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, c_in, c_out, dim_h, dim_w):
        super().__init__()
        
        self.lag = LAG(c_in, c_out, dim_h, dim_w)
        self.ffn = FFN(c_out, c_out)
        self.lnorm1 = nn.LayerNorm([c_in, dim_h, dim_w])
        self.lnorm2 = nn.LayerNorm([c_out, dim_h, dim_w])

    def forward(self, x):
        x = self.lnorm2(self.ffn(self.lnorm1(self.lag(x))))
        return x


class VisT_small(nn.Module):
    def __init__(self):
        super().__init__()
        torch.cuda.set_device(1)

        self.encoding = nn.Sequential(nn.Conv2d(2, 2, 7, stride=1, padding=3, bias=False), nn.BatchNorm2d(2))
        self.tb = TransformerBlock(2, 2, 65, 128)
        self.decoding = nn.Sequential(nn.Conv2d(2, 2, 7, stride=1, padding=3, bias=False))
        

    def forward(self, x):
        x = self.tb(self.encoding(x))
        x = self.decoding(x)
        return x


class VisibilityTFormer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoding = nn.Sequential(nn.Conv2d(2, 2, 7, stride=1, padding=3, bias=False), nn.BatchNorm2d(2))
        self.tb1 = TransformerBlock(2, 2, 65, 128)
        self.tb2 = TransformerBlock(4, 4, 33, 64)
        self.tb3 = TransformerBlock(8, 8, 17, 32)
        self.tb4 = TransformerBlock(16, 16, 9, 16)
        
        self.down1 = nn.Sequential(nn.Conv2d(2, 4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(4))
        self.down2 = nn.Sequential(nn.Conv2d(4, 8, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(8))
        self.down3 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16))
        
        self.tb5 = TransformerBlock(8, 8, 17, 32)
        self.tb6 = TransformerBlock(4, 4, 33, 64)
        self.tb7 = TransformerBlock(2, 2, 65, 128)
        
        self.up1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.up2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(8, 8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
        )
        self.up3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(4, 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
        )
        
        self.conv1 = nn.Sequential(nn.Conv2d(24, 8, 1, stride=1, bias=False), nn.BatchNorm2d(8))
        self.conv2 = nn.Sequential(nn.Conv2d(12, 4, 1, stride=1, bias=False), nn.BatchNorm2d(4))
        self.conv3 = nn.Sequential(nn.Conv2d(6, 2, 1, stride=1, bias=False), nn.BatchNorm2d(2))
        
        self.decoding = nn.Sequential(nn.Conv2d(2, 2, 7, stride=1, padding=3, bias=False))
        

    def forward(self, x):
        x1 = self.tb1(self.encoding(x))
        x2 = self.tb2(self.down1(x1))
        x3 = self.tb3(self.down2(x2))
        x4 = self.tb4(self.down3(x3))

        x5 = self.tb5(self.conv1(torch.concatenate([self.up1(x4)[:, :, :-1, :], x3], dim=1)))
        x6 = self.tb6(self.conv2(torch.concatenate([self.up2(x5)[:, :, :-1, :], x2], dim=1)))
        x7 = self.tb7(self.conv3(torch.concatenate([self.up3(x6)[:, :, :-1, :], x1], dim=1)))
        
        x = self.decoding(x7)
        return x


#VisibilityTFormer = torch.compile(VisTFormer())



class Attention_saver(nn.Module):
    def __init__( 
        self,
        c_in,
        c_out,
        dim_h,
        dim_w,
        dropout = 0.,
        heads=1,
        qkv_bias = False,
        attn_out_bias = True,
    ):
        super().__init__()
        self.dim = dim_h * dim_w
        dim_head = self.dim
        inner_dim = dim_head * heads
        self.inner_dim = dim_head * heads
        # self.fast_attention = FastAttention()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1, dropout=0.5, batch_first=True)
        self.heads = heads

        self.to_q = nn.Sequential(nn.Linear(self.dim, inner_dim, bias = qkv_bias), nn.BatchNorm1d(2))
        self.to_k = nn.Sequential(nn.Linear(self.dim, inner_dim, bias = qkv_bias), nn.BatchNorm1d(2))
        self.to_v = nn.Sequential(nn.Linear(self.dim, inner_dim, bias = qkv_bias), nn.BatchNorm1d(2))
        self.to_out = nn.Linear(inner_dim, self.dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm2d(2)

    def forward(self, x, **kwargs):
        b, n, H, W, h = *x.shape, self.heads
        q, k, v = self.to_q(x.reshape(b, n, -1)), self.to_k(x.reshape(b, n, -1)), self.to_v(x.reshape(b, n, -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        attn_outs = []
        if not empty(q):
            out = self.multihead_attention(q.view(-1, 2, self.dim), k.view(-1, 2, self.dim), v.view(-1, 2, self.dim), need_weights=False)
            attn_outs.append(out[0])

        out = torch.cat(attn_outs, dim = 1)
        # out = rearrange(out, 'b n h d -> b n (h d)')
        out =  self.to_out(out)
        return self.batch_norm(self.dropout(out).reshape(b, n, H, W))
