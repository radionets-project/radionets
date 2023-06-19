import torch
from torch import nn
from einops import rearrange
from math import sqrt
from radionets.dl_framework.model import SRBlock

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
    #n = torch.tensor(k.shape[-2], device="cuda").repeat(25, 1, 1, 2)
    #k_cumsum = k.sum(-2)
    upper = (v + q @ (k.transpose(2, 3) @ v))
    #lower = n + q * k_cumsum.view(25, 1, 1, 2)
    out = upper# / lower
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


class SelfAttentionMultiHead(nn.Module):
    def __init__(self, ni, nheads):
        super().__init__()
        self.nheads = nheads
        self.scale = sqrt(ni / nheads)
        self.norm = nn.BatchNorm2d(ni)
        self.qkv = nn.Linear(ni, ni * 3)
        self.proj = nn.Linear(ni, ni)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inp):
        n, c, h, w = inp.shape
        x = self.norm(inp).view(n, c, -1).transpose(1, 2)
        x = self.qkv(x)
        x = rearrange(x, "n s (h d) -> (n h) s d", h=self.nheads)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n s (h d) -> n h s d', h = self.nheads), (q, k, v))
        x = linear_attention(q, k, v)
        #s = (q@k.transpose(1, 2)) / self.scale
        #x = self.dropout(s.softmax(dim=-1)) @ v
        #x = rearrange(x, "(n h) s d -> n s (h d)", h=self.nheads)
        x = rearrange(x, "n h s d -> n s (h d)", h=self.nheads)
        x = self.dropout(self.proj(x)).transpose(1, 2).reshape(n, c, h, w)
        return x + inp


class LAG(nn.Module):
    def __init__(self, ni, nheads):
        super().__init__()
        
        self.lin_attention = SelfAttentionMultiHead(ni, nheads)
        self.gating = nn.Sequential(
                nn.Conv2d(ni, ni, 1, stride=1, bias=False),
                nn.BatchNorm2d(ni),
                nn.GELU(),
                )
        self.conv = nn.Sequential(
                nn.Conv2d(ni, ni, 1, stride=1, bias=False),
                nn.BatchNorm2d(ni),
                )

    def forward(self, x):
        return x + self.conv(torch.mul(self.lin_attention(x), self.gating(x)))


class FFN(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.n_inner = 64
        
        self.upper = nn.Sequential(
            nn.Conv2d(ni, self.n_inner, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.n_inner),
            nn.Conv2d(self.n_inner, self.n_inner, kernel_size=3, padding=1, groups=ni, bias=False),
            nn.BatchNorm2d(self.n_inner),
        )
        self.lower = nn.Sequential(
            nn.Conv2d(ni, self.n_inner, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.n_inner),
            nn.Conv2d(self.n_inner, self.n_inner, kernel_size=3, padding=1, groups=ni, bias=False),
            nn.BatchNorm2d(self.n_inner),
            nn.GELU(),
        )
        self.conv = nn.Conv2d(self.n_inner, ni, 1, stride=1, bias=False)

    def forward(self, x): 
        x = x + self.conv(torch.mul(self.upper(x), self.lower(x)))
        return x


class TransformerBlock(nn.Module):
     def __init__(self, ni, nheads):
        super().__init__()
        
        self.lag = LAG(ni, nheads)
        self.ffn = FFN(ni)
        self.norm1 = nn.LayerNorm([ni, 65, 128])
        self.norm2 = nn.LayerNorm([ni, 65, 128])

     def forward(self, x):
        x = self.norm2(self.ffn(self.norm1(self.lag(x))))
        return x


class VisT_small(nn.Module):
    def __init__(self): 
        super().__init__()
        #torch.cuda.set_device(1)

        self.encoding = nn.Sequential(
                nn.Conv2d(2, 2, 7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(2),
                )
        self.tb = TransformerBlock(2, 1)
        self.decoding = nn.Sequential (nn.Conv2d(2, 2, 7, stride=1, padding=3, bias=False))
        

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


class SRResNet_attention(nn.Module):
     def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
            LAG(64, 1),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(nn.Conv2d(64, 2, 9, stride=1, padding=4, groups=2))


     def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)

        return torch.cat([x0, x1], dim=1)


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
