import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# Window Partition / Reverse
# -----------------------
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# -----------------------
# Window Attention
# -----------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2,-1)).softmax(dim=-1)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        return x

# -----------------------
# MLP
# -----------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# -----------------------
# SwinIR Block
# -----------------------
class SwinIRBlock(nn.Module):
    def __init__(self, dim=96, input_resolution=(32,32), num_heads=3, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Patch embedding / unembedding
        self.embed = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.unembed = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        # Embed to higher dimension
        x = self.embed(x)  # (B, dim, H, W)
        x = x.permute(0,2,3,1)  # (B,H,W,dim)

        # Pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0,0,0,pad_w,0,pad_h))

        # Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))

        # Partition windows
        x_windows = window_partition(x, self.window_size)  # (num_windows*B, ws*ws, dim)
        x_windows = self.attn(self.norm1(x_windows))

        # Merge windows
        x = window_reverse(x_windows, self.window_size, x.shape[1], x.shape[2])

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1,2))

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        # Residual + MLP
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0,3,1,2)  # (B, dim, H, W)

        # Unembed to RGB
        x = self.unembed(x)

        # Final residual
        x = x + shortcut

        # Clamp output
        x = torch.clamp(x, 0, 1)
        return x
