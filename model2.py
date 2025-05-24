## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
##---------- Prompt Gen Module -----------------------
"""
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, num_heads=4):
        super(PromptGenBlock, self).__init__()
        self.prompt_rain = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.prompt_snow = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len * 2)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        # 新增注意力層
        self.attention = nn.MultiheadAttention(embed_dim=prompt_dim, num_heads=num_heads)

    def forward(self, x, de_id=None):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))  # (B, C)，圖像全局特徵嵌入

        # 加入注意力機制
        emb = emb.unsqueeze(0)  # (1, B, C)，增加時間維度以適配 MultiheadAttention
        attn_output, _ = self.attention(emb, emb, emb)  # 自注意力，(1, B, C)
        emb = attn_output.squeeze(0)  # (B, C)，恢復原始形狀

        # 計算雨和雪的 Prompt 權重
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)  # (B, prompt_len * 2)
        rain_weights = prompt_weights[:, :self.prompt_rain.size(1)]  # (B, prompt_len)
        snow_weights = prompt_weights[:, self.prompt_rain.size(1):]  # (B, prompt_len)

        # 根據特徵生成 Prompt
        prompt_rain = (rain_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 
                       self.prompt_rain.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1))
        prompt_snow = (snow_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 
                       self.prompt_snow.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1))

        # 訓練時使用 de_id 選擇 Prompt，測試時根據權重判斷
        if de_id is not None:  # 訓練階段
            prompt = torch.zeros_like(prompt_rain, device=x.device)
            task = torch.empty(B, dtype=torch.long, device=x.device)
            for b in range(B):
                if de_id[b] == 3:
                    prompt[b] = torch.sum(prompt_rain[b], dim=0)
                    task[b] = 0  # 0 表示 "derain"
                elif de_id[b] == 4:
                    prompt[b] = torch.sum(prompt_snow[b], dim=0)
                    task[b] = 1  # 1 表示 "desnow"
                else:
                    raise ValueError(f"de_id[{b}] must be 3 (derain) or 4 (desnow), got {de_id[b]}")
            # 將 task 映射為字串列表
            task_str = ["derain" if t == 0 else "desnow" for t in task.tolist()]
        else:  # 測試階段
            rain_score = torch.sum(rain_weights, dim=1, keepdim=True)
            snow_score = torch.sum(snow_weights, dim=1, keepdim=True)
            # 使用 torch.where 選擇索引（0 或 1）
            task = torch.where(rain_score > snow_score, 0, 1)  # (B, 1)
            task = task.squeeze(-1)  # (B,)
            prompt = torch.where((rain_score > snow_score).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                prompt_rain, prompt_snow)
            prompt = torch.sum(prompt, dim=1)
            # 將 task 映射為字串列表
            task_str = ["derain" if t == 0 else "desnow" for t in task.tolist()]

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt, task_str
    
##########################################################if attention takes too much time##############################################
"""

class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        # Learnable parameter for rain prompt, initialized with random values
        self.prompt_rain = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        # Learnable parameter for snow prompt, initialized with random values
        self.prompt_snow = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        # Linear layer to compute prompt weights based on input embedding
        self.linear_layer = nn.Linear(lin_dim, prompt_len * 2)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, de_id=None):
        # Extract batch size, channels, height, and width from input tensor
        B, C, H, W = x.shape
        # Compute spatial mean embedding from input for weight calculation
        emb = x.mean(dim=(-2, -1))

        # Compute softmax-normalized weights for rain and snow prompts
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        rain_weights = prompt_weights[:, :self.prompt_rain.size(1)]
        snow_weights = prompt_weights[:, self.prompt_rain.size(1):]

        # Weighted sum of rain prompts based on rain weights
        prompt_rain = (rain_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 
                       self.prompt_rain.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1))
        # Weighted sum of snow prompts based on snow weights
        prompt_snow = (snow_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 
                       self.prompt_snow.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1))

        # Training phase: Use de_id to select prompt
        if de_id is not None:
            prompt = torch.zeros_like(prompt_rain, device=x.device)
            task = torch.empty(B, dtype=torch.long, device=x.device)
            for b in range(B):
                if de_id[b] == 3:
                    prompt[b] = torch.sum(prompt_rain[b], dim=0)
                    task[b] = 0  # 0 indicates "derain"
                elif de_id[b] == 4:
                    prompt[b] = torch.sum(prompt_snow[b], dim=0)
                    task[b] = 1  # 1 indicates "desnow"
            # Map task indices to string labels
            task_str = ["derain" if t == 0 else "desnow" for t in task.tolist()]
        else:  # Testing phase: Determine task based on weights
            rain_score = torch.sum(rain_weights, dim=1, keepdim=True)
            snow_score = torch.sum(snow_weights, dim=1, keepdim=True)
            # print(f"rain:{rain_score}, snow:{snow_score}")
            # Select task based on higher score (0 for derain, 1 for desnow)
            task = torch.where(rain_score > snow_score, 0, 1)
            task = task.squeeze(-1)
            # Select prompt based on score comparison
            prompt = torch.where((rain_score > snow_score).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                 prompt_rain, prompt_snow)
            prompt = torch.sum(prompt, dim=1)
            # Map task indices to string labels
            task_str = ["derain" if t == 0 else "desnow" for t in task.tolist()]

        # Interpolate prompt to match input height and width
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        # Apply 3x3 convolution to refine the prompt
        prompt = self.conv3x3(prompt)
        return prompt, task_str

##########################################################################
##---------- PromptIR -----------------------
class PromptIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[6,8,8,10], 
        num_refinement_blocks=6, 
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
    ):
        super(PromptIR, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
        
        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))
        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param, task = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param, _ = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param, _ = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, task