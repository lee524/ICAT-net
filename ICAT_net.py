import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from functools import partial




def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim  ##否则为 x 的总维度数加上 dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride ##计算 padding 的大小，确保卷积的输出形状为 ceil(x.size(dim)/stride)，就是算出需要补充多少个padding可以被kernel正好整除
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)##根据维度的不同，构造 padding 元组，以便应用于 PyTorch 的 F.pad 函数。
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class UpSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        act_layer,
        i,
        norm_layer,
        k_size=7, stride=2,
    ):
        super().__init__()



        self.conv_padding_same = (
            (kernel_size - 1) // 2,
            kernel_size - 1 - (kernel_size - 1) // 2,
        )

        self.upsampling = nn.ConvTranspose1d(in_channels, in_channels, k_size, stride = stride,
                                                       padding=(k_size - stride + 1)//2,
                                                       groups = in_channels,
                                                       output_padding=(k_size-stride)%2)  ##这个模块用于将输入的张量进行上采样，将图像或特征图的尺寸沿着所有维度放大两倍。
        self.conv = nn.Conv1d(in_channels = in_channels,out_channels = out_channels, kernel_size = kernel_size,)
        self.relu = act_layer()
        self.iout= i==5
        self.norm = norm_layer(out_channels)





    def forward(self, x):
        x = self.upsampling(x)
        x = F.pad(x, self.conv_padding_same, "constant", 0)
        x = self.conv(x)
        if self.iout:
            x = x.sigmoid()
        else:
            x = self.norm(x)
            x = self.relu(x)

        return x

class MCA(nn.Module):
    def __init__(
        self,  out_dim, kernel_size, qkv_bias,down_bias,head_dim,key_drop_rate,attn_drop_rate,proj_drop_rate,group_dim,act_layer,norm_layer,
    ):
        super().__init__()
        self.num_heads = head_dim

        self.q_proj = nn.Conv1d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1, bias=qkv_bias, groups=group_dim,
        )
        self.k_proj = nn.Conv1d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1, bias=qkv_bias, groups=group_dim
        )
        self.v_proj = nn.Conv1d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1, bias=qkv_bias, groups=group_dim
        )
        self.v1_proj = nn.Conv1d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1, bias=qkv_bias, groups=group_dim
        )
        self.out_proj = nn.Conv1d(
            in_channels=3 * (out_dim), out_channels=out_dim, kernel_size=kernel_size, bias=down_bias, groups=group_dim
        )
        self.aggr = (
            Downsample(
                in_channels=64,
                out_channels=64,
                kernel_size=kernel_size,
                act_layer=act_layer,
                norm_layer=norm_layer,
            ))

        self.k_dropout = nn.Dropout(key_drop_rate)
        self.attn_dropout = nn.Dropout(attn_drop_rate)

        self.proj_dropout = nn.Dropout(proj_drop_rate)
        self.norm = norm_layer(out_dim)
        self.act = act_layer()



    def forward(self, x):
        b, c, l = x.size()
        # if self.i:
        #     cls_tokens = nn.Parameter(torch.zeros(c, 1, l))
        #     x = torch.cat((cls_tokens, x), dim=1)
        #     position_embeddings = nn.Parameter(torch.zeros(1, c + 1, l))
        #     x = x + position_embeddings
        x_res = x
        # q_conv = _auto_pad_1d(x, self.q_proj.kernel_size[0], 1)
        # k_conv = _auto_pad_1d(x, self.k_proj.kernel_size[0], 1)
        # v_conv = _auto_pad_1d(x, self.v_proj.kernel_size[0], 1)
        q = self.q_proj(x)
        x_atten = q.sigmoid()
        x_atten = x * x_atten
        q_ = self.k_proj(x_atten)
        x_attend = x_atten
        x_attend = self.aggr(x_attend)

        v = self.v_proj(x_attend)
        v1 = self.v1_proj(x_attend)

        q_view = q_.view(b, self.num_heads, c // self.num_heads, l)
        k_view = v.view(b, self.num_heads, c // self.num_heads, -1)
        v_view = v1.view(b, self.num_heads, c // self.num_heads, -1)
        k_view = self.k_dropout(k_view)
        N, Nh, E, L = q_view.size()
        q_scaled = q_view / math.sqrt(E)
        attn = (q_scaled.transpose(-1, -2) @ k_view).softmax(dim=-1)##-1, -2表示倒数第一个和倒数第二个互换位置
        attn = self.attn_dropout(attn)

        attn = (attn @ v_view.transpose(-1, -2)).transpose(-1, -2).reshape(b, c, l)

        x = attn + x_atten

        x_conv = torch.cat([x, q_, attn], dim=1)
        x_conv = _auto_pad_1d(x_conv, self.out_proj.kernel_size[0], 1)
        x_conv = self.out_proj(x_conv)
        x = x_conv + x

        x = self.proj_dropout(x)
        x = self.norm(x)
        x = self.act(x)

        return x



class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act_layer, norm_layer):
        super(Downsample, self).__init__()


        self.down_proj = nn.Conv1d(
            in_channels=2 * in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, groups=2
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer()


    def forward(self, x):
        odd_parts = x[:, :, 1::2]  # 奇数部分
        even_parts = x[:, :, 0::2]  # 偶数部分

        # 合并两个100*16*4096的tensor成100*32*4096
        x = torch.cat((odd_parts, even_parts), dim=1)
        x = _auto_pad_1d(x, self.down_proj.kernel_size[0], 1)
        x = self.down_proj(x)
        x = self.norm(x)
        x = self.act(x)

        return x



class ICAT_NET(nn.Module):
    """
    Seismogram Transformer.
    """

    def __init__(
        self,
        in_channels=3,
        d_channels=[8, 16, 32,32,64,64],
        d_kernel_sizes=[11, 9, 7, 7,5,3],
        out_channels=[64,32,32, 16, 8,3],
        out_kers=[ 3, 5,7,7,9,11],
        stem_kernel_sizes = [5, 3,2,3],
        key_drop_rate=0.1,
        attn_drop_rate=0.1,
        proj_drop_rate=0.2,
        qkv_bias=True,
        head_dim = [16,8,4,2],
        group_dims = [8,4,2,1],
        downbias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,

        **kwargs,
    ):
        super().__init__()

        self.dsampl = nn.ModuleList()

        for i, (inc, outc, kers) in enumerate(zip([in_channels] + d_channels[:-1], d_channels, d_kernel_sizes

        )):
            downsample_instance =Downsample(
                in_channels=inc,
                out_channels=outc,
                kernel_size=kers,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            dlayer_modules = []
            dlayer_modules.append(downsample_instance)

            self.dsampl.append(nn.Sequential(*dlayer_modules))



        self.use_checkpoint = use_checkpoint

        self.ICAT = nn.Sequential(
            *[
                MCA(
                    out_dim=64,
                    kernel_size=kers,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    qkv_bias=qkv_bias,
                    down_bias=downbias,
                    head_dim=headdim,
                    key_drop_rate=key_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    proj_drop_rate=proj_drop_rate,
                    group_dim=group_dim,

                )
                for i, (kers, headdim, group_dim) in enumerate(zip(

                stem_kernel_sizes,
                head_dim,
                group_dims,

        ))
            ]
        )


        self.out_up_samplings=nn.ModuleList()


        for i, (inc, ker, outc,) in enumerate(zip(
                d_channels[-1:] + out_channels,
                out_kers,
                out_channels,

        )):
            UP_sample = UpSamplingBlock(
                    in_channels=inc*2,
                    kernel_size=ker,
                    out_channels=outc,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    i=i,
                )
            down_layer_modules = []
            down_layer_modules.append(UP_sample)
            self.out_up_samplings.append(nn.Sequential(*down_layer_modules))




        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02) ##nn.init.trunc_normal_ 是 PyTorch 提供的截断正态分布初始化方法，它将权重初始化为截断的正态分布，标准差为 0.02。
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(
            m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d)
        ):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):

        x_cat=[]
        for d_layer in self.dsampl:
            x = d_layer(x)
            x_cat.append(x)

        if self.use_checkpoint and not (
                torch.jit.is_tracing() or torch.jit.is_scripting()
        ):
            x = checkpoint.checkpoint(self.ICAT, x)
        else:
            x = self.ICAT(x)
        for i, down_layer in enumerate(self.out_up_samplings):
            x =torch.cat(( x , x_cat[5-i]), dim=1)
            x = down_layer(x)



        return x




# if __name__ == '__main__':
#     x = torch.Tensor(100, 3, 8192)
#     carafe = ICAT_NET(in_channels=3)
#     oup = carafe(x)
#     print(oup.size())
#     total_params = sum(p.numel() for p in carafe.parameters() if p.requires_grad)
#     print("Total parameters:", total_params)
























