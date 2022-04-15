"""
RegNet models from https://arxiv.org/abs/2103.06877 and
https://github.com/facebookresearch/pycls/blob/main/pycls/models/anynet.py

TODO:
Add scaling rules from RegNetZ
Add correct initialization for ResNet
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'RegNet',
    'regnetx_6p4gf',
    'regnety_200mf',
    'regnety_800mf',
    'regnety_3p2gf',
    'regnety_4gf',
    'regnety_6p4gf',
    'regnety_8gf',
    'regnety_16gf'
]

def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
        
def conv_bn_act(
    nin,
    nout,
    kernel_size,
    stride=1,
    groups=1,
    activation=nn.ReLU(inplace=True)
):
    padding = (kernel_size - 1) // 2
    # regular convolution and batchnorm
    layers = [
        nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(nout)
    ]
    
    # add activation if necessary
    if activation:
        layers.append(activation)
        
    return nn.Sequential(*layers)

class Resample2d(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        stride=1,
        activation=None
    ):
        super(Resample2d, self).__init__()
        
        # convolution to downsample channels, if needed
        if nin != nout or stride > 1:
            self.conv = conv_bn_act(nin, nout, kernel_size=1, stride=stride, activation=activation)
        else:
            self.conv = nn.Identity()
            
    def forward(self, x):
        x = self.conv(x)
        return x
    
class SqueezeExcite(nn.Module):
    def __init__(self, nin):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AvgPool2d((1, 1))
        
        # hard code the squeeze factor at 4
        ns = nin // 4
        self.se = nn.Sequential(
            nn.Conv2d(nin, ns, 1, bias=True), # squeeze
            nn.ReLU(inplace=True),
            nn.Conv2d(ns, nin, 1, bias=True), # excite
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.se(self.avg_pool(x))

class Stem(nn.Module):
    """
    Simple input stem.
    """
    def __init__(self, w_in, w_out, kernel_size=3):
        super(Stem, self).__init__()
        self.cbr = conv_bn_act(w_in, w_out, kernel_size, stride=2)

    def forward(self, x):
        x = self.cbr(x)
        return x

class Bottleneck(nn.Module):
    """
    ResNet-style bottleneck block.
    """
    def __init__(
        self,
        w_in,
        w_out,
        bottle_ratio=1,
        groups=1,
        stride=1,
        use_se=False
    ):
        super(Bottleneck, self).__init__()
        w_b = int(round(w_out * bottle_ratio))
        self.a = conv_bn_act(w_in, w_b, 1)
        self.b = conv_bn_act(w_b, w_b, 3, stride=stride, groups=groups)
        self.se = SqueezeExcite(w_b) if use_se else None
        self.c = conv_bn_act(w_b, w_out, 1, activation=None)
        self.c[1].final_bn = True # layer 1 is the BN layer

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
            
        return x

class BottleneckBlock(nn.Module):
    def __init__(
        self,
        w_in,
        w_out,
        bottle_ratio,
        groups=1,
        stride=1,
        use_se=False
    ):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = Bottleneck(w_in, w_out, bottle_ratio, groups, stride, use_se)
        self.downsample = Resample2d(w_in, w_out, stride=stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.downsample(x) + self.bottleneck(x))

class Stage(nn.Module):
    def __init__(
        self,
        block,
        w_in,
        w_out,
        depth,
        bottle_r=1,
        groups=1,
        stride=1,
        use_se=False
    ):
        super(Stage, self).__init__()

        assert depth > 0, "Each stage has minimum depth of 1 layer."

        for i in range(depth):
            if i == 0:
                # only the first layer in a stage 
                # has expansion and downsampling
                layer = block(w_in, w_out, bottle_r, groups, stride, use_se=use_se)
            else:
                layer = block(w_out, w_out, bottle_r, groups, use_se=use_se)

            self.add_module(f'block{i + 1}', layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)

        return x
    
class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

class RegNet(nn.Module):
    """
    Simplest RegNetX/Y-like encoder without classification head
    """
    def __init__(
        self,
        cfg,
        im_channels=1,
        output_stride=32,
        block=BottleneckBlock,
        normalize=False,
        output_dim=0,
        hidden_mlp=0,
        nmb_prototypes=0,
        eval_mode=False
    ):
        super(RegNet, self).__init__()
        
        assert output_stride in [16, 32]
        if output_stride == 16:
            cfg.strides[-1] = 1

        # make the stages with correct widths and depths
        self.cfg = cfg
        groups = cfg.groups
        depths = cfg.depths
        w_ins = [cfg.w_stem] + cfg.widths[:-1]
        w_outs = cfg.widths
        strides = cfg.strides
        use_se = cfg.use_se
        
        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        
        self.stem = Stem(im_channels, cfg.w_stem, kernel_size=3)
        
        for i in range(cfg.num_stages):
            stage = Stage(block, w_ins[i], w_outs[i], depths[i],
                          groups=groups[i], stride=strides[i], use_se=use_se)

            self.add_module(f'stage{i + 1}', stage)
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        # projection head
        num_out_filters = w_outs[-1]
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
            
        self.apply(init_weights)

    def forward_backbone(self, x):
        x = self.padding(x)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
                
            start_idx = end_idx
            
        return self.forward_head(output)
    
class RegNetConfig:
    w_stem = 32
    bottle_ratio = 1
    strides = [2, 2, 2, 2]

    def __init__(
        self,
        depth,
        w_0,
        w_a,
        w_m,
        group_w,
        q=8,
        use_se=False,
        **kwargs
    ):
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
        self.depth = depth
        self.w_0 = w_0
        self.w_a = w_a
        self.w_m = w_m
        self.group_w = group_w
        self.q = q
        self.use_se = use_se
        
        for k,v in kwargs.items():
            setattr(self, k, v)

        self.set_params()
        self.adjust_params()

    def adjust_params(self):
        """
        Adjusts widths and groups to guarantee compatibility.
        """
        ws = self.widths
        gws = self.group_widths
        b = self.bottle_ratio

        adj_ws = []
        adj_groups = []
        for w, gw in zip(ws, gws):
            # group width can't exceed width
            # in the bottleneck
            w_b = int(max(1, w * b))
            gw = int(min(gw, w_b))

            # fix width s.t. it is always divisible by
            # group width for any bottleneck_ratio
            m = np.lcm(gw, b) if b > 1 else gw
            w_b = max(m, int(m * round(w_b / m)))
            w = int(w_b / b)

            adj_ws.append(w)
            adj_groups.append(w_b // gw)

        assert all(w * b % g == 0 for w, g in zip(adj_ws, adj_groups))
        self.widths = adj_ws
        self.groups = adj_groups

    def set_params(self):
        """
        Generates RegNet parameters following:
        https://arxiv.org/pdf/2003.13678.pdf
        """
        # capitals for complete sets
        # widths of blocks
        U = self.w_0 + np.arange(self.depth) * self.w_a # eqn (2)

        # quantize stages by solving eqn (3) for sj
        S = np.round(
            np.log(U / self.w_0) / np.log(self.w_m)
        )

        # block widths from eqn (4)
        W = self.w_0 * np.power(self.w_m, S)

        # round the widths to nearest factor of q
        # (makes best use of tensor cores)
        W = self.q * np.round(W / self.q).astype(int)

        # group stages by the quantized widths, use
        # as many stages as there are unique widths
        W, D = np.unique(W, return_counts=True)
        assert len(W) == 4, "Bad parameters, only 4 stage networks allowed!"

        self.num_stages = len(W)
        self.group_widths = len(W) * [self.group_w]
        self.widths = W.tolist()
        self.depths = D.tolist()
        
def regnetx_6p4gf(**kwargs):
    params = {
        'depth': 17, 'w_0': 184, 'w_a': 60.83,
        'w_m': 2.07, 'group_w': 56
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_200mf(**kwargs):
    params = {
        'depth': 13, 'w_0': 24, 'w_a': 36.44,
        'w_m': 2.49, 'group_w': 8
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_800mf(**kwargs):
    params = {
        'depth': 14, 'w_0': 56, 'w_a': 38.84,
        'w_m': 2.4, 'group_w': 16
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_3p2gf(**kwargs):
    params = {
        'depth': 21, 'w_0': 80, 'w_a': 42.63,
        'w_m': 2.66, 'group_w': 24
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_4gf(**kwargs):
    params = {
        'depth': 22, 'w_0': 96, 'w_a': 31.41,
        'w_m': 2.24, 'group_w': 64
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_6p4gf(**kwargs):
    params = {
        'depth': 25, 'w_0': 112, 'w_a': 33.22,
        'w_m': 2.27, 'group_w': 72, 'use_se': True
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_8gf(**kwargs):
    params = {
        'depth': 17, 'w_0': 192, 'w_a': 76.82,
        'w_m': 2.19, 'group_w': 56, 'use_se': True
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_16gf(**kwargs):
    params = {
        'depth': 18, 'w_0': 200, 'w_a': 106.23,
        'w_m': 2.48, 'group_w': 112, 'use_se': True
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)