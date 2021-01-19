"""Bilateral Segmentation Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.base_models.resnet import resnet18, resnet101
from core.models.base_models.resnetv1b import resnet101_v1s, resnet50_v1s
from core.nn import _ConvBNReLU
from .segbase import SegBaseModel

__all__ = ['augBiSeNet', 'get_augbisenet', 'get_augbisenet_resnet18_citys', 'get_augbisenet_resnet101_citys', 'get_augbisenet_resnet50_citys']


class augBiSeNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet18', aux = False, pretrained_base=True, **kwargs):
        super(augBiSeNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.left_path = _SpatialPath(3, 128, nclass, **kwargs)
        self.attention = _DAHead(2048, nclass, **kwargs)
        inter_channels = 128
        self.global_context = _GlobalAvgPooling(512, inter_channels, **kwargs)

        self.ffm = FeatureFusion(256, 256, 4, **kwargs)
        self.head = _BiSeHead(256, 64, nclass, **kwargs)

        self.__setattr__('exclusive', ['left_path', 'attention', 'global_context', 'ffm', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        left_out = self.left_path(x)
        _, _, _, c4 = self.base_forward(x)
        attention_out = self.attention(c4)
        global_out = self.global_context(attention_out)
        fusion_out = self.ffm(left_out, global_out)
        outputs = []
        x = self.head(fusion_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)


        return tuple(outputs)

class _SpatialPath(nn.Module):
    """Spatial path"""

    def __init__(self, in_channels, out_channels, nclass, aux = False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_SpatialPath, self).__init__()
        inter_channels = 64
        self.conv7x7 = _ConvBNReLU(in_channels, inter_channels, 7, 2, 3, norm_layer=norm_layer)
        self.conv3x3_1 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)
        self.conv3x3_2 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)
        self.conv1x1 = _ConvBNReLU(inter_channels, out_channels, 1, 1, 0, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)

        return x

class _BiSeHead(nn.Module):
    def __init__(self, in_channels, inter_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_BiSeHead, self).__init__()
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        # self.out = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Conv2d(inter_channels, nclass, 1)
        # )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c


        return feat_fusion

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out

class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class _GlobalAvgPooling(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(_GlobalAvgPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = _ConvBNReLU(in_channels, out_channels, 1, 1, 0, norm_layer=norm_layer, **kwargs)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(out_channels, out_channels // reduction, 1, 1, 0, norm_layer=norm_layer),
            _ConvBNReLU(out_channels // reduction, out_channels, 1, 1, 0, norm_layer=norm_layer),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out


def get_augbisenet(dataset='pascal_voc', backbone='resnet18', pretrained=False, root='~/.torch/models',
                pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = augBiSeNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(0)
        model.load_state_dict(torch.load(get_model_file('augbisenet_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device), False)
    return model


def get_augbisenet_resnet18_citys(**kwargs):
    return get_augbisenet('pascal_voc', 'resnet18', **kwargs)

def get_augbisenet_resnet101_citys(**kwargs):
    return get_augbisenet('pascal_voc', 'resnet101', **kwargs)

def get_augbisenet_resnet50_citys(**kwargs):
    return get_augbisenet('pascal_voc', 'resnet50', **kwargs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 224, 224)
    model = BiSeNet(19, backbone='resnet18')
    print(model.exclusive)
