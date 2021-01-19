"""Bilateral Segmentation Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.base_models.resnet import resnet18
#from core.nn import _ConvBNReLU

__all__ = ['BiSeNetv2', 'get_bisenetv2', 'get_bisenetv2_resnet18_pascal_voc']

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = norm_layer(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

class BiSeNetv2(nn.Module):
    def __init__(self, nclass, backbone='resnet18', aux=True, jpu=False, pretrained_base=True, **kwargs):
        super(BiSeNetv2, self).__init__()
        self.aux = aux
        self.detail = DetailBranch(**kwargs)
        self.segment = SegmentBranch(**kwargs)
        self.bga = BGALayer(**kwargs)

        self.head = SegmentHead(128, 1024, nclass, up_factor=8, aux=False,**kwargs)
        if self.aux:
            self.aux2 = SegmentHead(16, 128, nclass, up_factor=4,**kwargs)
            self.aux3 = SegmentHead(32, 128, nclass, up_factor=8,**kwargs)
            self.aux4 = SegmentHead(64, 128, nclass, up_factor=16,**kwargs)
            self.aux5_4 = SegmentHead(128, 128, nclass, up_factor=32,**kwargs)

        self.init_weights()

        self.__setattr__('exclusive',
                         ['detail', 'segment', 'bga', 'head', 'aux2', 'aux3', 'aux4', 'aux5_4'] if aux else [
                             'detail', 'segment', 'bga', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        logits = self.head(feat_head)
        outputs = []
        outputs.append(logits)

        if self.aux:
            logits_aux2 = self.aux2(feat2)
            # outputs.append(logits_aux2)
            logits_aux3 = self.aux3(feat3)
            # outputs.append(logits_aux3)
            logits_aux4 = self.aux4(feat4)
            # outputs.append(logits_aux4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            # outputs.append(logits_aux5_4)
            return tuple(outputs)
        # pred = logits.argmax(dim=1)
        return tuple(outputs)

    def init_weights(self,**kwargs):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

class DetailBranch(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2, norm_layer=nn.BatchNorm2d),
            ConvBNReLU(64, 64, 3, stride=1, norm_layer=nn.BatchNorm2d),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2, norm_layer=nn.BatchNorm2d),
            ConvBNReLU(64, 64, 3, stride=1, norm_layer=nn.BatchNorm2d),
            ConvBNReLU(64, 64, 3, stride=1, norm_layer=nn.BatchNorm2d),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2, norm_layer=nn.BatchNorm2d),
            ConvBNReLU(128, 128, 3, stride=1, norm_layer=nn.BatchNorm2d),
            ConvBNReLU(128, 128, 3, stride=1, norm_layer=nn.BatchNorm2d),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat

class BGALayer(nn.Module):
    def __init__(self,**kwargs):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(size = (45, 80))
        self.up2 = nn.Upsample(size = (45, 80))
        self.up1_1 = nn.Upsample(size=(80, 45))
        self.up2_2 = nn.Upsample(size=(80, 45))
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        if dsize[0] > dsize[1]:
            right1 = self.up1_1(right1)
        else:
            right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        if dsize[0] > dsize[1]:
            right = self.up2_2(right1)
        else: right = self.up2(right)
        out = self.conv(left + right)
        return out

class StemBlock(nn.Module):

    def __init__(self,norm_layer=nn.BatchNorm2d, **kwargs):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2, norm_layer=nn.BatchNorm2d)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0, norm_layer=nn.BatchNorm2d),
            ConvBNReLU(8, 16, 3, stride=2, norm_layer=nn.BatchNorm2d),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat

class SegmentBranch(nn.Module):

    def __init__(self,**kwargs):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1, norm_layer=nn.BatchNorm2d)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes * up_factor * up_factor
        if aux:
            self.conv_out = nn.Sequential(
                ConvBNReLU(mid_chan, up_factor * up_factor, 3, stride=1, norm_layer=nn.BatchNorm2d),
                nn.Conv2d(up_factor * up_factor, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_chan, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6,**kwargs):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1, norm_layer=nn.BatchNorm2d)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6,**kwargs):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1, norm_layer=nn.BatchNorm2d)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat

class CEBlock(nn.Module):

    def __init__(self,**kwargs):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0, norm_layer=nn.BatchNorm2d)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


def get_bisenetv2(dataset='pascal_voc', backbone='resnet18', pretrained=False, root='~/.torch/models',
                pretrained_base=False, aux = True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = BiSeNetv2(datasets[dataset].NUM_CLASS, backbone=backbone, aux = aux, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(0)
        model.load_state_dict(torch.load(get_model_file('bisenet_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device), False)
    return model

def get_bisenetv2_resnet18_pascal_voc(**kwargs):
    return get_bisenetv2('pascal_voc', 'resnet18', aux = True, **kwargs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 224, 224)
    model = BiSeNetv2(19, backbone='resnet18')
    print(model.exclusive)
