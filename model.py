import torch
import torch.nn as nn
import torch.nn.functional as F
from model import vgg19 as vgg19pre

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g)
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

class VGG(nn.Module):
    def __init__(self, features, num_classes=7):
        super(VGG, self).__init__()

        self.premod = vgg19pre(num_classes=2).cuda()
        checkpoint = torch.load('./facemodel.pt')
        self.premod.load_state_dict(checkpoint)
        self.premod.eval()

        self.feat1, self.feat2, self.feat3, self.feat4 = features[:20], features[20:33], features[33:46], features[46:]
        self.proj = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, bias=False)
        self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=True)
        self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=True)
        self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*3, out_features=512),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def get_feat(self, org, layers=('38')):
        x = org.clone()
        res = []
        for name, module in self.premod.features._modules.items():
            x = module(x)
            if name in layers :
                x = F.interpolate(x, size=44, mode='bilinear', align_corners=True)
                res.append(x)
        return res

    def forward(self, x):
        el1, el2, el3 = self.get_feat(x, ('19', '32', '45'))
        el1, el2, el3 = el1.mean(1).unsqueeze(1), el2.mean(1).unsqueeze(1), el3.mean(1).unsqueeze(1)
        x = torch.cat(((1+el1)*x, (1+el2)*x, (1+el3)*x), 1)
        x = self.feat1(x)
        l1 = x.clone()
        x = self.feat2(x)
        l2 = x.clone()
        x = self.feat3(x)
        l3 = x.clone()
        g = self.feat4(x)

        c1, g1 = self.attn1(self.proj(l1), g)
        c2, g2 = self.attn2(l2, g)
        c3, g3 = self.attn3(l3, g)
        g = torch.cat((g1, g2, g3), 1)
        x = self.classifier(g)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg19(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
