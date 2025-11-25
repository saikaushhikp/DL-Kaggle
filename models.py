import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Basic building blocks
# -----------------------------

class ConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, dilation=1, act='leaky', bias=False):
        super().__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        
        if act == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualConv(nn.Module):
    """Two convs with residual (projection when channels differ)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBnAct(in_ch, out_ch)
        self.conv2 = ConvBnAct(out_ch, out_ch)
        self.need_proj = (in_ch != out_ch)
        if self.need_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.need_proj:
            identity = self.bn_proj(self.proj(identity))
        return out + identity


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))


# -----------------------------
# MiniInception with residual
# -----------------------------
class MiniInceptionRes(nn.Module):
    """
    Mini-inception with three stages, residual connection.
    Each split uses one 3x3 and one dilated 3x3 conv.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 2 == 0, "out_channels must be divisible by 2"
        half = out_ch // 2
        
        # stage 1
        self.c1l = ConvBnAct(in_ch, half, padding=1, dilation=1, act='leaky')
        self.c1r = ConvBnAct(in_ch, half, padding=2, dilation=2, act='leaky')
        
        # stage 2
        self.c2l = ConvBnAct(out_ch, half, padding=1, dilation=1, act='leaky')
        self.c2r = ConvBnAct(out_ch, half, padding=2, dilation=2, act='leaky')
        
        # stage 3
        self.c3l = ConvBnAct(out_ch, half, padding=1, dilation=1, act='leaky')
        self.c3r = ConvBnAct(out_ch, half, padding=2, dilation=2, act='leaky')

        self.need_proj = (in_ch != out_ch)
        if self.need_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = torch.cat((self.c1l(x), self.c1r(x)), dim=1)  # out_ch
        y = torch.cat((self.c2l(y), self.c2r(y)), dim=1)
        y = torch.cat((self.c3l(y), self.c3r(y)), dim=1)
        ident = x
        if self.need_proj:
            ident = self.bn_proj(self.proj(ident))
        return self.act(y + ident)


# -----------------------------
# Corrected MFNet (channel-safe)
# -----------------------------
class MFNet(nn.Module):
    """
    Fully corrected MFNet:
      - Clean channel bookkeeping to avoid concat mismatches
      - Residual mini-inceptions
      - Optional SE after fusion
      - Deep supervision heads
    """
    def __init__(self, in_ch=16, n_class=1, use_se=True, deep_supervision=True):
        super().__init__()
        self.n_class = n_class
        self.use_se = use_se
        self.deep_supervision = deep_supervision

        # choose widths (bigger than original MFNet)
        # level indices: 1..5 (1 shallow -> 5 deepest)
        rgb_ch = [32, 96, 160, 256, 320]    # channels at levels 1..5 for RGB
        inf_ch = [32, 64, 96, 128, 160]     # channels at levels 1..5 for INF

        # RGB branch
        self.conv1_rgb   = ResidualConv(3, rgb_ch[0])
        self.conv2_1_rgb = ResidualConv(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ResidualConv(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ResidualConv(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ResidualConv(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb   = MiniInceptionRes(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb   = MiniInceptionRes(rgb_ch[3], rgb_ch[4])

        # INF branch (if present)
        self.inf_in_ch = max(0, in_ch - 3)
        if self.inf_in_ch > 0:
            self.conv1_inf   = ResidualConv(self.inf_in_ch, inf_ch[0])
            self.conv2_1_inf = ResidualConv(inf_ch[0], inf_ch[1])
            self.conv2_2_inf = ResidualConv(inf_ch[1], inf_ch[1])
            self.conv3_1_inf = ResidualConv(inf_ch[1], inf_ch[2])
            self.conv3_2_inf = ResidualConv(inf_ch[2], inf_ch[2])
            self.conv4_inf   = MiniInceptionRes(inf_ch[2], inf_ch[3])
            self.conv5_inf   = MiniInceptionRes(inf_ch[3], inf_ch[4])
        else:
            # placeholders
            self.conv1_inf = None

        # Precompute channel sizes so we never miscalculate:
        self.rgb_ch = rgb_ch
        self.inf_ch = inf_ch

        # deepest fused channels
        deepest_rgb = rgb_ch[4]   # 320
        deepest_inf = inf_ch[4] if self.inf_in_ch > 0 else 0  # 160 or 0
        self.fused_deep_ch = deepest_rgb + deepest_inf  # e.g. 480

        # skip channel counts
        skip4_ch = rgb_ch[3] + (inf_ch[3] if self.inf_in_ch > 0 else 0)  # 256 + 128 = 384
        skip3_ch = rgb_ch[2] + (inf_ch[2] if self.inf_in_ch > 0 else 0)  # 160 + 96  = 256
        skip2_ch = rgb_ch[1] + (inf_ch[1] if self.inf_in_ch > 0 else 0)  # 96  + 64  = 160
        skip1_ch = rgb_ch[0] + (inf_ch[0] if self.inf_in_ch > 0 else 0)  # 32  + 32  = 64

        # Decoding projections (concat upsampled fused/prev + skip) -> next-level channels
        # decode4: (fused_deep) + skip4 -> project to skip3_ch
        dec4_in_ch = self.fused_deep_ch + skip4_ch     # e.g. 480 + 384 = 864
        dec4_out_ch = skip3_ch                         # 256

        dec3_in_ch = dec4_out_ch + skip3_ch            # 256 + 256 = 512
        dec3_out_ch = skip2_ch                         # 160

        dec2_in_ch = dec3_out_ch + skip2_ch            # 160 +160 = 320
        dec2_out_ch = skip1_ch                         # 64

        dec1_in_ch = dec2_out_ch                       # 64
        dec1_out_ch = dec1_in_ch                       # keep same, head maps to logits

        # Residual projection convs
        self.decode4_proj = ResidualConv(dec4_in_ch, dec4_out_ch)
        self.decode3_proj = ResidualConv(dec3_in_ch, dec3_out_ch)
        self.decode2_proj = ResidualConv(dec2_in_ch, dec2_out_ch)
        self.decode1_proj = ResidualConv(dec1_in_ch, dec1_out_ch)

        # Heads
        self.head = nn.Conv2d(dec1_out_ch, self.n_class, kernel_size=1)
        if self.deep_supervision:
            self.head_ds3 = nn.Conv2d(dec3_out_ch, self.n_class, kernel_size=1)
            self.head_ds4 = nn.Conv2d(dec4_out_ch, self.n_class, kernel_size=1)

        # Optional SE after fusion
        if self.use_se:
            self.se = SEBlock(self.fused_deep_ch, reduction=8)

        self._init_weights()

    def forward(self, x):
        # x: B, C, H, W
        assert x.shape[1] >= 3, "input must contain at least 3 channels for RGB"
        x_rgb = x[:, :3, :, :]
        x_inf = x[:, 3:, :, :] if self.inf_in_ch > 0 else None

        #  RGB encode (store skips) 
        x_rgb = self.conv1_rgb(x_rgb)               # level1
        x_rgb = F.max_pool2d(x_rgb, 2)
        x_rgb = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)          # skip level2
        x_rgb = F.max_pool2d(x_rgb_p2, 2)
        x_rgb = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)          # skip level3
        x_rgb = F.max_pool2d(x_rgb_p3, 2)
        x_rgb_p4 = self.conv4_rgb(x_rgb)            # skip level4
        x_rgb = F.max_pool2d(x_rgb_p4, 2)
        x_rgb = self.conv5_rgb(x_rgb)               # deepest rgb

        #  INF encode 
        if x_inf is not None:
            x_inf = self.conv1_inf(x_inf)
            x_inf = F.max_pool2d(x_inf, 2)
            x_inf = self.conv2_1_inf(x_inf)
            x_inf_p2 = self.conv2_2_inf(x_inf)
            x_inf = F.max_pool2d(x_inf_p2, 2)
            x_inf = self.conv3_1_inf(x_inf)
            x_inf_p3 = self.conv3_2_inf(x_inf)
            x_inf = F.max_pool2d(x_inf_p3, 2)
            x_inf_p4 = self.conv4_inf(x_inf)
            x_inf = F.max_pool2d(x_inf_p4, 2)
            x_inf = self.conv5_inf(x_inf)
        else:
            
            # create zero placeholders with correct channel counts so concat works seamlessly
            B, _, Hd, Wd = x_rgb.shape
            device = x_rgb.device
            x_inf = torch.zeros(B, 0, Hd, Wd, device=device)  # deepest INF channels = 0
            x_inf_p4 = None
            x_inf_p3 = None
            x_inf_p2 = None

        #  fusion at deepest level 
        if x_inf.shape[1] == 0:
            fused = x_rgb
        else:
            fused = torch.cat([x_rgb, x_inf], dim=1)

        if self.use_se:
            fused = self.se(fused)

        #  decode level 4 
        x = F.interpolate(fused, scale_factor=2.0, mode='nearest')  # up -> level4 spatial
        # build skip4 (rgb_p4 + inf_p4 if available)
        if x_inf is not None and x_inf_p4 is not None:
            skip4 = torch.cat([x_rgb_p4, x_inf_p4], dim=1)
        else:
            skip4 = x_rgb_p4

        # concat upsampled fused + skip4
        x = torch.cat([x, skip4], dim=1)    # channels = fused_deep_ch + skip4_ch
        x = self.decode4_proj(x)            # out channels = dec4_out_ch
        ds4 = self.head_ds4(x) if self.deep_supervision else None

        #  decode level 3 
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if x_inf is not None and x_inf_p3 is not None:
            skip3 = torch.cat([x_rgb_p3, x_inf_p3], dim=1)
        else:
            skip3 = x_rgb_p3
        x = torch.cat([x, skip3], dim=1)    # channels = dec4_out_ch + skip3_ch
        x = self.decode3_proj(x)            # out channels = dec3_out_ch
        ds3 = self.head_ds3(x) if self.deep_supervision else None

        #  decode level 2 
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if x_inf is not None and x_inf_p2 is not None:
            skip2 = torch.cat([x_rgb_p2, x_inf_p2], dim=1)
        else:
            skip2 = x_rgb_p2
        x = torch.cat([x, skip2], dim=1)
        x = self.decode2_proj(x)            # out channels = dec2_out_ch

        # final upsample to original resolution 
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.decode1_proj(x)
        main_logits = self.head(x)

        if self.deep_supervision:
            # ensure ds shapes match main_logits spatial size
            ds3_up = F.interpolate(ds3, size=main_logits.shape[2:], mode='bilinear', align_corners=False)
            ds4_up = F.interpolate(ds4, size=main_logits.shape[2:], mode='bilinear', align_corners=False)
            # return (main, ds3_up, ds4_up)
            return main_logits, ds3_up, ds4_up
        else:
            return main_logits

    def _init_weights(self):
        
        # Kaiming init for convs, BN ones and zeros
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d,)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)