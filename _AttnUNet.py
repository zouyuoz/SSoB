import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    標準的 U-Net 卷積區塊：(Conv3x3 -> BN -> ReLU) x 2
    """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # 確保 g1 與 x1 尺寸一致 (若有尺寸差異可在此處插值，但標準架構通常輸入前已對齊)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class UpBlock(nn.Module):
    """
    Decoder 區塊：Upsample -> Attention -> Concat -> Double Conv
    """
    def __init__(self, in_ch, out_ch, skip_ch):
        super(UpBlock, self).__init__()

        # 1. Upsampling (論文使用 Transpose Conv)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # 2. Attention Gate
        # F_g: 上採樣後的特徵圖通道數 (out_ch)
        # F_l: Skip Connection 的通道數 (skip_ch)
        self.att = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2)

        # 3. Double Convolution (融合特徵)
        # 輸入通道 = 上採樣特徵 (out_ch) + Attention處理過的 Skip Connection (skip_ch)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # 處理 padding 問題 (若輸入尺寸不是 2 的倍數，上採樣後可能會有 1 pixel 的誤差)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffX > 0 or diffY > 0:
            x = F.pad(
                x,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )

        # 計算 Attention (g=decoder feature, x=encoder skip connection)
        x_att = self.att(g=x, x=skip)

        # Concatenate
        x = torch.cat([x_att, x], dim=1)

        # Convolution
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()

        # Filters (依據論文通常是 64, 128, 256, 512, 1024)
        filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.conv1 = ConvBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Center (Bridge)
        self.conv5 = ConvBlock(filters[3], filters[4])

        # Decoder (包含 Attention)
        self.up5 = UpBlock(filters[4], filters[3], filters[3])
        self.up4 = UpBlock(filters[3], filters[2], filters[2])
        self.up3 = UpBlock(filters[2], filters[1], filters[1])
        self.up2 = UpBlock(filters[1], filters[0], filters[0])

        # Final Output
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool1(e1))
        e3 = self.conv3(self.pool2(e2))
        e4 = self.conv4(self.pool3(e3))

        # Center
        e5 = self.conv5(self.pool4(e4))

        # Decoder
        d5 = self.up5(e5, e4)
        d4 = self.up4(d5, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)

        return torch.sigmoid(self.final_conv(d2))