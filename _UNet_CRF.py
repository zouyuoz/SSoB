import torch
import torch.nn as nn
from _UNet import UNet
from _AttnUNet import AttentionUNet
from crfseg import CRF

class UNetWithCRF(nn.Module):
    """
    整合 UNet 與可微分 CRF (Conditional Random Field) 的模型。

    這使用了 crfseg 套件，它是基於高斯濾波 (Gaussian Permutohedral Lattice) 的
    高效 PyTorch 實作，支援 GPU 加速與反向傳播。
    """
    def __init__(self, n_channels, n_classes, backbone_path=None):
        super(UNetWithCRF, self).__init__()

        # 1. 載入原本的 UNet 作為特徵提取器 (Encoder-Decoder)
        # self.unet = UNet(n_channels, n_classes)
        self.unet = AttentionUNet(n_channels, n_classes)

        # 若有預訓練權重，可在這裡載入
        if backbone_path:
            self.unet.load_state_dict(torch.load(backbone_path))
            print(f"Loaded UNet weights from {backbone_path}")

        # 2. 定義 CRF 層
        # filter_size, n_iter 等參數可根據顯存大小調整，預設值通常效果就不錯
        self.crf = CRF(
            n_spatial_dims=2,  # n_spatial_dims=2 代表處理 2D 影像
            filter_size=7,     # 類似 pydensecrf 的 sxy
            n_iter=5,          # 迭代次數，越多越準但越慢 (訓練時建議 3-5)
            requires_grad=True # 開啟梯度，讓 CRF 參數(權重)可被訓練
        )

    def forward(self, x):
        """
        :param x: 原始輸入影像 [Batch, Channels, Height, Width]
        """
        # 1. 取得 UNet 的輸出 (Unnormalized Logits)
        # 形狀: [B, n_classes, H, W]
        unary_logits = self.unet(x)

        # 2. 通過 CRF 層進行精修
        # crfseg 的 forward 通常接受 (unary_logits, image)
        # 注意: crfseg 內部會處理 Softmax，所以這裡傳入 Logits 即可
        # 原始影像 x 用於提供邊緣資訊 (Bilateral Potential)
        refined_log_proba = self.crf(unary_logits)

        # 回傳的是 Log Probability (經過 LogSoftmax)
        # 若需要機率分布可再取 torch.exp()
        return refined_log_proba