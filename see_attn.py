import torch
import torch.nn as nn

# 1. 模擬輸入資料
# 假設有一張圖片：Batch=1, Channels=3 (RGB), Height=224, Width=224
img = torch.randn(1, 3, 224, 224)

# 2. 設定參數
patch_size = 16  # 每個方塊是 16x16 像素
embed_dim = 768  # 每個 Patch 轉換後的向量長度 (就像 Word Embedding 的維度)

# 計算預期序列長度 (Sequence Length)
# (224 / 16) * (224 / 16) = 14 * 14 = 196 個 Patches
# 這意味著這張圖會變成一個有 196 個「單字」的句子

# 3. Patch Embedding 層
# 技巧：使用 Conv2d，kernel_size 和 stride 都設為 patch_size
# 這等同於把圖片切成互不重疊的方塊，並對每個方塊做 Linear Projection
patch_embed_layer = nn.Conv2d(
    in_channels=3, 
    out_channels=embed_dim, 
    kernel_size=patch_size, 
    stride=patch_size
)

# 4. 前向傳播與維度變換
x = patch_embed_layer(img) 
# 此時 x 的形狀是: [1, 768, 14, 14] (Batch, Embed_Dim, Grid_H, Grid_W)

# 5. Flatten 與 Transpose (關鍵步驟)
# 我們需要將 2D grid (14x14) 拉平成 1D sequence (196)
# x.flatten(2): 將第 2 維度之後展平 -> [1, 768, 196]
# .transpose(1, 2): 交換維度以符合 Attention 輸入標準 -> [1, 196, 768]
x = x.flatten(2).transpose(1, 2)

print(f"原始圖片形狀: {img.shape}")
print(f"進入 Attention 前的形狀: {x.shape}")
# 輸出: torch.Size([1, 196, 768]) -> (Batch_Size, Sequence_Length, Embedding_Dim)

# 1. 定義 Position Embedding
# 這是一個 "可學習的參數" (Learnable Parameter)
# 我們隨機初始化它，讓神經網路在訓練過程中自己去學每個位置代表什麼意義
# 形狀必須跟 x 一模一樣：196 個位置，每個位置由 768 維向量表示
pos_embed = nn.Parameter(torch.randn(1, 196, 768))

# 2. 加入位置資訊
# 關鍵操作：直接相加 (Element-wise Addition)，而不是串接 (Concatenation)
# 這保留了原本的維度，讓資訊融合
x = x + pos_embed 

print(f"加入位置編碼後的形狀: {x.shape}") 
# 輸出: torch.Size([1, 196, 768]) -> 維度沒變，但現在 x 裡面隱含了位置資訊