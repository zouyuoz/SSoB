from _init import *
from _data import *
import torch.optim as optim
from _UNet_CRF import UNetWithCRF
from tqdm import tqdm  # 若無 tqdm 可移除，但建議保留以觀察進度

# --- 設定參數 ---
NUM_EPOCHS = 20       # 訓練總回數，CRF 收斂通常較快，可視情況調整
LR = 1e-3             # 學習率
WEIGHT_DECAY = 1e-4   # 權重衰減
SAVE_NAME = "BEST_CRF.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. 初始化模型與載入權重 ---
pretrained_path = "BEST.pt"
print(f"Loading UNet weights from {pretrained_path}...")
# 確保這裡的 n_classes 與你資料集一致
model = UNetWithCRF(n_channels=3, n_classes=3, backbone_path=pretrained_path).to(device)

# --- 2. 凍結 UNet 參數 (Freeze Encoder/Decoder) ---
for param in model.unet.parameters():
    param.requires_grad = False

print("\n[Check] Trainable parameters in CRF:")
for name, param in model.crf.named_parameters():
    print(f"  - {name}: requires_grad={param.requires_grad}")
print("-" * 50)

# --- 3. 定義優化器與損失函數 ---
# 只優化 requires_grad=True 的參數 (即 CRF 的參數)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LR, 
    weight_decay=WEIGHT_DECAY
)

# 因為 crfseg 的 forward 回傳 log_softmax，所以使用 NLLLoss
criterion = nn.CrossEntropyLoss() 

# 紀錄最佳驗證損失
best_val_loss = float('inf')

print(f"Start Training for {NUM_EPOCHS} epochs...")

# --- 4. 訓練迴圈 ---
for epoch in range(NUM_EPOCHS):
    # ================= Training Phase =================
    # 策略：讓 UNet 保持 eval (固定 BN 統計量)，只讓 CRF 進入 train 模式
    model.eval() 
    model.crf.train()
    
    train_loss = 0.0
    
    # 建立進度條
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", unit="batch")
    
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward: UNet (Frozen) -> CRF (Trainable)
        outputs = model(inputs)
        
        # 計算 Loss
        # outputs: [B, n_classes, H, W] (Log Probability)
        # labels:  [B, H, W] (Integer Class Index)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 累加 Loss (乘以 batch size 以便後續算平均)
        train_loss += loss.item() * inputs.size(0)
        
        # 更新進度條上的當前 Loss
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    # 計算該 Epoch 平均 Training Loss
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # ================= Validation Phase =================
    model.eval() # 驗證階段全開 eval (CRF 也不更新)
    val_loss = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val  ]", unit="batch")
        
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    avg_val_loss = val_loss / len(val_loader.dataset)
    
    # ================= Summary & Save =================
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    # 如果驗證損失創新低，則儲存模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), SAVE_NAME)
        print(f"  >>> New Best Model Saved to {SAVE_NAME} (Val Loss: {best_val_loss:.6f})")
    
    print("-" * 50)

print("Training Completed.")