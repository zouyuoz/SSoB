from _init import *
from _data import *
import torch.optim as optim
from _UNet_CRF import UNetWithCRF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_path = "BEST.pt"
print(f"Loading UNet weights from {pretrained_path}...")
model = UNetWithCRF(n_channels=3, n_classes=3, backbone_path=pretrained_path).to(device)

for param in model.unet.parameters():
    param.requires_grad = False

print("CRF trainable parameters:")
for name, param in model.crf.named_parameters():
    print(f"  - {name}: requires_grad={param.requires_grad}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-3, weight_decay=1e-4
)
criterion = nn.NLLLoss() # 因為 crfseg 輸出 log_softmax

# --- 開始訓練迴圈 ---
model.train() # 注意：即使凍結，這裡還是要開 train() 模式，除非你想關掉 BN/Dropout

# 示範一個 batch
# inputs: [B, 3, H, W], labels: [B, H, W]
inputs, labels = next(iter(train_loader))
inputs, labels = inputs.to(device), labels.to(device)

optimizer.zero_grad()
outputs = model(inputs) # Forward 會經過 frozen UNet -> trainable CRF
loss = criterion(outputs, labels)
loss.backward() # 梯度只會計算到 CRF 層，UNet 層梯度為 None 或 0
optimizer.step()