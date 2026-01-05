#!/bin/bash

# 定義變數 (根據你的 Python 程式碼設定)
epoch=100
max_lr=0.01
batch_size=64 # 請根據你實際的 batch_size 修改

# 構建 Message 字串
# 使用 \n 代表換行
Message="No reach final epoch, loss became nan QQ
num_epochs=${epoch}
max_lr=${max_lr}
batch_size=${batch_size}
using gradient_accumulate=4
data aug using jitter and crop
citerion1=FocalLoss(), no param"

# 輸出檢查
echo "${Message}"

# 提交原始預測結果
kaggle competitions submit -c lab-4-semantic-segmentation-on-bcss-639401 -f pred_mask_output.csv -m "${Message}"

# 構建 Message_crf 並提交
Message_crf="${Message}
using crf post-processing"

kaggle competitions submit -c lab-4-semantic-segmentation-on-bcss-639401 -f refined_mask_output.csv -m "${Message_crf}"