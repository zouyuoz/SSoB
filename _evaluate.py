from _init import *
from _data import *
from _UNet import *
from _AttnUNet import *

TEST_IMAGE_PATH = './BCSS_ori/test/'
test_df = create_df(TEST_IMAGE_PATH)
print('Total Test Images: ', len(test_df))
X_test = test_df['id'].to_numpy()

class BCSSTestDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        return img, self.X[idx]

test_set = BCSSTestDataset(TEST_IMAGE_PATH, None, X_test, mean, std)

model = AttentionUNet(in_channels=3, out_channels=3)
model.load_state_dict(torch.load("BEST.pt"))
model = model.to(device)

def predict_image(model, image):
    model.eval()
    model.to(device); image=image.to(device)
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        masked = torch.argmax(pred, dim=1)
    return masked.cpu().squeeze(0)

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
# params: sg=10, cg=25, sb=40, cr=10, cb=5  → avg mIoU = 0.5971
def crf(image, label_mask, gt_prob=0.9, n_iter=3, n_labels=3,
        sxy_g=(10,10), compat_g=25,
        sxy_b=(40,40), srgb_b=(10,10,10), compat_b=5):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    label_mask = np.ascontiguousarray(label_mask, dtype=np.int32)
    H, W = label_mask.shape
    d = dcrf.DenseCRF2D(W, H, n_labels)
    U = unary_from_labels(label_mask, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=sxy_g, compat=compat_g)
    d.addPairwiseBilateral(sxy=sxy_b, srgb=srgb_b, rgbim=image, compat=compat_b)
    Q = d.inference(n_iter)
    refined = np.argmax(Q, axis=0).reshape(H, W)
    return refined

data = []

for i in tqdm(range(len(test_set)), ncols=50):
    img, filename = test_set[i]
    with torch.no_grad():
        pred_mask = predict_image(model, img.to(device))
    pred_mask_np = pred_mask.numpy()

    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # 套用 CRF
    refined_mask = crf(img_np, pred_mask_np.astype(np.int32))

    # 儲存結果
    data.append({
        'index': filename,
        'pred_mask': pred_mask_np.tolist(),
        'refined_mask': refined_mask.tolist()
    })

df = pd.DataFrame(data)

df.to_csv('output.csv', index=False)

# 拆出 pred_mask
pred_df = df[['index', 'pred_mask']]
pred_df.to_csv('pred_mask_output.csv', index=False)

# 拆出 refined_mask
refined_df = df[['index', 'refined_mask']]
refined_df.rename(columns={'refined_mask': 'pred_mask'}, inplace=True)
refined_df.to_csv('refined_mask_output.csv', index=False)