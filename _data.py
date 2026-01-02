from _init import *

TRAIN_IMAGE_PATH = './BCSS_ori/train/'
TRAIN_MASK_PATH  = './BCSS_ori/train_mask/'
VAL_IMAGE_PATH   = './BCSS_ori/val/'
VAL_MASK_PATH    = './BCSS_ori/val_mask/'

n_classes = 3

def create_df(IMAGE_PATH):
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    name = sorted(name)
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

train_df = create_df(TRAIN_IMAGE_PATH)
val_df = create_df(VAL_IMAGE_PATH)

# use for debug only, not for real training
# train_df = train_df.sample(n=1280, random_state=42).reset_index(drop=True)
# val_df = val_df.sample(n=640, random_state=42).reset_index(drop=True)

print('Total Train Images: ', len(train_df))
print('Total Val Images: ', len(val_df))

X_train = train_df['id'].to_numpy()
X_val = val_df['id'].to_numpy()

class BCSSDataset(Dataset):

    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        # if self.transform is None:
        #     img = Image.fromarray(img)
        # t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        # img = t(img)
        mask = mask.squeeze(0).long()

        return img, mask

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# For TRAIN
transforms_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.RandomResizedCrop(size=(224, 224), scale=(0.9375, 1), ratio=(1, 1)),
    # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5), # only works on image, mask won't
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ToTensorV2(),
])

# For VAL
transforms_val = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.9375, 1), ratio=(1, 1)),
    A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ToTensorV2(),
])

#datasets
train_set = BCSSDataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, X_train, mean, std, transforms_train)
val_set   = BCSSDataset(VAL_IMAGE_PATH  , VAL_MASK_PATH  , X_val  , mean, std, transforms_val)

#dataloader
batch_size = 64
num_workers = 8 # 可以根據實際情況調整

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers)