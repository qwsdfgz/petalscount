import os
import numpy as np
import h5py
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as T
from model import CSRNet


# CKPT_PATH = r'I:\xsp\CSRNet\CSRNet\plots\SE\0model_best.pth.tar'
# IMG_DIR   = r'I:\xsp\CSRNet\CSRNet\data\part_B_final\test\images'
# GT_DIR    = r'I:\xsp\CSRNet\CSRNet\data\part_B_final\test\ground_truth'
CKPT_PATH = './0model_best.pth.tar'
IMG_DIR = './images'
GT_DIR = './gt'


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

model = CSRNet().cuda()
ckpt = torch.load(CKPT_PATH, weights_only=True)
filtered = {k: v for k, v in ckpt['state_dict'].items()
            if not k.endswith(('.total_ops', '.total_params'))}
model.load_state_dict(filtered, strict=False)
model.eval()


img_files = [f for f in os.listdir(IMG_DIR)
             if f.lower().endswith(('.jpg', '.png', '.JPG'))]


total_mae = 0.0
total_mse = 0.0
processed_count = 0


for img_name in img_files:
    try:
        img_path = os.path.join(IMG_DIR, img_name)
        img_pil = Image.open(img_path).convert('RGB')
        inp = transform(img_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            density_map = model(inp).squeeze().cpu().numpy()
        pred_cnt = float(density_map.sum())

        gt_path = os.path.join(GT_DIR,
                               img_name.replace('.jpg', '.h5')
                                       .replace('.png', '.h5')
                                       .replace('.JPG', '.h5'))
        with h5py.File(gt_path, 'r') as hf:
            gt_map = np.array(hf['density'])
        gt_cnt = float(gt_map.sum())

        mae = abs(pred_cnt - gt_cnt)
        mse = (pred_cnt - gt_cnt) ** 2

        total_mae += mae
        total_mse += mse
        processed_count += 1

    except Exception as e:
        print(f"Error on {img_name}: {e}")
        continue

if processed_count > 0:
    mae_mean = total_mae / processed_count
    mse_mean = total_mse / processed_count
    rmse_mean = np.sqrt(mse_mean)

    print("\n===== Overall Performance =====")
    print(f"Total images: {processed_count}")
    print(f"Average MAE : {mae_mean:.4f}")
    print(f"Average MSE : {mse_mean:.4f}")
    print(f"Average RMSE: {rmse_mean:.4f}")
else:
    print("No images processed.")