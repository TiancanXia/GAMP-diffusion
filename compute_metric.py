from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch

# device = 'cuda:0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

task = 'SR'
factor = 4
sigma = 0.1
scale = 1.0


label_root = Path(f'./results/CS/label')
delta_recon_root = Path(f'./results/CS/recon') # GAMP-DMPS_1323_12288_1000_1 2 2.5 all
# normal_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps+/{scale}/recon')

psnr_delta_list = []
psnr_normal_list = []

lpips_delta_list = []
lpips_normal_list = []

ssim_delta_list =[]
ssim_normal_list = []  # 

with torch.no_grad():  # 
    for idx in tqdm(range(100)):
        fname = str(idx).zfill(5)

        # float32 [0, 1]
        label_np = plt.imread(label_root / f'{fname}.png')[:, :, :3].astype(np.float32)
        recon_np = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3].astype(np.float32)

        # SSIM
        ssim_delta = structural_similarity(
            label_np, recon_np,
            data_range=1.0,
            channel_axis=2
        )
        ssim_delta_list.append(ssim_delta)

        # PSNR
        psnr_delta = peak_signal_noise_ratio(label_np, recon_np, data_range=1.0)
        psnr_delta_list.append(psnr_delta)

        # LPIPS
        t_recon = torch.from_numpy(recon_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2. - 1.
        t_label = torch.from_numpy(label_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2. - 1.

        # LPIPS
        dist = loss_fn_vgg(t_recon, t_label)
        lpips_delta_list.append(dist.item())

        print(f'DPS PSNR: {psnr_delta:.4f} | SSIM: {ssim_delta:.4f} | LPIPS: {dist.item():.4f}')

# 
psnr_avg = np.mean(psnr_delta_list)
ssim_avg = np.mean(ssim_delta_list)
lpips_avg = np.mean(lpips_delta_list)

print(f'DPS PSNR: {psnr_avg:.4f} | SSIM: {ssim_avg:.4f} | LPIPS: {lpips_avg:.4f}')

# for idx in tqdm(range(3)):
#     fname = str(idx).zfill(5)
#
#     label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
#     delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
#     # normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]
#
#     # ========== SSIM([0,1]) ==========
#     ssim_delta = structural_similarity(
#         label, delta_recon,
#         data_range=1.0,  # [0,1]
#         channel_axis=2,  # 
#         multichannel=True  # skimage
#     )
#     ssim_delta_list.append(ssim_delta)
#
#     psnr_delta = peak_signal_noise_ratio(label, delta_recon)
#     # psnr_normal = peak_signal_noise_ratio(label, normal_recon)
#
#     psnr_delta_list.append(psnr_delta)
#     # psnr_normal_list.append(psnr_normal)
#
#     delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
#     # normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
#     label = torch.from_numpy(label).permute(2, 0, 1).to(device)
#
#     delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
#     # normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
#     label = label.view(1, 3, 256, 256) * 2. - 1.
#
#     delta_d = loss_fn_vgg(delta_recon, label)
#     # normal_d = loss_fn_vgg(normal_recon, label)
#
#     lpips_delta_list.append(delta_d)
#     # lpips_normal_list.append(normal_d)
#
# psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
# lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)
# ssim_delta_avg = sum(ssim_delta_list) / len(ssim_delta_list)  # Added
#
# # psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
# # lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)
#
# # print(f'Delta PSNR: {psnr_delta_avg}')
# # print(f'Delta LPIPS: {lpips_delta_avg}')
#
# print(f'DPS PSNR: {psnr_delta_avg:.4f} | SSIM: {ssim_delta_avg:.4f} | LPIPS: {lpips_delta_avg:.4f}')
#
# # print(f'Normal PSNR: {psnr_normal_avg}')
# # print(f'Normal LPIPS: {lpips_normal_avg}')