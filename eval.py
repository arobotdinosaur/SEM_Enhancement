import argparse
from cleanfid import fid
from glob import glob
from tqdm import tqdm
import os

from torcheval.metrics.functional import peak_signal_noise_ratio
from torch.utils.data import DataLoader

from core.dataset import TestImagePairDataset
from core.utils import inception_score, tensor2img

from Palette.core.base_dataset import BaseDataset
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from PIL import Image 
import matplotlib.pyplot as mpl

def metrics(true_image_folder, pred_image_folder, root, mask = None):
    """
    Calculates the mean squared error of the true and predicted images 
    """
    dataset = TestImagePairDataset(size=512, folder_A=true_image_folder, folder_B=pred_image_folder, mask=mask)
    dataloader = DataLoader(dataset, batch_size= 1)
    SSIMs = []
    PSNRs = []
    MSEs = []

    for true, pred in tqdm(dataloader):
        np_true = tensor2img(true[0])
        np_pred = tensor2img(pred[0])
        concat = np.concatenate((np_true,np_pred))
        range_im = concat.max() - concat.min()
        if range_im != 0:
            PSNRs.append(peak_signal_noise_ratio(target = true, input = pred))
            MSEs.append(mse(image0 = np_true, image1 = np_pred))
            SSIMs.append(ssim(im1 = np_true, im2 = np_pred, channel_axis = 2, gaussian_weights = True, 
            sigma = 1.5, use_sample_covariance = False, data_range= range_im))
    return (SSIMs, PSNRs, MSEs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tru', type=str, help='Ground truth images directory')
    parser.add_argument('-g', '--gen', type=str, help='Generate images directory')
    parser.add_argument('-d', '--dst', type=str, help="Destination of results", required = False)
    parser.add_argument('-m', '--mask', type=str, help="Directory of image masks", required = False)
    parser.add_argument('-p', '--pixel', type=bool, help="If you only want pixel-wise metrics", required = False)
   
    ''' parser configs '''
    args = parser.parse_args()

    if not args.pixel:
        fid_score = fid.compute_fid(args.tru, args.gen)
        print(fid_score)
        is_mean, is_std = inception_score(BaseDataset(args.gen), cuda=True, batch_size=4, resize=True, splits=10)
    else:
        fid_score = np.array([0])
        is_mean = 0
        is_std = 0
    ssim, PSNR, MSE = metrics(args.tru, args.gen, '', mask = args.mask)
    
    if args.dst:
        # Writing to file
        os.makedirs(args.dst, exist_ok=True)
        np.save(os.path.join(args.dst,"ssim.npy"), ssim)
        np.save(os.path.join(args.dst,"psnr.npy"), PSNR)
        
        with open(os.path.join(args.dst,"output.txt"), "w+") as f:
            # Writing data to a file
            f.write(f'length of images: {len(ssim)}, FID: {fid_score}, IS:{is_mean} {is_std}, SSIM:{np.mean(ssim)} {np.std(ssim)}, PSNR: {np.mean(PSNR)} {np.std(PSNR)}, MSE: {np.mean(MSE)} {np.std(MSE)}')
    else:
        print(f'length of images: {len(ssim)}')
        print(f'FID: {fid_score}')
        print(f'IS:{is_mean} {is_std}')
        print(f'SSIM:{np.mean(ssim)} {np.std(ssim)}')
        print(f'PSNR: {np.mean(PSNR)} {np.std(PSNR)}')
        print(f'MSE: {np.mean(MSE)} {np.std(MSE)}')