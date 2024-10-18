import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
from tqdm import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp

from training import fit, predict_image_mask_miou, miou_score, pixel_acc_score
from drone_datasets import DroneDataset, DroneTestDataset
from utils import create_df, make_folder
import plotting

if __name__ == '__main__':
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\data\uav_graz\dataset\semantic_drone_dataset'
    OUTPUT_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\output\uav_graz'
    MODEL_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\models'

    IMAGE_PATH = os.path.join(DATA_BASE, r'original_images')
    MASK_PATH = os.path.join(DATA_BASE, r'label_images_semantic')

    model_folder = '181024'
    mode = 'whole_model' #'state_dict'  # 'whole_model'
    model_name = f'Unet-Resnet34_181024_mIoU454_{mode}.pt'
    MODEL_DICT_PATH = os.path.join(MODEL_BASE, model_folder, model_name)

    # create saving directory
    output_folder = os.path.join(OUTPUT_BASE, model_name.split('.')[0])
    make_folder(output_folder)
    make_folder(os.path.join(output_folder, 'predictions'))
    make_folder(os.path.join(output_folder, 'predictions_plots'))

    load_state_dict = False
    load_whole_model = not load_state_dict

    plot_image = True
    save_predictions = True
    save_plots = True

    print('Creating a DataSet')
    n_classes = 23
    df = create_df(image_path=IMAGE_PATH)
    print('Total Images: ', len(df))

    # create test and train datasets: training 76.5?%), testing (13.5%), validation (10%)
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)

    # load model
    print('Loading model')
    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23, activation=None, encoder_depth=5,
                     decoder_channels=[256, 128, 64, 32, 16])

    print('Loading weights')
    # load model state_dict
    if load_state_dict:
        model.load_state_dict(torch.load(MODEL_DICT_PATH, weights_only=True))
        model.eval()

    if load_whole_model:
        model = torch.load(MODEL_DICT_PATH, weights_only=False, map_location=device)
        model.eval()

    # create test dataset
    t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)
    transform = T.ToPILImage()

    print('Doing inference...')
    for i, (image, mask) in tqdm(enumerate(test_set)):
        pred_mask, score = predict_image_mask_miou(model, image, mask, device)
        img_index = test_set.X[i]


        if save_predictions:
            pred_mask_img = np.array(pred_mask, dtype=np.uint8)
            xx = Image.fromarray(pred_mask_img)
            xx.save(os.path.join(output_folder, 'predictions', f'{img_index}.png'))

        if plot_image:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
            ax1.imshow(image)
            ax1.set_title('Picture')

            ax2.imshow(mask)
            ax2.set_title('Ground truth')
            ax2.set_axis_off()

            ax3.imshow(pred_mask)
            ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score))
            ax3.set_axis_off()
            if save_plots:
                fig.savefig(os.path.join(output_folder, 'predictions_plots', f'{img_index}.png'))
            else:
                plt.show()

            plt.close(fig=fig)

