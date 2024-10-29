import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
from utils import create_df, make_folder, open_class_csv, create_image_legend
import plotting

if __name__ == '__main__':
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\data\uav_graz\dataset\semantic_drone_dataset'
    OUTPUT_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\output\uav_graz'

    IMAGE_PATH = os.path.join(DATA_BASE, r'original_images')
    MASK_PATH = os.path.join(DATA_BASE, r'label_images_semantic')

    # File location for collected drone image data
    # DATA_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\data\uav_addis_01'
    # OUTPUT_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\output\uav_addis_01'
    #
    # IMAGE_PATH = DATA_BASE
    # MASK_PATH = DATA_BASE
    #

    MODEL_BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\models'

    models_folder = os.listdir(MODEL_BASE)
    print(f'available models: {models_folder}')

    mode = 'whole_model'

    # changes these variables according to the models name and location you have trained
    model_folder = 'resnext50'

    model_name = f'Unet-resnext50_32x4d_211024_mIOU572_{mode}.pt'
    # model_name = f'Unet-Mobilenet_v2_161024_mIoU385_{mode}.pt'
    # model_name = f'Unet-Resnet34_181024_mIoU454_{mode}.pt'

    MODEL_DICT_PATH = os.path.join(MODEL_BASE, model_folder, model_name)

    # create saving directory
    output_folder = os.path.join(OUTPUT_BASE, model_name.split('.')[0])
    make_folder(output_folder)
    make_folder(os.path.join(output_folder, 'predictions'))
    make_folder(os.path.join(output_folder, 'predictions_plots'))

    class_dict = open_class_csv(r'C:\Users\PC\Coding\ethiopia_uav_segmentation\data\uav_graz\class_dict_seg.csv')
    class_colors = {i: (row['r'], row['g'], row['b']) for i, row in class_dict.iterrows()}

    create_image_legend(class_colors, class_dict)

    load_state_dict = False
    load_whole_model = not load_state_dict

    plot_image = True
    save_predictions = True
    save_plots = True

    print('Creating a DataSet')
    n_classes = 23
    image_hw = (704, 1056)
    df = create_df(image_path=IMAGE_PATH)
    print('Total Images: ', len(df))

    # create test and train datasets: training 76.5?%), testing (13.5%), validation (10%)
    test_size = 0.1
    predict_all = True

    X_trainval, X_test = train_test_split(df['id'].values, test_size=test_size, random_state=19)

    # load model
    print('Loading model')
    if 'Mobilenet' in model_name:
        model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23, activation=None,
                         encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
    elif 'Resnet34' in model_name:
        model = smp.Unet('resnet34', encoder_weights='imagenet', classes=23, activation=None,
                         encoder_depth=5)
    elif 'Resnext50' in model_name:
        model = smp.Unet('resnext50_32x4d', encoder_weights='imagenet', classes=23, activation=None,
                         encoder_depth=5)

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
    test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test, mask_post='.jpg')
    #prediction_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_trainval, transform=t_test)
    pred_image_nr = None #[236] # None
    if pred_image_nr is not None:
        test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_trainval, transform=t_test)

    all_gt, all_pred = [], []

    colored_image_np = np.zeros((768, 1152, 3), dtype=np.uint8)

    print('Doing inference...')
    for i, (image, mask) in tqdm(enumerate(test_set)):
        if pred_image_nr is not None:
            if i not in pred_image_nr:
                continue
        pred_mask, score = predict_image_mask_miou(model, image, mask, device)
        img_index = test_set.X[i]

        # color prediction mask
        pred_mask_np = pred_mask.numpy()

        for class_value, rgb in class_colors.items():
            colored_image_np[pred_mask_np == class_value] = rgb

        colored_image = Image.fromarray(colored_image_np)

        # calculate confusion values
        all_gt.append(mask.view(-1).cpu().numpy())
        all_pred.append(pred_mask.view(-1).cpu().numpy())

        color_ = True

        if save_predictions:
            if color_:
                colored_image.save(os.path.join(output_folder, 'predictions', f'{img_index}.png'))
            else:
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

    gt_cm = np.concatenate(all_gt)
    pred_cm = np.concatenate(all_pred)
    cm = confusion_matrix(gt_cm, pred_cm)
    cm_percentage = cm.astype('float') / cm.sum() * 100

    annot = np.empty_like(cm_percentage, dtype=object)
    for i in range(cm_percentage.shape[0]):
        for j in range(cm_percentage.shape[1]):
            if cm_percentage[i, j] == 0:
                annot[i, j] = '0'  # Display '0' for zero values
            else:
                annot[i, j] = f'{cm_percentage[i, j]:.1f}'  # Display two decimals for non-zero values

    # classes: tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle,
    #          roof, wall, fence, fence-pole, window, door, obstacle
    figcm, axcm = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(cm_percentage, annot=annot, fmt='', cbar=True, cmap='Blues', xticklabels=class_dict['name'],
                yticklabels=class_dict['name'])
    plt.savefig('cm.png')
