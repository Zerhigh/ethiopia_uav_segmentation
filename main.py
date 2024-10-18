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
    # https://www.kaggle.com/code/ligtfeather/semantic-segmentation-is-easy-with-pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE = r'C:\Users\PC\Coding\ethiopia_uav_segmentation\data\uav_graz\dataset\semantic_drone_dataset'
    IMAGE_PATH = os.path.join(BASE, r'original_images')
    MASK_PATH = os.path.join(BASE, r'label_images_semantic')
    show_example_image = False

    # create datasets from the provided imagery
    n_classes = 23
    df = create_df(image_path=IMAGE_PATH)
    print('Total Images: ', len(df))

    # create test and train datasets: training 76.5?%), testing (13.5%), validation (10%)
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))
    print('Test Size    : ', len(X_test))

    if show_example_image:
        img = Image.open(IMAGE_PATH + '/' + df['id'][100] + '.jpg')
        mask = Image.open(MASK_PATH + '/' + df['id'][100] + '.png')
        print('Image Size', np.asarray(img).shape)
        print('Mask Size', np.asarray(mask).shape)

        plt.imshow(img)
        plt.imshow(mask, alpha=0.6)
        plt.title('Picture with Mask Appplied')
        plt.show()

    # what does this do?
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # create a  datset for re-configuration of Droone Dataset?
    t_train = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
                         A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                         A.GaussNoise()])

    t_val = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(),
                       A.GridDistortion(p=0.2)])

    # apply drone dataset configurations
    train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False)
    val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=False)

    # dataloader
    batch_size = 4

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # define model and hyperparameters
    model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23, activation=None, encoder_depth=5,
                     decoder_channels=[256, 128, 64, 32, 16])

    max_lr = 1e-3
    epoch = 50  #1  #6 #15
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_loader))

    model_save_folder = 'models/171024'
    make_folder(model_save_folder)

    history = fit(epoch, model, train_loader, val_loader, criterion, optimizer, sched, device,
                  model_name='Unet-Mobilenet_v2_161024', model_save_folder=model_save_folder)

    # inference
    # t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    # test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)
    #
    # image, mask = test_set[3]
    # pred_mask, score = predict_image_mask_miou(model, image, mask, device)
    # mob_miou = miou_score(model, test_set, device)
    # mob_acc = pixel_acc_score(model, test_set, device)
    #
    # # plot single image
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    # ax1.imshow(image)
    # ax1.set_title('Picture')
    #
    # ax2.imshow(mask)
    # ax2.set_title('Ground truth')
    # ax2.set_axis_off()
    #
    # ax3.imshow(pred_mask)
    # ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score))
    # ax3.set_axis_off()

    pass
