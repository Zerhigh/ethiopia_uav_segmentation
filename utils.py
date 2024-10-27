import os
import pickle

import pandas as pd
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def create_df(image_path):
    name = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


def make_folder(dp):
    if not os.path.exists(dp):
        os.mkdir(dp)
    return


def open_class_csv(filepath):
    data = pd.read_csv(filepath)
    data.columns = ['name', 'r', 'g', 'b']

    # remove "conflicting" row
    data = data[data['name'] != 'conflicting']

    return data

def create_image_legend(class_to_rgb, class_labels):
    # Parameters for the layout
    square_size = 50  # Size of each square
    num_classes = len(class_to_rgb)
    legend_width = 200  # Width of the legend
    image_width = square_size + legend_width
    image_height = num_classes * square_size

    # Create a blank image
    img = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(img)

    # Add squares and corresponding labels
    for i, (class_id, rgb) in enumerate(class_to_rgb.items()):
        # Draw the square
        top_left = (0, i * square_size)
        bottom_right = (square_size, (i + 1) * square_size)
        draw.rectangle([top_left, bottom_right], fill=rgb)

        # Add text labels (class name)
        text_position = (square_size + 10, i * square_size + 10)
        draw.text(text_position, class_labels.loc[class_id]['name'], fill='black')

    # Save and display the image
    img.show()
    img.save('colored_squares_with_legend.png')


def extract_gdal_transformations(image_paths, agg_trafo_path):
    transformations = {}

    # save image extensions?
    for img in os.listdir(image_paths):
        src = gdal.Open(os.path.join(image_paths, img))
        trafo = src.GetGeoTransform()
        transformations[img] = trafo

    with open(agg_trafo_path, 'wb') as ftrafo:
        pickle.dump(transformations, ftrafo)

    return


def reapply_gdal_transformations(image_paths, agg_trafo_paths, ref_epsg=4326):

    with open(agg_trafo_paths, 'rb') as ftrafo:
        transformations = pickle.load(ftrafo)

    # hadnle_img_extensions
    for img in os.listdir(image_paths):
        img_wo_ext = os.path.splitext(img)[0]

        trafo = transformations[os.path.join(img_wo_ext, '.tif')]

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(ref_epsg)

        # update image georeference
        ds = gdal.Open(os.path.join(image_paths, img_wo_ext, '.png'), gdal.GA_Update)
        #ds = gdal.Open(img_path, gdal.GA_Update)
        ds.SetGeoTransform(trafo)
        ds.SetProjection(srs.ExportToWkt())
        ds.FlushCache()

        # save as tif?


    return