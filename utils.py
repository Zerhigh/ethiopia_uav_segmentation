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
    #img.show()
    img.save('colored_squares_with_legend.png')

