import os
import pandas as pd
import numpy as np


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
    data.index = data['name']
    return data


def open_class_csv(filepath):
    data = pd.read_csv(filepath)
    data.index = data['name']
    return data
