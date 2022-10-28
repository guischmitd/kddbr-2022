from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
from context import DATADIR

def read_before_and_after(img_path, switch_rb=False, normalize=False):
    out = cv2.imread(str(img_path))
    out = out[:, :, ::-1] if switch_rb else out
    out = out / 255. if normalize else out
    
    h, w, c = out.shape
    
    return out[:, :h, :], out[:, h:, :]

def plot_row(row, datadir = DATADIR):
    print(row)
    
    path = datadir / 'raw' / 'train' / 'train' / row['Filename']
    if not path.exists():
        path = datadir / 'raw' / 'test' / 'test' / row['Filename']
    
    img1, img2 = read_before_and_after(path)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img1[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(img2[:, :, ::-1])

    return img1, img2