import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

def read_img(path, greyscale=True):
    try:
        # 尝试打开图像
        img = Image.open(path)
    except FileNotFoundError:
        # 如果文件不存在，打印警告并跳过
        print(f"Warning: File not found - {path}")
        return None
    if greyscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.array(img) 
    
def save_img(img, path):
    img = Image.fromarray(img)
    img.save(path)
    print(path, "is saved!")

def display_img(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
