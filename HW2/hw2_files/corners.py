import os

from HW2.hw2_files.filters import sobel_operator
from common import read_img, save_img 
import matplotlib.pyplot as plt
import numpy as np

from filters import *

def corner_score(image, u=5, v=5, window_size=(5,5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image. 

    H, W = image.shape
    output = np.zeros((H, W))

    # 计算窗口的半尺寸
    half_h = window_size[0] // 2
    half_w = window_size[1] // 2

    # 为了处理边界，计算填充大小
    pad_h = half_h + abs(v)
    pad_w = half_w + abs(u)

    # 对图像进行零填充
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # 遍历图像中的每个像素
    for i in range(H):
        for j in range(W):
            # 计算在填充图像中的坐标
            i_padded = i + pad_h
            j_padded = j + pad_w

            # 提取以当前像素为中心的窗口
            window_current = padded_image[i_padded - half_h:i_padded + half_h + 1,
                             j_padded - half_w:j_padded + half_w + 1]

            # 提取偏移后的窗口
            i_shifted = i_padded + v
            j_shifted = j_padded + u
            window_shifted = padded_image[i_shifted - half_h:i_shifted + half_h + 1,
                             j_shifted - half_w:j_shifted + half_w + 1]

            # 计算窗口之间的SSD
            ssd = np.sum((window_current - window_shifted) ** 2)
            output[i, j] = ssd

    return output

def harris_detector(image, window_size=(5,5)):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    # 
    # You can use same-padding for intensity (or zero-padding for derivatives) 
    # to handle window values outside of the image. 

    ## compute the derivatives 
    Ix = None
    Iy = None
    Ix, Iy, _ = sobel_operator(image, normalize=False)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 3. 对梯度乘积进行加权求和（卷积）
    # 创建高斯核
    h, w = window_size
    sigma = h / 2  # 根据窗口大小选择合适的 sigma
    kernel_size = h

    # 创建二维高斯核
    x, y = np.mgrid[-(kernel_size // 2):(kernel_size // 2) + 1, -(kernel_size // 2):(kernel_size // 2) + 1]
    gauss_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    # 对 Ixx, Iyy, Ixy 进行卷积
    Sxx = convolve(Ixx, gauss_kernel)
    Syy = convolve(Iyy, gauss_kernel)
    Sxy = convolve(Ixy, gauss_kernel)

    # 4. 计算 Harris 响应
    alpha = 0.05  # Harris 角点检测器的参数，通常取 0.04 到 0.06

    # 计算行列式和迹
    det_M = Sxx * Syy - Sxy ** 2
    trace_M = Sxx + Syy

    # For each location of the image, construct the structure tensor and calculate the Harris response
    response = det_M - alpha * (trace_M ** 2)

    return response

def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Feature Detection #####  
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # Define shifts and window size
    shifts = [(-5, 0), (5, 0), (0, -5), (0, 5)]  # left, right, up, down shifts
    window_size = (5, 5)

    # Compute and save corner scores for each shift
    for idx, (u, v) in enumerate(shifts):
        score = corner_score(img, u=u, v=v, window_size=window_size)

        # Normalize score for visualization
        score_norm = (score - score.min()) / (score.max() - score.min()) * 255
        score_norm = score_norm.astype(np.uint8)

        # Save the output image
        direction = ['left', 'right', 'up', 'down'][idx]
        save_img(score_norm, f"./feature_detection/corner_score_{direction}.png")

    # define offsets and window size and calulcate corner score
    u, v, W = 0, 2, (5,5)
    
    score = corner_score(img, u, v, W)

    # **添加归一化和类型转换**
    score_norm = (score - score.min()) / (score.max() - score.min()) * 255
    score_norm = score_norm.astype(np.uint8)

    save_img(score_norm, "./feature_detection/corner_score.png")

    harris_corners = harris_detector(img)

    # 对响应值进行归一化，以便保存为图像
    harris_norm = (harris_corners - harris_corners.min()) / (harris_corners.max() - harris_corners.min()) * 255
    harris_norm = harris_norm.astype(np.uint8)

    save_img(harris_norm, "./feature_detection/harris_response.png")

if __name__ == "__main__":
    main()
