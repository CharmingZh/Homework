from common import * 
import matplotlib.pyplot as plt
import numpy as np



def gaussian_filter(image, sigma):
    # Given an image, apply a Gaussian filter with the input kernel size
    # and standard deviation
    # Input-    image: image of size HxW
    #           sigma: scalar standard deviation of Gaussian Kernel
    # Output-   Gaussian filtered image of size HxW
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2*sigma) + 1)

    # make sure that kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H,W)//2)
    if kernel_size % 2 == 0: kernel_size = kernel_size + 1

    #TODO implement gaussian filtering with size kernel_size x kernel_size
    # feel free to use your implemented convolution function or a convolution function from a library
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))

    kernel /= np.sum(kernel)

    from filters import convolve
    filtered_image = convolve(image, kernel)

    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    filtered_image = scipy_gaussian_filter(image, sigma=sigma)
    return filtered_image

def scale_space(image, min_sigma, k=np.sqrt(2), S=8):
    # Calcualtes a DoG scale space of the image
    # Input-    image: image of size HxW
    #           min_sigma: smallest sigma in scale space
    #           k: scalar multiplier for scale space
    #           S: number of scales considers
    # Output-   Scale Space of size HxWx(S-1)
    H, W = image.shape
    scale_space = np.zeros((H, W, S - 1))
    for s in range(S):
        sigma = min_sigma * (k ** s)
        gauss = gaussian_filter(image, sigma)
        if s > 0:
            DoG = gauss - prev_gauss
            scale_space[:, :, s - 1] = DoG
        prev_gauss = gauss
    return scale_space


##### You shouldn't need to edit the following 3 functions 
def find_maxima(scale_space, k_xy=5, k_s=1):
    # Extract the peak x,y locations from scale space
    # Input-    scale_space: Scale space of size HxWxS
    #           k: neighborhood in x and y 
    #           ks: neighborhood in scale
    # Output-   list of (x,y) tuples; x<W and y<H
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 

    H,W,S = scale_space.shape
    maxima = []
    for i in range(H):
        for j in range(W):
            for s in range(S):
                # extracts a local neighborhood of max size (2k_xy+1, 2k_xy+1, 2k_s+1)
                neighbors = scale_space[max(0, i-k_xy):min(i+k_xy,H), 
                                        max(0, j-k_xy):min(j+k_xy,W), 
                                        max(0, s-k_s) :min(s+k_s,S)]
                mid_pixel = scale_space[i,j,s]
                num_neighbors = np.prod(neighbors.shape) - 1
                # if mid_pixel is larger than all the neighbors; append maxima 
                if np.sum(mid_pixel > neighbors) == num_neighbors:
                    maxima.append( (i,j,s) )
    return maxima

def visualize_scale_space(scale_space, min_sigma, k, file_path=None):
    # Visualizes the scale space
    # Input-    scale_space: scale space of size HxWxS
    #           min_sigma: the minimum sigma used 
    #           k: the sigma multiplier 
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 
    H, W, S = scale_space.shape

    # number of subplots
    p_h = int(np.floor(np.sqrt(S))) 
    p_w = int(np.ceil(S/p_h))
    for i in range(S):
        plt.subplot(p_h, p_w, i+1)
        plt.axis('off')
        plt.title('{:.1f}:{:.1f}'.format(min_sigma * k**i, min_sigma * k**(i+1)))
        plt.imshow(scale_space[:, :, i])

    # plot or save to fig 
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    

def visualize_maxima(image, maxima, min_sigma, k, file_path=None):
    # Visualizes the maxima on a given image
    # Input-    image: image of size HxW
    #           maxima: list of (x,y) tuples; x<W, y<H
    #           file_path: path to save image. if None, display to screen
    # Output-   None 
    H, W = image.shape
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for maximum in maxima:
        y,x,s= maximum 
        assert x < W and y < H and x >= 0 and y >= 0
        radius = np.sqrt(2 * min_sigma * (k ** s))
        circ = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_patch(circ)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    


def main():
    image = read_img('polka.png')

    ### -- Detecting Polka Dots -- ## 
    print("Detect small polka dots")
    # -- Detect Small Circles
    r_small = 5

    sigma_small_1  = r_small / np.sqrt(2)
    sigma_small_2  = sigma_small_1 * 1.5    # 调整比例

    gauss_small_1  = gaussian_filter(image, sigma_small_1 )
    gauss_small_2  = gaussian_filter(image, sigma_small_2 )

    # calculate difference of gaussians
    DoG_small = gauss_small_2  - gauss_small_1

    # visualize maxima 
    maxima = find_maxima(DoG_small, k_xy=5, k_s=1)
    visualize_scale_space(DoG_small, sigma_small_1 , sigma_small_2 / sigma_small_1 ,'polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_small_1 , sigma_small_2 / sigma_small_1 , 'polka_small.png')
    
    # -- Detect Large Circles
    print("Detect large polka dots")
    r_large = 11

    sigma_large_1  = r_large / np.sqrt(2)
    sigma_large_2  = sigma_large_1  * 1.5
    gauss_large_1  = gaussian_filter(image, sigma_large_1 )
    gauss_large_2  = gaussian_filter(image, sigma_large_2 )

    # calculate difference of gaussians
    DoG_large = gauss_large_2  - gauss_large_1

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_large_1 , sigma_large_2 /sigma_large_1 , 'polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_large_1 , sigma_large_2 /sigma_large_1 , 'polka_large.png')


    # ## -- TODO Implement scale_space() and try to find both polka dots
    #
    # # 生成尺度空间并检测多尺度斑点
    # min_sigma = sigma_small_1
    # k = np.sqrt(2)
    # S = 8
    # scale_space_result = scale_space(image, min_sigma, k, S)
    #
    # # 可视化尺度空间
    # visualize_scale_space(scale_space_result, min_sigma, k, 'polka_scale_space.png')
    #
    # ## 自动检测极大值
    # k_xy_values = [3, 5, 7]
    # k_s_values = [1, 2]
    # for k_xy in k_xy_values:
    #     for k_s in k_s_values:
    #         maxima = find_maxima(scale_space_result, k_xy=k_xy, k_s=k_s)
    #         output_filename = f'polka_maxima_kxy{k_xy}_ks{k_s}.png'
    #         visualize_maxima(image, maxima, min_sigma, k, output_filename)
    #         print(f"k_xy={k_xy}, k_s={k_s}, 检测到的斑点数量：{len(maxima)}")
    #
    # ## -- TODO Try to find the cells in any of the cell images in vgg_cells
    # print("检测细胞并计数")
    # import os
    # cell_images = [f'cells/{i:03d}cell.png' for i in range(1, 201)]
    #
    # import cupy as cp
    # from cupyx.scipy.ndimage import gaussian_filter as cuda_gaussian_filter
    #
    # for img_path in cell_images:
    #     print(f"处理图像：{img_path}")
    #     image = read_img(img_path)
    #     if image is None:
    #         continue
    #
    #     # 预处理，如高斯平滑或对比度增强
    #
    #     preprocessed_image = gaussian_filter(image, sigma=1)
    #
    #     # 生成尺度空间
    #     min_sigma = 2
    #     k = 1.2
    #     S = 10
    #     scale_space_result = scale_space(preprocessed_image, min_sigma, k, S)
    #
    #     # 自动检测极大值
    #     k_xy = 5
    #     k_s = 2
    #     maxima = find_maxima(scale_space_result, k_xy=k_xy, k_s=k_s)
    #
    #     # 可视化结果
    #     output_filename = os.path.basename(img_path).replace('.png', '_detection.png')
    #     visualize_maxima(image, maxima, min_sigma, k, output_filename)
    #
    #     # 输出检测到的细胞数量
    #     print(f"{img_path} 检测到的细胞数量：{len(maxima)}")

if __name__ == '__main__':
    main()
