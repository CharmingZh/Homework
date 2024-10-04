import numpy as np
import os
from common import *
import cv2


## Image Patches ##
def image_patches(image, patch_size=(16,16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N 
    # Output- results: a list of images of size M x N

    # TODO: Use slicing to complete the function
    H, W = image.shape

    H_new, W_new = H - (H % patch_size[0]), W - (W % patch_size[1])
    img_gray = cv2.resize(image, (W_new, H_new))

    output = []

    for i in range(0, img_gray.shape[0], patch_size[0]):
        for j in range(0, img_gray.shape[1], patch_size[1]):
            # divide the image into 16x16 pixel patches
            patch_tmp = img_gray[i:i + patch_size[0], j:j + patch_size[1]]
            # normalize
            patch_mean = np.mean(patch_tmp)
            patch_std = np.std(patch_tmp)
            normalized_patch = (patch_tmp - patch_mean) / patch_std

            output.append(normalized_patch)

    return output


## Gaussian Filter ##
def convolve(image, kernel):
    # Return the convolution result: image * kernel.
    # Reminder to implement convolution and not cross-correlation!
    # Input- image: H x W
    #        kernel: h x w
    # Output- convolve: H x W

    H, W = image.shape
    h, w = kernel.shape

    # Pad the image
    padded_image = np.pad(image,
                          ((h // 2, h // 2), (w // 2, w // 2)),
                          mode = 'constant',
                          constant_values = 0
                          )

    output = image.copy()

    # Convolution operation
    for i in range(H):
        for j in range(W):
            # Extract the region of interest
            roi = padded_image[i:i + h, j:j + w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(roi * kernel)

    return output


## Edge Detection ##
def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    # TODO: Fix kx, ky
    kx = np.array([[np.float(-0.5), np.float(0), np.float(0.5)]])  # 1 x 3
    ky = np.array([[np.float(-0.5)], [np.float(0)], [np.float(0.5)]])  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    # 将梯度幅值归一化到 0-255，并转换为 uint8
    grad_magnitude = (grad_magnitude / np.max(grad_magnitude)) * 255
    grad_magnitude = grad_magnitude.astype(np.uint8)

    return grad_magnitude, Ix, Iy


## Sobel Operator ##
def sobel_operator(image, normalize=True):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    # TODO: Use convolve() to complete the function
    # Sobel kernels
    Gx_kernel = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
    Gy_kernel = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])

    # 使用 convolve() 进行卷积
    Gx = convolve(image, Gx_kernel)
    Gy = convolve(image, Gy_kernel)

    # 计算梯度幅值
    grad_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    if normalize:
        # 归一化以便于可视化
        Gx = (Gx - Gx.min()) / (Gx.max() - Gx.min()) * 255
        Gx = Gx.astype(np.uint8)

        Gy = (Gy - Gy.min()) / (Gy.max() - Gy.min()) * 255
        Gy = Gy.astype(np.uint8)

        grad_magnitude = (grad_magnitude / grad_magnitude.max()) * 255
        grad_magnitude = grad_magnitude.astype(np.uint8)

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=[0, np.pi/6, np.pi/3, np.pi/2, np.pi*2/3, np.pi*5/6]):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W
    # You are encouraged not to use sobel_operator() in this function.

    # TODO: Use convolve() to complete the function
    output = []

    # Sobel kernels
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    for alpha in angles:
        # Compute K(alpha)
        K_alpha = np.cos(alpha) * Kx + np.sin(alpha) * Ky

        # Convolve image with K(alpha)
        S = convolve(image, K_alpha)

        # Normalize S for visualization
        S_norm = (S - S.min()) / (S.max() - S.min()) * 255
        S_norm = S_norm.astype(np.uint8)

        output.append(S_norm)

    return output




def main():
    # The main function
    ########################
    img = read_img('./grace_hopper.png')

    ##### Image Patches #####
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # Q1
    patches = image_patches(img)
    # TODO choose a few patches and save them
    import random
    random_indices = random.sample(range(len(patches)), 3)
    chosen_patches = [patches[i] for i in random_indices]
    chosen_patches = np.hstack(chosen_patches)
    # 将浮点数的 NumPy 数组转换为 8 位整型
    chosen_patches = np.uint8(chosen_patches * 255)  # 假设图像的值在 [0, 1] 范围内

    Image.fromarray(chosen_patches).save("./image_patches/q1_patch.png")
    # Image.fromarray(chosen_patches).show()

    # save_img(Image.fromarray(chosen_patches), "./image_patches/q1_patch.png")

    # Q2: No code

    ##### Gaussian Filter #####
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.  There is tolerance for the kernel.
    sigma = np.sqrt(1 / (2 * np.log(2)))
    kernel_size = 3
    kernel_gaussian = np.zeros((kernel_size, kernel_size))

    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - kernel_size // 2, j - kernel_size // 2
            kernel_gaussian[i, j] = (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x**2 + y**2) / (2 * (sigma ** 2)))

    kernel_gaussian /= np.sum(kernel_gaussian)
    print(kernel_gaussian)

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")

    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    ########################

    ##### Sobel Operator #####
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    ########################

    #####LoG Filter#####
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0]
    ])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # Q2: No code

    print("LoG Filter is done. ")
    ########################


if __name__ == "__main__":
    main()
