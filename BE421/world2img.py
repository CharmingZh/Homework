import cv2
import numpy as np
import random


# 加载图片
image_path = 'jig.jpeg'  # 替换为你的图片路径
image = cv2.imread(image_path)

# 定义放大倍数
scale_factor = 3  # 将图片放大2倍
image_height, image_width = image.shape[:2]
scaled_image = cv2.resize(image, (image_width * scale_factor, image_height * scale_factor))

# 用于绘制的副本
image_display = scaled_image.copy()

# 定义点击相关变量
click_count = 0
max_clicks = 14
clicks = []

# 定义点的字母顺序
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'o', 'p']

# 定义三维坐标信息（你提供的3D点）
world_coordinates = {
    'A': [0, 0, 0], 'B': [0, 6, 0], 'C': [11, 6, 0], 'D': [11, 0, 0],
    'E': [8.25, 0, -4.5], 'F': [2.75, 0, -4.5], 'G': [5.5, 5, -3.5], 'H': [5.5, 6, -3.5],
    'K': [2, 0, 0], 'L': [2, 6, 0], 'M': [9, 6, 0], 'N': [9, 0, 0],
    'O': [8.25, 0, -1.8125], 'P': [2.75, 0, -1.8125]
}

# 初始化所有对应的像素位置
pixel_coordinates = {
    'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [],
    'g': [], 'h': [], 'k': [], 'l': [], 'm': [], 'n': [],
    'o': [], 'p': []
}

# 随机生成颜色
def generate_random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])

# 鼠标点击回调函数
def mouse_callback(event, x, y, flags, param):
    global click_count, clicks, image_display

    if event == cv2.EVENT_LBUTTONDOWN and click_count < max_clicks:
        # 将点击位置从放大后的图像映射回原始图像坐标
        original_x = x // scale_factor
        original_y = y // scale_factor
        label = labels[click_count]
        print(f"Clicked original image pixel coordinates for {label}: {original_x}, {original_y}")

        # 记录点击点
        pixel_coordinates[label] = [original_x, original_y]
        clicks.append((original_x, original_y))
        click_count += 1

        # 在放大的图像上绘制圆形标记点击位置
        cv2.circle(image_display, (x, y), 5, generate_random_color(), -1)
        cv2.putText(image_display, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 如果点击次数达到14次
        if click_count == max_clicks:
            print("All 14 points have been clicked.")
            print("Final pixel coordinates:")
            for key, value in pixel_coordinates.items():
                print(f"{key} = {value}")

# 设置窗口并绑定鼠标事件
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Image', mouse_callback)

# 主循环
while True:
    # 显示图片
    cv2.imshow('Image', image_display)

    # 按键处理，按ESC键退出
    key = cv2.waitKey(1)
    if key == 27:  # ESC 键的 ASCII 码是27
        break

# 保存带有标记点的图片
cv2.imwrite('image_with_points.jpg', image_display)
print("Image with points saved as 'image_with_points.jpg'.")

# 释放窗口
cv2.destroyAllWindows()