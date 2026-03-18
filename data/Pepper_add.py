import os
import cv2
import numpy as np
from skimage import util

def process_images(input_folder, output_folder):
    """
    读取输入文件夹中的灰度图，添加椒盐噪声并保存到输出文件夹
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """

    # 支持的图像文件扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 遍历输入文件夹中的所有文件
    for ratio in pepper_list:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(valid_extensions):
                exp_i = filename.split('.jpg')[0]
                # 读取灰度图
                input_path = os.path.join(input_folder, filename)
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取

                if image is None:
                    print(f"无法读取图像: {input_path}")
                    continue

                # 将图像归一化到 [0, 1] 范围（scikit-image 要求）
                image_normalized = image / 255.0

                # 添加 50% 椒盐噪声
                noisy_image = util.random_noise(image_normalized, mode='s&p', amount=ratio)

                # 将图像转换回 0-255 范围并转换为 uint8 类型
                noisy_image = (noisy_image * 255).astype(np.uint8)

                # 保存到输出文件夹
                output_path = os.path.join(output_folder, str(ratio))
                if not os.path.exists(output_path):
                        os.makedirs(output_path)
                cv2.imwrite(output_path + '/' + str(exp_i) + '.png', noisy_image)
                print(f"已处理并保存: {output_path}")

                # 可选：显示原始图像和加噪图像（调试用）
                # plt.subplot(1, 2, 1)
                # plt.title("Original")
                # plt.imshow(image, cmap='gray')
                # plt.subplot(1, 2, 2)
                # plt.title("Noisy")
                # plt.imshow(noisy_image, cmap='gray')
                # plt.show()

# 设置输入和输出文件夹路径
input_folder = '/share/home/leader/yuzhuliang/MJJ/denoise/data/Images_gray'
output_folder = '/share/home/leader/yuzhuliang/MJJ/denoise/data/Images_pepper_salt'

# 执行处理
pepper_list = [0.1, 0.2, 0.3, 0.4, 0.5]
process_images(input_folder, output_folder)