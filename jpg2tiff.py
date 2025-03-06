import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing


def morphological_processing(index_array, class_index):
    """对指定类别进行形态学闭运算"""
    binary_mask = (index_array == class_index)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    processed_mask = binary_closing(binary_mask, structure=structure, iterations=1)
    return processed_mask


def jpg_to_tiff_mask(jpg_path, output_folder):
    # 读取图像
    img = Image.open(jpg_path).convert("RGB")
    rgb_array = np.array(img)

    # 初始化索引矩阵
    h, w = rgb_array.shape[:2]
    index_array = np.zeros((h, w), dtype=np.uint8)

    # 矢量化的颜色判断
    red_mask = np.logical_and(
        np.logical_and(125 <= rgb_array[:, :, 0], rgb_array[:, :, 0] <= 130),
        np.logical_and(rgb_array[:, :, 1] <= 5, rgb_array[:, :, 2] <= 5)
    )
    green_mask = np.logical_and(
        np.logical_and(0 <= rgb_array[:, :, 0], rgb_array[:, :, 0] <= 50),
        np.logical_and(80 <= rgb_array[:, :, 1], rgb_array[:, :, 1] <= 150),
        np.logical_and(0 <= rgb_array[:, :, 2], rgb_array[:, :, 2] <= 50)
    )
    black_mask = np.logical_and(
        np.logical_and(rgb_array[:, :, 0] == 0, rgb_array[:, :, 1] == 0),
        rgb_array[:, :, 2] == 0
    )

    # 优先级：黑色 > 红色 > 绿色
    index_array[black_mask] = 0
    index_array[red_mask] = 1
    index_array[green_mask] = 2

    # 形态学处理
    red_processed = morphological_processing(index_array, 1)
    green_processed = morphological_processing(index_array, 2)
    index_array[red_processed] = 1
    index_array[green_processed] = 2
    index_array[black_mask] = 0

    # 转换为单通道灰度图 (L模式)
    mask_img = Image.fromarray(index_array, mode='L')

    # 保存为单通道TIFF
    output_path = os.path.join(output_folder,
                               os.path.basename(jpg_path).replace('.jpg', '_mask.tiff'))
    mask_img.save(output_path, compression="tiff_lzw")
    print(f"Successfully saved: {output_path}")


if __name__ == "__main__":
    input_folder = "C:/Users/18281/OneDrive/Desktop/label"
    output_folder = "C:/Users/18281/OneDrive/Desktop/label"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            jpg_path = os.path.join(input_folder, filename)
            jpg_to_tiff_mask(jpg_path, output_folder)