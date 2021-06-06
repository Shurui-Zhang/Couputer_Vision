import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders

    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    # Your code here. You'll need to vectorise your implementation to ensure it runs
    # at a reasonable speed.

    # 使用padding为图像边缘填充
    image_rows = image.shape[0]
    image_cols = image.shape[1]

    kernel_rows = kernel.shape[0]
    kernel_cols = kernel.shape[1]

    new_image_rows = image_rows + kernel_rows - 1
    new_image_cols = image_cols + kernel_cols - 1

    new_image = np.zeros((new_image_rows, new_image_cols))
    rows_add_term = int((new_image_rows - image_rows) / 2)
    cols_add_term = int((new_image_cols - image_cols) / 2)
    for i in range(image_rows):
        for j in range(image_cols):
            new_image[i + rows_add_term, j + cols_add_term] = image[i, j]

    # 矩阵相乘，返回结果
    result = np.zeros((image_rows, image_cols))
    for i in range(image_rows):
        for j in range(image_cols):
            area = new_image[i:i + kernel_rows, j:j + kernel_cols]
            result[i, j] = np.multiply(area, kernel).sum()

    return result