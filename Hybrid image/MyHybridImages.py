import math
import numpy as np

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:


    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour
    shape=(rows,cols,channels))
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float

    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour
    shape=(rows,cols,channels))
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage
    before subtraction to create the high-pass filtered image
    :type float

    :returns returns the hybrid image created
       by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
       a high-pass image created by subtracting highImage from highImage convolved with
       a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """
    # Your code here.

    high_original_image = np.copy(highImage)

    # 创建卷积核
    lowImage_kernel = makeGaussianKernel(lowSigma)
    highImage_kernel = makeGaussianKernel(highSigma)

    # 判断是否为彩色图像
    if lowImage.ndim == 3 and highImage.ndim == 3:
        for i in range(lowImage.shape[2]):
            # 对彩色图像中每一个通道卷积后放入原图的矩阵中
            lowImage[:, :, i] = convolve(lowImage[:, :, i], lowImage_kernel)
            highImage[:, :, i] = convolve(highImage[:, :, i], highImage_kernel)

    elif lowImage.ndim == 3 and highImage.ndim == 2:
        for i in range(lowImage.shape[2]):
            lowImage[:, :, i] = convolve(lowImage[:, :, i], lowImage_kernel)

        # 对灰度图像直接处理
        highImage = convolve(highImage, highImage_kernel)

    elif lowImage.ndim == 2 and highImage.ndim == 3:
        for i in range(highImage.shape[2]):
            highImage[:, :, i] = convolve(highImage[:, :, i], highImage_kernel)

        lowImage = convolve(lowImage, lowImage_kernel)

    else:
        highImage = convolve(highImage, highImage_kernel)
        lowImage = convolve(lowImage, lowImage_kernel)

    # 获得highImage
    highImage = high_original_image - highImage

    hybridImage = highImage + lowImage

    return hybridImage


def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    # Your code here.

    # 计算kernel的大小
    kernel_size = int(8 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.zeros((kernel_size, kernel_size))

    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - int(kernel_size / 2)
            y = j - int(kernel_size / 2)
            # 二维高斯核函数
            kernel[i, j] = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
            sum += kernel[i, j]

    return kernel / sum