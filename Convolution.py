import cv2
import numpy as np
import matplotlib.pyplot as plt

img =cv2.imread("Thang_1.jpg")
img = cv2.resize(img, (500, 500))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#print(img_gray.shape)

def conv2d(input, kernelSize):
    height, width = input.shape
    kernel = np.random.randn(kernelSize, kernelSize)
    #print(kernel)
    result = np.zeros((height - kernelSize + 1, width + kernelSize + 1))

    for row in range(0, height - kernelSize + 1):
        for col in range(0, width - kernelSize + 1):
            result[row, col] = np.sum(input[row : row + kernelSize, col : col + kernelSize] * kernel)
    return result

img_gray_cov2d = conv2d(img_gray, 3)

plt.imshow(img_gray_cov2d, cmap='gray')
plt.show()