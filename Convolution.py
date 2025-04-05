import cv2
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.panel.sandwich_covariance_generic import kernel

img =cv2.imread("Thang_1.jpg")
img = cv2.resize(img, (500, 500))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#print(img_gray.shape)

class Conv2d:
    def __init__(self, input, kernelSize):
        self.input = input
        self.height, self.width = input.shape
        self.kernel = np.random.randn(kernelSize, kernelSize)
        self.result = np.zeros((self.height - kernelSize + 1, self.width - kernelSize + 1))

    #Vùng quan tâm
    def getROI(self):
        for row in range(self.height - self.kernel.shape[0] + 1):
            for col in range(self.width - self.kernel.shape[1] + 1):
                roi = self.input[row: row + self.kernel.shape[0], col : col + self.kernel.shape[1]]
                yield row, col, roi

    def operate(self):
        for row, col, roi in self.getROI():
            self.result[row, col] = np.sum(roi * self.kernel)

        return self.result


conv2d = Conv2d(img_gray, 3)
img_gray_conv2d = conv2d.operate()

plt.imshow(img_gray_conv2d, cmap="gray")
plt.show()












"""def conv2d(input, kernelSize):
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
plt.show()"""



