import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('D:\study\B.Tech\Project\Inputs\Roadway.jpeg.png',0)
img = cv2.GaussianBlur(img1,(3,3),0)


# convolute with proper kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)


plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()