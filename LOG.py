import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('D:\study\B.Tech\Project\Inputs\Roadway.jpeg.png',0)
#laplacian of Guassain
laplacian_Guassain = np.array([[0,0,-1,0,0],
                    [0, -1, -2,-1,0],
                    [-1, -2, 16,-2,-1],
                    [0, -1, -2,-1,0],
                    [0, 0, -1,0,0]])
Guassain = cv2.filter2D(img1, -1, laplacian_Guassain)
plt.subplot(2, 2, 1), plt.imshow(img1, cmap='gray')
plt.title('Gray scale'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(Guassain, cmap='gray')
plt.title('laplacian of Guassain'), plt.xticks([]), plt.yticks([])
plt.show()