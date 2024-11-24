import cv2
import numpy as np
from matplotlib import pyplot as plt
imag = cv2.imread('D:\study\B.Tech\Project\Inputs\Roadway.jpeg.png',cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(imag,(3,3),0)
edges=cv2.Canny(img,50,150)
plt.figure(figsize=(12,6))
plt.subplot(1,1,1),plt.imshow(edges,cmap='gray')
plt.title('canny Edge Detection'),plt.axis('off')
plt.show()