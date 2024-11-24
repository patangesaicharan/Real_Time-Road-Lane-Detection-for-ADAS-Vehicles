import cv2
import numpy as np
from matplotlib import pyplot as plt
imag = cv2.imread('D:\study\B.Tech\Project\Inputs\Roadway.jpeg.png',0)
ret, thresh = cv2.threshold(imag, 130, 145, cv2.THRESH_BINARY)

# plot image
plt.figure(figsize=(10,10))
plt.imshow(thresh, cmap= "gray")
plt.show()