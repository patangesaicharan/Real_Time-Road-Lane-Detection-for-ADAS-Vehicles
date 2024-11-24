import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image
image = cv2.imread('D:/study/B.Tech/Project/Inputs/road.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(image,(3,3),0)
edges=cv2.Canny(img,50,150)

# Perform Hough line detection
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=310)

# Draw detected lines on the original image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the results
plt.figure(figsize=(12,6))

plt.subplot(1,1,1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image with Hough Lines'), plt.axis('off')

# plt.subplot(132), plt.imshow(edges, cmap='gray')
# plt.title('Canny Edge Detection'), plt.axis('off')
plt.show()