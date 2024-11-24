import cv2
import numpy as np
import  matplotlib.pylab as plt
image=cv2.imread("D:\study\B.Tech\Project\Inputs\Roadway.jpeg.png")
ret, thresh = cv2.threshold(image, 130, 145, cv2.THRESH_BINARY)

# plot image
plt.subplot(2,2,1)
plt.imshow(thresh, cmap= "gray")


mask = np.zeros_like(thresh)
# if you pass an image with more then one channel
if len(thresh.shape) > 2:
		channel_count = thresh.shape[2]
		ignore_mask_color = (255,) * channel_count
	# our image only has one channel so it will go under "else"
else:
		# color of the mask polygon (white)
    ignore_mask_color = 255
	# creating a polygon to focus only on the road in the picture
	# we have created this polygon in accordance to how the camera was placed
rows, cols = thresh.shape[:2]
bottom_left = [cols * 0.1, rows * 0.95]
top_left	 = [cols * 0.4, rows * 0.6]
bottom_right = [cols * 0.9, rows * 0.95]
top_right = [cols * 0.6, rows * 0.6]
vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# filling the polygon with white color and generating the final mask
cv2.fillPoly(mask, vertices, ignore_mask_color)
	# performing Bitwise AND on the input image and mask to get only the edges on the road
masked_image = cv2.bitwise_and(thresh, mask)
plt.subplot(2,2,2)
plt.imshow(masked_image)
plt.title("Region Selection")

laplacian_Guassain = np.array([[0,0,-1,0,0],
                    [0, -1, -2,-1,0],
                    [-1, -2, 16,-2,-1],
                    [0, -1, -2,-1,0],
                    [0, 0, -1,0,0]])
Guassain = cv2.filter2D(masked_image, -1, laplacian_Guassain)
plt.subplot(2, 2, 3), plt.imshow(Guassain, cmap='gray')
plt.title('laplacian of Guassain'), plt.xticks([]), plt.yticks([])

img = cv2.GaussianBlur(Guassain,(3,3),0)
edges=cv2.Canny(img,50,150)
plt.subplot(2,2,4),plt.imshow(edges,cmap='gray')
plt.title('canny Edge Detection')
plt.show()
