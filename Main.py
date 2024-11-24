import cv2
import numpy as np
import  matplotlib.pylab as plt
image=cv2.imread("D:/study/B.Tech/Project/Inputs/Roadway.jpeg.png")
plt.subplot(3,3,1)
plt.imshow(image)
plt.title("Original image")
# create an array of the same size as of the input image
# region selection
mask = np.zeros_like(image)
# if you pass an image with more then one channel
if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	# our image only has one channel so it will go under "else"
else:
		# color of the mask polygon (white)
    ignore_mask_color = 255
	# creating a polygon to focus only on the road in the picture
	# we have created this polygon in accordance to how the camera was placed
rows, cols = image.shape[:2]
bottom_left = [cols * 0.1, rows * 0.95]
top_left	 = [cols * 0.4, rows * 0.6]
bottom_right = [cols * 0.9, rows * 0.95]
top_right = [cols * 0.6, rows * 0.6]
vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# filling the polygon with white color and generating the final mask
cv2.fillPoly(mask, vertices, ignore_mask_color)
	# performing Bitwise AND on the input image and mask to get only the edges on the road
masked_image = cv2.bitwise_and(image, mask)
plt.subplot(3,3,2)
plt.imshow(masked_image)
plt.title("Region Selection")


# threshold
ret, thresh = cv2.threshold(image, 130, 145, cv2.THRESH_BINARY)

# plot image
plt.subplot(3,3,3)
# plt.figure(figsize=(10,10))
plt.imshow(thresh, cmap= "gray")
plt.title("threshold")

# Sobel
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator
sobel_x = np.array([[-1,0,1],
                    [ -2, 0 , 2],
                    [ -1,0,1]])
     

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_y = cv2.filter2D(image, -1, sobel_y)
filtered_image_x = cv2.filter2D(image, -1, sobel_x)
plt.subplot(3,3,4),plt.imshow(filtered_image_x,cmap = 'gray')
plt.title('sobel x'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5),plt.imshow(filtered_image_y,cmap = 'gray')
plt.title('sobel y'), plt.xticks([]), plt.yticks([])


#LOG
laplacian_Guassain = np.array([[0,0,-1,0,0],
                    [0, -1, -2,-1,0],
                    [-1, -2, 16,-2,-1],
                    [0, -1, -2,-1,0],
                    [0, 0, -1,0,0]])
Guassain = cv2.filter2D(image, -1, laplacian_Guassain)
plt.subplot(3, 3, 6), plt.imshow(image, cmap='gray')
plt.title('Gray scale'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 7), plt.imshow(Guassain, cmap='gray')
plt.title('laplacian of Guassain'), plt.xticks([]), plt.yticks([])
 
# canny
img = cv2.GaussianBlur(image,(3,3),0)
edges=cv2.Canny(img,50,150)
plt.subplot(3,3,8),plt.imshow(edges,cmap='gray')
plt.title('canny Edge Detection')

# Hough
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # Perform Hough line detection
# lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=105)

# # Draw detected lines on the original image
# if lines is not None:
#     for line in lines:
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# plt.subplot(3,3,9), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image with Hough Lines')

# plt.subplot(132), plt.imshow(edges, cmap='gray')
# plt.title('Canny Edge Detection'), plt.axis('off')
plt.show()
