import numpy as np
import cv2
from PIL import Image

IMAGE_DIR = 'images/'


# motion blur algorithm
def horizontal_motion_blur(orig_img):
	print('horizontal motion blur!')
	new_img = orig_img

	# apply blurring to each row independently
	for i in range(orig_img.shape[0]):
		N = round(orig_img.shape[0] * 0.1)
		for j in range(orig_img.shape[1]):
			N = N-1 if (j + N > orig_img.shape[1]) else N
			new_img[i,j] = np.mean(new_img[i][j:j+N])
	print(new_img[0,0:3])


# read img
img_addr = IMAGE_DIR + 'toronto.png'
img = cv2.imread(img_addr, cv2.IMREAD_GRAYSCALE) # bnw (1 channel)
img = np.array(img)
cv2.imshow('input img',img)
blurred_img = horizontal_motion_blur(img)


# close CV2 windows with 'q'
cv2.waitKey(0)