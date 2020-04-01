import numpy as np
import cv2
from PIL import Image
from os import path
from scipy.linalg import solve
import LU
import time

IMAGE_DIR = 'images/'
img1 = IMAGE_DIR + 'coffee.jpg' # 200 x 200
img2 = IMAGE_DIR + 'turtle.png' # 100 x 100


'''
helper
'''
def show_img(title, img, show):
	if show: cv2.imshow(title, img)


'''
masks must be applied 
to flattened-images
'''
def flatten_mult(mask, A):
	flattened_img = A.flatten()
	result = np.matmul(mask, A)
	# restore image-dimensions
	return result.reshape(A.shape)


'''
apply horizontal
motion-blur to img
'''
def blur(orig_img):
	L = orig_img.shape[0]
	N = int(round(0.1 * orig_img.shape[0], 0))
	_n = N
	# create mask
	mask = np.zeros((L, L))
	for r, row in enumerate(mask):
		if r+N > L:
			_n -= 1
		row[r:r+_n] = [round(1/N, 2)]*_n
	# blur image using the mask
	blurred_img = flatten_mult(mask, orig_img)
	cv2.imwrite('blurred_img.png', blurred_img)
	# normalize pixels, which is required in cv2.imshow()
	blurred_img = (blurred_img-blurred_img.min())/(blurred_img.max()-blurred_img.min())
	
	return blurred_img, mask


'''
unblur an image
given the mask
'''
def unblur(blurred_img, mask):
	print('de-blurring...')
	inv_mask = LU.inv(mask)
	unblurred_img = flatten_mult(inv_mask, blurred_img)
	cv2.imwrite('unblurred_img.png', unblurred_img*255)
	return unblurred_img


# .......... demo ..........

img = np.array(cv2.imread(img2, cv2.IMREAD_GRAYSCALE))
blurred_img, mask = blur(img)
unblurred_img = unblur(blurred_img, mask)

# display
show_img('orig', img, False)
show_img('blurred', blurred_img, False)
show_img('unblurred', unblurred_img, True)
cv2.waitKey(0)