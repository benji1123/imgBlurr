import numpy as np
import cv2
from PIL import Image
from os import path
from scipy.linalg import solve
import LU
import time, sys


'''
masks must be applied 
to flattened-images
'''
def flatten_mult(mask, img):
	result = np.matmul(mask, img)
	# restore image-dimensions
	return result


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
		if r+N > L: _n -= 1
		row[r:r+_n] = [round(1/N, 2)]*_n
	# blur image using the mask
	blurred_img = flatten_mult(mask, orig_img)
	if WRITE:
		cv2.imwrite('images/blurred_img.png', blurred_img)
	# normalize pixels, which is required in cv2.imshow()
	blurred_img = (blurred_img-blurred_img.min())/(blurred_img.max()-blurred_img.min())
	
	return blurred_img, mask


'''
focus-blur
on an img
'''
def focusblur(orig_img):
	L = orig_img.shape[0]
	N = int(round(0.1 * orig_img.shape[0], 0))
	_n = 0

	vec = [0] * (2*N)
	vec[0] = 1
	vec[N-1] = 1
	vec[N] = 4
	vec[N+1] = 1
	vec[2*N-1] = 1

	# create mask
	mask = np.zeros((L, L))
	_n = 2*N
	for r in range(N):
		_n -= 1
		mask[r][:r+1] = vec[_n:]

	_n = 0
	for r in range(N, mask.shape[0]):
		if (r+(N-_n)) > L: _n += 1
		mask[r][r-N:r+(N-_n)] = vec[:2*N-_n]

	# blur img
	blurred_img = flatten_mult(mask, orig_img)
	if WRITE:
		cv2.imwrite('images/focusblur_img.png', blurred_img)
	# normalize pixels for cv2.imshow()
	blurred_img = (blurred_img-blurred_img.min())/(blurred_img.max()-blurred_img.min())
	
	return blurred_img, mask



'''
unblur an image
given the mask
'''
def unblur(blurred_img, mask):
	inv_mask = LU.solve(mask, np.identity(blurred_img.shape[0]))
	unblurred_img = flatten_mult(inv_mask, blurred_img)
	if WRITE:
		cv2.imwrite('images/unblurred_img.png', unblurred_img*255)
	return unblurred_img

# .................... demo ....................

img1 = 'images/offee.jpg'  # 200 x 200
img2 = 'images/turtle.png'  # 100 x 100
img3 = 'images/toronto.png' # 360 x 254



img = img2
img = np.array(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
blurred_img, mask = focusblur(img)
unblurred_img = unblur(blurred_img, mask)

# display
merged = np.hstack((img/256, blurred_img, unblurred_img))
cv2.imshow('input, blurred, unblurred', merged)
cv2.waitKey(0)