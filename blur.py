import numpy as np
import cv2
from PIL import Image

IMAGE_DIR = 'images/'

def show_img(title, img, show):
	if show:
		cv2.imshow(title, img)

# motion blur algorithm
def iterative_blur(orig_img):
	print('horizontal motion blur!')
	new_img = orig_img

	# apply blurring to each row independently
	for i in range(orig_img.shape[0]):
		N = round(orig_img.shape[0] * 0.1)
		for j in range(orig_img.shape[1]):
			N = N-1 if (j + N > orig_img.shape[1]) else N
			new_img[i,j] = np.mean(new_img[i][j:j+N])
	return new_img, N


def blur(orig_img):
	flattened_img = orig_img.flatten()
	L = flattened_img.shape[0]
	N = int(round(0.1 * orig_img.shape[0], 0))

	# mask (A)
	mask = np.zeros((L, L))
	for r, row in enumerate(mask[0:-N]):
		row[r:r+N] = [round(1/N, 2)]*N

	# blurred img = A * flattened_img
	print('starting blurring')
	blurred_img = np.matmul(mask, flattened_img)
	blurred_img = blurred_img.reshape(orig_img.shape)
	cv2.imwrite('blurred_img.png', blurred_img)

	# normalize img to [0,1]
	blurred_img = (
		blurred_img - blurred_img.min()) / (blurred_img.max()-blurred_img.min())
	return blurred_img, mask


# read img
img1 = IMAGE_DIR + 'turtle.png'
img = cv2.imread(img1, cv2.IMREAD_GRAYSCALE) # bnw (1 channel)
img = np.array(img)
show_img('input', img, False)

# motion blurr
blurred_img, mask = blur(img)
show_img('blurred', blurred_img, True)

cv2.waitKey(0)