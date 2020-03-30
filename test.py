import numpy as np


def horizontal_motion_blur(orig_img):
	print('horizontal motion blur!')
	new_img = orig_img

	for i in range(orig_img.shape[0]):
		for j in range(orig_img.shape[1] - 3):
			new_img[i,j] = np.mean(new_img[i][j:j+3])
	return new_img


x = np.random.rand(5,5) * 10
x = x.round(decimals=0)
print(x)

print(horizontal_motion_blur(x))

def horizontal_motion_blur(orig_img):
	print('horizontal motion blur!')
	new_img = orig_img

	for i in range(orig_img.shape[0]):
		for j in range(orig_img.shape[1] - 3):
			new_img[i,j] = np.mean(new_img[i][j:j+3])
	return new_img