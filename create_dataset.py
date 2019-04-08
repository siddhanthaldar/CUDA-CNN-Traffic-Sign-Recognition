import numpy as np
import os
import cv2

file = open("dataset.txt", 'w')
dataset = []

classes = os.listdir("GTSRB/Final_Training/Images/")
for class_no, folder in enumerate(classes):
	print class_no
	images = os.listdir('GTSRB/Final_Training/Images/' + folder)
	for image in images:
		name = image[9:11]
		if name != '29':
			continue
		img = cv2.imread('GTSRB/Final_Training/Images/' + folder + '/' + image, 1)
		img = cv2.resize(img, (32, 32))
		# print img.shape
		data = [int(folder)]
		for i in range(0, 3):
			data += list(img[:, :, i].reshape(1, -1).squeeze())
		dataset.append(data)

dataset = np.array(dataset, dtype = np.uint8)
print dataset.shape
shuffle = np.arange(len(dataset))
np.random.shuffle(shuffle)
dataset	= dataset[shuffle]
np.savetxt("dataset.txt", dataset, fmt = '%i', delimiter = ' ', newline = '\n')

## For testing the saved dataset
# file = open("dataset.txt", 'r')
# data = file.read()
# d = data.split('\n')[33]
# d = d.split(' ')
# print d[0]
# d = d[1:]
# img = np.zeros((32, 32, 3), dtype = np.uint8)
# img[:, :, 0] = np.array(d[0:32*32]).reshape(32, 32)
# img[:, :, 1] = np.array(d[32*32:32*32*2]).reshape(32, 32)
# img[:, :, 2] = np.array(d[32*32*2:32*32*3]).reshape(32, 32)
# cv2.imshow("image", img)
# cv2.waitKey(0)