import numpy as np
import os
import cv2
# 9->0, 35->1, 13->2, 4->3, 38->4, 11->5, 12->6, 25->7, 7->8, 1->9, 3->10, 10->11, 8->12, 2->13, 5->14
file = open("new_dataset.txt", 'w')
dataset = []
k = 0
c = [1,2,3,4,5,7,8,9,10,11,12,13,25,35,38]
classes = os.listdir("GTSRB/Final_Training/Images/")
print(classes)

d = np.zeros((43,43))

for class_no, folder in enumerate(classes):
	print (folder)
	for j in range(len(c)):
		if(c[j] == int(folder)):
			images = os.listdir('GTSRB/Final_Training/Images/' + folder)
			for image in images:
				d[k][int(folder)] = 1
				name = image[9:11]
				if name != '29':
					continue
				img = cv2.imread('GTSRB/Final_Training/Images/' + folder + '/' + image, 1)
				img = cv2.resize(img, (32, 32))
				# print(img.shape)
				data = [int(k)]
				for i in range(0, 3):
					data += list(img[:, :, i].reshape(1, -1).squeeze())
				dataset.append(data)
			k+=1
			print("hello", end=' ')
			print(k)


dataset = np.array(dataset, dtype = np.uint8)
print (dataset.shape)
# for i in range(43):
# 	for j in range(43):
# 		print(d[i][j], end=" ")
# 	print("\n")
shuffle = np.arange(len(dataset))
np.random.shuffle(shuffle)
dataset	= dataset[shuffle]
np.savetxt("new_dataset.txt", dataset, fmt = '%i', delimiter = ' ', newline = '\n')

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