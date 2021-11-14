import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import gzip

plt.gray()
np.set_printoptions(threshold=np.inf)

f = open('data/nums/images.txt', 'r')
img_strings = f.read()
img_strings = img_strings.split('\n')
img_strings = img_strings[:-1]
imgs = []
for strng in img_strings:
	split_strng = strng.split(' ')
	num = np.array([int(float(d)) for d in split_strng])
	imgs.append(num)

images = np.array(imgs)
denom = (1.0/(images.shape[0]-1.0))

ims_T = images.T
mu = np.mean(ims_T, axis=1)
mu_col = mu.reshape(961,1)
sigma =  denom*np.dot((ims_T - mu_col),(ims_T - mu_col).T)
eig_value, eig_vectors = np.linalg.eigh(sigma)
print(eig_value)

for i in range(20):
	plt.figure()
	plt.imshow(eig_vectors[:,-1-i].T.reshape(31,31))
	plt.axis('off')
	plt.savefig("pca_out_ims/component_" + str(i+1) + ".png")
