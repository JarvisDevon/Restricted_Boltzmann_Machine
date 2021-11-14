import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import gzip

plt.gray()
np.set_printoptions(threshold=np.inf)

f = gzip.open('data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = np.load(f, encoding='latin1')
read_train_x, read_train_y = training_data
read_val_x, read_val_y = validation_data
read_test_x, read_test_y = test_data

pos_train_x = np.where(read_train_x >= 0.5, 1.0, 0.0)
pos_val_x = np.where(read_val_x >= 0.5, 1.0, 0.0)
pos_test_x = np.where(read_test_x >= 0.5, 1.0, 0.0)

denom = (1.0/(pos_train_x.shape[0]-1.0))

ims_T = pos_train_x.T
mu = np.mean(ims_T, axis=1)
mu_col = mu.reshape(784,1)
sigma =  denom*np.dot((ims_T - mu_col),(ims_T - mu_col).T)
eig_value, eig_vectors = np.linalg.eigh(sigma)
print(eig_value)

for i in range(20):
	plt.figure()
	plt.imshow(eig_vectors[:,-1-i].T.reshape(28,28))
	plt.axis('off')
	plt.savefig("pca_out_mnist/component_" + str(i+1) + ".png")
