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

print(images.shape)

weights = np.random.uniform(-0.1, 0.1, (20,961))
#weights = np.random.uniform(-0.1, 0.1, (200,784))
#weights = np.random.uniform(-0.1, 0.1, (200, 1568))

learn_rate = 0.01
epochs = 60

def sigmoid(x):
	out = np.zeros(x.shape)
	for i in range(out.shape[0]):
		if x[i] >= 0:
			out[i] = 1/(1+np.exp(-x[i]))
		else:
			out[i] = np.exp(x[i])/(1+np.exp(x[i]))
	return out
	#return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

def softmax(x):
	return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)), axis=0)		#NOTE If we using batches we will need axis=1

for k in range(epochs):
	print("Starting epoch: ", k)
	for v in images:
		#print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$-h-$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
		h = np.random.binomial(1,sigmoid(np.dot(weights, v)))
		pos_grad = np.dot(h.reshape(20,1), v.reshape(1,961))
		for i in range(10):
			#v_prime = np.random.binomial(1, softmax(np.dot(h, weights)))
			v_prime = np.random.binomial(1, sigmoid(np.dot(h, weights)))
			h = np.random.binomial(1, sigmoid(np.dot(weights, v_prime)))
		neg_grad = np.dot(h.reshape(20,1), v_prime.reshape(1, 961))
		delta_w = pos_grad - neg_grad
		#print("################################################################")
		#print(delta_w)
		#print("################################################################")
		weights = weights + (learn_rate * delta_w)
		#print(np.max(weights))

for i in range(10):
	test_img = np.copy(images[i*10])
	fig, (ax1, ax2) = plt.subplots(1,2)
	ax1.imshow(test_img.reshape(31,31))
	h_val = np.random.binomial(1,sigmoid(np.dot(weights, test_img)))
	#out = softmax(np.dot(h_val, weights))
	out = sigmoid(np.dot(h_val, weights))
	#ax3.imshow(np.random.binomial(1, softmax(np.dot(h_val, weights))).reshape(28,28))
	ax2.imshow(np.random.binomial(1, sigmoid(np.dot(h_val, weights))).reshape(31,31))
	plt.savefig("test_" + str(i))

np.savetxt("weights.txt", weights)
