import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import gzip

np.set_printoptions(threshold=np.inf)

f = gzip.open('data/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = np.load(f, encoding='latin1', allow_pickle=True)
read_train_x, read_train_y = training_data
read_val_x, read_val_y = validation_data
read_test_x, read_test_y = test_data

#train_x = 1 - train_x
#val_x = 1 - val_x
#test_x = 1 - test_x

pos_train_x = np.where(read_train_x >= 0.5, 1.0, 0.0)
pos_val_x = np.where(read_val_x >= 0.5, 1.0, 0.0)
pos_test_x = np.where(read_test_x >= 0.5, 1.0, 0.0)

#plt.imshow(pos_train_x[0].reshape(28,28), cmap='gray')
#plt.show()

#neg_train_x = np.copy(pos_train_x)
#neg_val_x = np.copy(pos_val_x)
#neg_test_x = np.copy(pos_test_x)

#neg_train_x = 1 - neg_train_x
#neg_val_x = 1 - neg_val_x
#neg_test_x = 1 - neg_test_x

#train_x = np.dstack([pos_train_x, neg_train_x])
#val_x = np.dstack([pos_val_x, neg_val_x])
#test_x = np.dstack([pos_test_x, neg_test_x])

weights = np.random.uniform(-0.3, 0.3, (300,784))
#weights = np.random.uniform(-0.1, 0.1, (200,784))
#weights = np.random.uniform(-0.1, 0.1, (200, 1568))

learn_rate = 0.01
epochs = 10
fine_epochs = 2

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
	for v in pos_train_x:
		#print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$-h-$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
		h = np.random.binomial(1,sigmoid(np.dot(weights, v)))
		pos_grad = np.dot(h.reshape(300,1), v.reshape(1,784))
		#for i in range(10):
		#v_prime = np.random.binomial(1, softmax(np.dot(h, weights)))
		v_prime = np.random.binomial(1, sigmoid(np.dot(h, weights)))
		h_prime = np.random.binomial(1, sigmoid(np.dot(weights, v_prime)))
		#v_prime_prime = np.random.binomial(1, sigmoid(np.dot(h_prime, weights)))
		#h_prime_prime = np.random.binomial(1, sigmoid(np.dot(weights, v_prime_prime)))
		#v_prime_prime_prime = np.random.binomial(1, sigmoid(np.dot(h_prime_prime, weights)))
		#h_prime_prime_prime = np.random.binomial(1, sigmoid(np.dot(weights, v_prime_prime_prime)))
		neg_grad = np.dot(h_prime.reshape(300,1), v_prime.reshape(1, 784))
		delta_w = pos_grad - neg_grad
		#print("################################################################")
		#print(delta_w)
		#print("################################################################")
		weights = weights + (learn_rate * delta_w)
		#print(np.max(weights))

for i in range(10):
    print("Saving image " + str(i))
    broken_img = np.copy(pos_train_x[i*10])
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(broken_img.reshape(28,28), cmap='gray')
    for j in range(len(broken_img)):
        if(np.random.randint(0, 100) > 96):	#3% of pixels distorted
            if broken_img[j] == 0:
                broken_img[j] = 1
            else:
                broken_img[j] = 0
    h_val = np.random.binomial(1,sigmoid(np.dot(weights, broken_img)))
    #out = softmax(np.dot(h_val, weights))
    out = sigmoid(np.dot(h_val, weights))
    ax2.imshow(broken_img.reshape(28,28), cmap='gray')
    #ax3.imshow(np.random.binomial(1, softmax(np.dot(h_val, weights))).reshape(28,28))
    ax3.imshow(np.random.binomial(1, sigmoid(np.dot(h_val, weights))).reshape(28,28), cmap='gray')
    plt.savefig("test_denoise/img_" + str(i) + ".png")

g = open('data/nums/images.txt', 'r')
img_strings = g.read()
img_strings = img_strings.split('\n')
img_strings = img_strings[:-1]
imgs = []
for strng in img_strings:
	split_strng = strng.split(' ')
	num = np.array([int(float(d)) for d in split_strng])
	imer = num.reshape(31,31)
	new_imer = imer[2:30, 2:30]
	new_num = new_imer.reshape(28*28,)
	imgs.append(new_num)

images = np.array(imgs)
print(images.shape)

for k in range(fine_epochs):
	print("Starting epoch: ", k)
	for v in images:
		#print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$-h-$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
		h = np.random.binomial(1,sigmoid(np.dot(weights, v)))
		pos_grad = np.dot(h.reshape(300,1), v.reshape(1,784))
		#for i in range(10):
		#v_prime = np.random.binomial(1, softmax(np.dot(h, weights)))
		v_prime = np.random.binomial(1, sigmoid(np.dot(h, weights)))
		h_prime = np.random.binomial(1, sigmoid(np.dot(weights, v_prime)))
		#v_prime_prime = np.random.binomial(1, sigmoid(np.dot(h_prime, weights)))
		#h_prime_prime = np.random.binomial(1, sigmoid(np.dot(weights, v_prime_prime)))
		#v_prime_prime_prime = np.random.binomial(1, sigmoid(np.dot(h_prime_prime, weights)))
		#h_prime_prime_prime = np.random.binomial(1, sigmoid(np.dot(weights, v_prime_prime_prime)))
		neg_grad = np.dot(h_prime.reshape(300,1), v_prime.reshape(1, 784))
		delta_w = pos_grad - neg_grad
		#print("################################################################")
		#print(delta_w)
		#print("################################################################")
		weights = weights + (learn_rate * delta_w)
		#print(np.max(weights))

for i in range(10):
	broken_img = np.copy(pos_train_x[i*10])
	fig, (ax1, ax2, ax3) = plt.subplots(1,3)
	ax1.imshow(broken_img.reshape(28,28), cmap='gray')
	for j in range(len(broken_img)):
		if(np.random.randint(0, 100) > 96):	#3% of pixels distorted
			if broken_img[j] == 0:
				broken_img[j] = 1
			else:
				broken_img[j] = 0
	h_val = np.random.binomial(1,sigmoid(np.dot(weights, broken_img)))
	#out = softmax(np.dot(h_val, weights))
	out = sigmoid(np.dot(h_val, weights))
	ax2.imshow(broken_img.reshape(28,28), cmap='gray')
	#ax3.imshow(np.random.binomial(1, softmax(np.dot(h_val, weights))).reshape(28,28))
	ax3.imshow(np.random.binomial(1, sigmoid(np.dot(h_val, weights))).reshape(28,28), cmap='gray')
	plt.savefig("test_denoise/img_fine_" + str(i) + ".png")

