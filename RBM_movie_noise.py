import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import gzip

np.set_printoptions(threshold=np.inf)

f = open('data/ACML_Movies.csv', 'r')
movie_strngs = f.read()
movie_strngs = movie_strngs.split('\n')
movie_strngs = movie_strngs[1:]
movie_strngs = movie_strngs[:-1]
ratings = []
for strng in movie_strngs:
        split_strng = strng.split(',')
        rate = np.array([int(d) for d in split_strng])
        ratings.append(rate)

ratings = np.array(ratings)
ratings = ratings[:, 1:]

test_ratings = np.copy(ratings[-11:])
ratings = ratings[:-11]

weights = np.random.uniform(-0.3, 0.3, (20,35*5))

learn_rate = 0.01
epochs = 400

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

loss = np.array([])
div_loss = np.array([])
for k in range(epochs):
        #print("Starting epoch: ", k)
        divergence = np.array([])
        for v in ratings:
            rate_matrix = np.zeros((v.shape[0], 5))
            for i in range(v.shape[0]):
                if v[i] != -1:
                    rate_matrix[i, v[i]-1] = 1
            v_in = rate_matrix.reshape(35*5,)
            h = np.random.binomial(1,sigmoid(np.dot(weights, v_in)))
            pos_grad = np.dot(h.reshape(20,1), v_in.reshape(1,175))
            v_prime = np.zeros((v.shape[0], 5))
            vis_active = np.dot(h, weights)
            vis_active_matrix = vis_active.reshape(v.shape[0], 5)
            for movie_index in range(len(vis_active_matrix)):
                v_prime[movie_index] = np.random.binomial(1, softmax(vis_active_matrix[movie_index]))
            #v_prime = np.random.binomial(1, sigmoid(np.dot(h, weights)))
            for i in range(len(v)):
                if v[i] == -1:
                    v_prime[i] = np.zeros(5)
            h_prime = np.random.binomial(1, sigmoid(np.dot(weights, v_prime.reshape(35*5,))))
            neg_grad = np.dot(h_prime.reshape(20,1), v_prime.reshape(1, 175))
            delta_w = pos_grad - neg_grad
            weights = weights + (learn_rate * delta_w)
            divergence = np.append(divergence, delta_w)
        epoch_div = np.abs(np.mean(divergence))
        div_loss = np.append(div_loss, epoch_div)

        new_ratings = np.array([])
        for v in test_ratings:
            rate_matrix = np.zeros((v.shape[0], 5))
            for i in range(v.shape[0]):
                if np.random.randint(0,100) > 3:
                    rate_matrix[i, v[i]-1] = 1
            v_in = rate_matrix.reshape(35*5,)
            h = np.random.binomial(1,sigmoid(np.dot(weights, v_in)))
            pos_grad = np.dot(h.reshape(20,1), v_in.reshape(1,175))
            v_prime = np.zeros((v.shape[0], 5))
            vis_active = np.dot(h, weights)
            vis_active_matrix = vis_active.reshape(v.shape[0], 5)
            for movie_index in range(len(vis_active_matrix)):
                v_prime[movie_index] = np.random.binomial(1, softmax(vis_active_matrix[movie_index]))
            new_entry = np.argmax(v_prime, axis=1)
            new_ratings = np.append(new_ratings, new_entry + 1)
        new_ratings = new_ratings.reshape(11, 35)
        new_loss = np.mean(np.power(new_ratings - test_ratings, 2))
        loss = np.append(loss, new_loss)
        print("Epoch ", k, " Loss: ", new_loss)

plt.figure()
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("Movies_loss.png")

plt.figure()
plt.plot(div_loss)
plt.xlabel('Epoch')
plt.ylabel('CD Loss')
plt.savefig("Movies_cd_loss.png")

new_ratings = np.array([])
for v in ratings:
	rate_matrix = np.zeros((v.shape[0], 5))
	for i in range(v.shape[0]):
		if v[i] != -1:
			rate_matrix[i, v[i]-1] = 1
	v_in = rate_matrix.reshape(35*5,)
	h = np.random.binomial(1,sigmoid(np.dot(weights, v_in)))
	pos_grad = np.dot(h.reshape(20,1), v_in.reshape(1,175))
	v_prime = np.zeros((v.shape[0], 5))
	vis_active = np.dot(h, weights)
	vis_active_matrix = vis_active.reshape(v.shape[0], 5)
	for movie_index in range(len(vis_active_matrix)):
		v_prime[movie_index] = np.random.binomial(1, softmax(vis_active_matrix[movie_index]))
	new_entry = np.argmax(v_prime, axis=1)
	new_ratings = np.append(new_ratings, new_entry)

new_ratings = new_ratings + 1
new_ratings = new_ratings.reshape(ratings.shape)

np.savetxt("new_ratings.csv", new_ratings.astype(np.int16), delimiter=",")
