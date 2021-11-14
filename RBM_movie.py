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


for k in range(epochs):
        print("Starting epoch: ", k)
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

np.savetxt("RBM_movies_weights.txt", weights)

for i in range(20):
	h_set = np.zeros(20)
	h_set[i] = 1
	vis_active = np.dot(h_set, weights)
	vis_active_matrix = vis_active.reshape(v.shape[0], 5)
	plt.figure()
	plt.imshow(vis_active_matrix)
	plt.axis('off')
	plt.savefig("RBM_movie_pc_out_ims/component_" + str(i+1) + ".png")
	np.savetxt("RBM_movie_pc_out_ims/component_" + str(i+1) + ".txt", vis_active_matrix)
