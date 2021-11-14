import numpy as np

in_string = input("Please Enter Comma-seperated Movie Ratings \n")
v = np.array(in_string)

print("###############################################################")

weights = np.genfromtxt('RBM_movies_weights.txt')

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
	return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)), axis=0)

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
new_entry = new_entry + 1
new_entry = new_entry.astype(np.int64)

for i in v:
	print str(i)+',',

print ' '

for i in new_entry:
	print str(i)+',',
