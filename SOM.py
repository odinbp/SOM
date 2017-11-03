import math
import random

class SOM(object):

	def __init__(self, d, n, n_iterations=100, lr0 = None, tlr = None, size0 = None, tsize = None):
		#something
		self.d = d #No. input neurons, 2 for TSP
		self.n = n #No. output neurons, equals no. cities maybe?
		if lr0 == None:
			self.lr0 = 0.2
		else:
			self.lr0 = float(lr0)
		if size0 == None:
			self.size0 = max(m,n)/2.0
		else:
			self.size0 = float(size0)
		self.lr = self.lr0
		self.size = self.size0
		self.tlr = tlr
		self.tsize = tsize

		self.n_iterations = n_iterations

		self.sizeChanges = [size0]
		self.lrateChanges = [lr0]

	def init_neurons(self, d, count):
		#return[[0,0],[1,0],[0,1],[1,1]]
		return[[random.uniform(0.0,1.0) for i in range(d)] for j in range(count*8)]


	def discriminant(self, iv, weight):
		d = 0
		for i in range(len(iv)):
			d += (iv[i]-weight[i])**2
		return d

	def neighbourhood(self, distance, size):
		p = 2*size**2
		if p == 0:
			return 0
		return math.exp(-distance**2/p)
		

	def som(self, neurons, inputs, iterations, k):
		
		#You need only compute total distance (D) and show diagrams at every k steps
		
		for i in range(iterations+1):
			self.som_one_step(neurons, inputs, i)
			if i%k == 0:
				#print something
				#plot something
				pass
			
	def som_one_step(self, neurons, inputs, i):

		#Pick a random input vector
		iv = inputs[random.randint(0,len(inputs)-1)]

		#Find winner of cometition
		winningIndex, winner = self.find_winner(iv, neurons)

		for neuronIndex, neuron in enumerate(neurons):
			d = self.circle_distance(len(neurons), neuronIndex, winningIndex)
			tn = self.neighbourhood(d, self.size)
			for i in range(len(neuron)):
				neuron[i] += self.lr*tn*(iv[i]-neuron[i])

		self.size_decay(i)
		self.learning_decay(i)

	def find_winner(self, iv, neurons):
		mi,mn = -100,[999999999999999.0,999999999999999.0]
		for index,neuron in enumerate(neurons):
			if self.discriminant(iv,neuron) < self.discriminant(iv,mn):
				mi = index
				mn = neuron
		return mi,mn

	def circle_distance(self, n, i, j):
		d = abs(i-j)
		return min(d, n-d)

	def size_decay(self, t):
		self.sizeChanges.append(self.size0*math.exp(-t/self.tsize)-self.size)
		self.size = self.size0*math.exp(-t/self.tsize)


	def learning_decay(self, t):
		self.lrateChanges.append(self.lr0*math.exp(-t/self.tlr)-self.lr)
		self.lr = self.lr0*math.exp(-t/self.tlr)

	def main(self, cities):
		neurons = self.init_neurons(self.d, self.n)
		cities = cities
		self.som(neurons = neurons, inputs = cities, iterations = self.n_iterations, k = 100000000000)
		print(neurons)


som = SOM(d=2,n=4,n_iterations=10000,lr0 = 0.7, tlr = 0.99999, size0 = 4, tsize = 0.99999) 
#cities = [(0,0)]
cities = [(0,0),(0,1),(1,0),(1,1)]
som.main(cities)








