import math
import random
import csv
import matplotlib.pyplot as plt

class SOM(object):

	def __init__(self, n_iterations=100, lr0 = None, tlr = None, size0 = None, tsize = None):
		#something
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
			if i%k == 0:
				self.plot_map(inputs, neurons, i)
			self.som_one_step(neurons, inputs, i)
			
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

	def main(self, file):
		noCities, cities = self.read_data(file)
		scaleFactor, scaled = self.normalize(cities)
		neurons = self.init_neurons(len(cities[0]), noCities)
		self.som(neurons = neurons, inputs = scaled, iterations = self.n_iterations, k = 50)
		print(len(neurons))
		#for i in range(len(neurons)):
		#	if (i-2)%5 == 0:
		#		print(neurons[i][0]*scaleFactor[0], neurons[i][1]*scaleFactor[1])

	def read_data(self, file):
		cities = []
		noCities = 0
		with open('./DataSets/'+ file, newline='') as inputfile:
			noCities = sum( 1 for _ in inputfile) - 4
		iterations = 0
		with open('./DataSets/'+ file, newline='') as inputfile:
			for row in csv.reader(inputfile, delimiter=' '):
				if row[0]=='EOF':
					break
				if iterations != 0 and iterations != 1:
					cities.append((float(row[1]),float(row[2])))
				iterations += 1
		return noCities, cities

	def normalize(self, data):
		maxX = max(d[0] for d in data)
		maxY = max(d[1] for d in data)

		return (maxX, maxY),[(d[0]/maxX, d[1]/maxY) for d in data]

	def plot_map(self, inputs, neurons, iteration):		
		plt.clf()
		plt.scatter(*zip(*inputs), color='red', s=20)
		plt.scatter(*zip(*neurons), color='green', s=5)
		plt.plot(*zip(*(neurons+[neurons[0]])), color = 'blue')
		plt.title('Iteration #{:06d}'.format(iteration))
		plt.axis('off')
		plt.draw()
		plt.pause(0.01)




som = SOM(n_iterations=500,lr0 = 0.7, tlr = 0.999, size0 = 29*8/10, tsize = 0.995) 
som.main('test.txt')







