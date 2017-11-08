import math
import random
import csv
import matplotlib.pyplot as plt
import mnist_basics as mb

class SOMMnist(object):

	def __init__(self, n_iterations=100, lr0 = None, tlr = None, size0 = None, tsize = None, multiplier = 2):
		#something
		if lr0 == None:
			self.lr0 = 0.2
		else:
			self.lr0 = float(lr0)
		if size0 == None:
			self.size0 = max(m,n)/2.0
		else:
			self.size0 = float(size0*multiplier)
		self.lr = self.lr0
		self.size = self.size0
		self.tlr = tlr
		self.tsize = tsize
		self.multiplier = multiplier

		self.n_iterations = n_iterations

		self.sizeChanges = [size0]
		self.lrateChanges = [lr0]

	def init_neurons(self, d, count):
		weights = [[random.uniform(0.0,1.0) for i in range(d)] for j in range(100)] 
		grid = [[0]*10]*10
		return weights, grid


	def discriminant(self, iv, weight):
		d = 0
		#print(len(iv[0]), "iv") #[[vector], class], må gjøres om til bare [vector]
		#print(len(weight), "weight") #[x,y] -> wtf?
		for i in range(len(iv)):
			d +=  (iv[0][i]-weight[i])**2
		return math.sqrt(d)

	def neighbourhood(self, distance, size):
		p = 2*size**2
		if p == 0:
			return 0
		return math.exp((-distance**2)/p)
		

	def som(self, neurons, inputs, iterations, k):
		
		#You need only compute total distance (D) and show diagrams at every k steps
		#print(neurons, "som") #feil
		for i in range(iterations+1):
			#if i%k == 0:
			#	self.plot_map(inputs, neurons, i)
			self.som_one_step(neurons, inputs, i)
			
	def som_one_step(self, neurons, inputs, iter):

		#Pick a random input vector
		iv = inputs[random.randint(0,len(inputs)-1)]

		#Find winner of cometition
		#print(neurons, "somonestep") #feil
		winningIndex, winner = self.find_winner(iv, neurons)
		iv = iv[0]
		for neuronIndex, neuron in enumerate(neurons):
			d = self.grid_distance(neuronIndex, winningIndex)
			tn = self.neighbourhood(d, self.size)
			for i in range(len(neuron)):
				#print(neuron[i], "neuron[i]")
				#print(iv[i], "iv[i]")
				neuron[i] += self.lr*tn*(iv[i]-neuron[i])

		self.size_decay(iter)
		self.learning_decay(iter)

	def find_winner(self, iv, neurons):
		mi,mn = -100,[255]*786
		for index,neuron in enumerate(neurons):
			#print(neuron, "BOOOOOYEAH") #denne blir feil
			if self.discriminant(iv,neuron) < self.discriminant(iv,mn):
				mi = index
				mn = neuron
		return mi,mn

	#Manhattan distance
	def grid_distance(self, neuronIndex, winningIndex):
		neuronIndex = str(neuronIndex)
		winningIndex = str(winningIndex)

		if len(neuronIndex) == 1:
			neuronIndex+= '0'
		if len(winningIndex) == 1:
			winningIndex+='0'

		NIX = int(neuronIndex[0])
		NIY = int(neuronIndex[1])
		WIX = int(winningIndex[0])
		WIY = int(winningIndex[1])

		return (abs(NIX-WIX) + abs(NIY - WIY))


	def size_decay(self, t):
		self.sizeChanges.append(self.size0*math.exp(-t/self.tsize)-self.size)
		self.size = self.size0*math.exp(-t/self.tsize)
		

	def learning_decay(self, t):
		self.lrateChanges.append(self.lr0*math.exp(-t/self.tlr)-self.lr)
		self.lr = self.lr0*math.exp(-t/self.tlr)

	def main(self):
		noOfImages, images = self.read_data(5)
		scaleFactor, scaled = self.normalize(images)
		#print(len(images[0])) == 2
		#print(images[5], "images")
		neurons, grid = self.init_neurons(len(images[0][0]), noOfImages)
		#print(neurons, "main")
		self.som(neurons = neurons, inputs = scaled, iterations = self.n_iterations, k = 50)
		return grid

	def read_data(self, noOfImages):
		images = []

		for i in range(noOfImages):
			a,b = mb.load_all_flat_cases()
			images.append([a[i],b[i]])

		return noOfImages, images

	def normalize(self, data):
		scale = 255
		for d in range(len(data)):
			for e in range(len(data[d][0])):
				data[d][0][e] = data[d][0][e]/scale 
		return scale,data

som = SOMMnist(n_iterations=200,lr0 = 0.7, tlr = 1000, size0 = 70/10, tsize = 200, multiplier = 10) 
print(som.main())



